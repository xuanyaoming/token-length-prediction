"""
The model architecture of our EST header

Author: Yaoming Xuan
"""


from trl import AutoModelForCausalLMWithValueHead
from torchtune.modules import RotaryPositionalEmbeddings
import torch
import os
import torch.nn as nn


class ESTHead(nn.Module):
    r"""
    The ValueHead class implements a head for GPT2 that returns a scalar for each output token.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()

        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder"):
            if config.is_encoder_decoder and hasattr(config, "decoder"):
                if hasattr(config.decoder, "hidden_size"):
                    hidden_size = config.decoder.hidden_size
        
        self.num_heads = kwargs["est_num_heads"]
        self.head_dim = kwargs["est_head_dim"]

        self.rope = RotaryPositionalEmbeddings(self.head_dim)
        self.act = nn.SiLU()
        self.est_query = nn.Parameter(torch.empty(size=(self.num_heads, self.head_dim)))
        self.k_heads = nn.Linear(hidden_size, self.head_dim * self.num_heads, bias=False)
        self.v_heads = nn.Linear(hidden_size, self.head_dim * self.num_heads, bias=False)
        # self.softmax = nn.Softmax(-1)
        self.attn_out = nn.Linear(self.head_dim * self.num_heads, hidden_size, bias=False)
        self.linears = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size * 3, bias=False),    #TODO:正确值3*，现在在做实验看线性层大小的影响
            nn.Linear(hidden_size, hidden_size * 3, bias=False),
            nn.Linear(hidden_size * 3, hidden_size, bias=False)
        ])
        self.summary = nn.Linear(hidden_size, 2048)

        self.softmax = nn.Softmax(-1)
        self.norm = nn.LayerNorm(hidden_size)
        self.flatten = nn.Flatten()

    def init_weights(self, std=0.02):
        self.est_query.data.normal_(0, std)
        self.k_heads.weight.data.normal_(0, std)
        self.v_heads.weight.data.normal_(0, std)
        self.attn_out.weight.data.normal_(0, std)
        for i in range(len(self.linears)):
            self.linears[i].weight.data.normal_(0, std)
        self.summary.bias.data.normal_(0, std)
        self.summary.weight.data.normal_(0, std)

    def forward(self, hidden_states, single_q = True):
        """
        Args:
            - single_q: 是否扩展q到seqlen，注意这项设置会影响est head的输出形状。如果是True则输出（bsz，2048），否则（bsz, seqlen, 2048)
        """
        output = self.dropout(hidden_states)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.linears[0].weight.dtype:
            output = output.to(self.linears[0].weight.dtype)

        bsz, seqlen, hidden_size = hidden_states.shape
        k = self.k_heads(hidden_states).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        v = self.v_heads(hidden_states).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.rope(k)
        v = self.rope(v).transpose(1, 2)
        k = k.permute(0, 2, 3, 1)   # (bsz, num_heads, head_dim, seqlen)
        q = self.est_query.unsqueeze(0).expand(bsz, self.num_heads, self.head_dim).unsqueeze(-2)    # (bsz, num_heads, 1, head_dim)
        if not single_q:
            q = q.expand(-1, -1, seqlen, -1)    # (bsz, num_heads, seqlen, head_dim)
        attn_weights = self.softmax(q @ k)  # (bsz, num_heads, 1, seqlen) if single_q else (bsz, num_heads, seqlen, seqlen)
        if single_q:
            output = (attn_weights @ v).squeeze(-2).reshape(bsz, -1) # (bsz, num_heads* head_dim)
        else:
            output = (attn_weights @ v).squeeze(-2).reshape(bsz, seqlen, -1)    # (bsz, seqlen, num_heads* head_dim)
        output = self.attn_out(output)  # (bsz, hidden_size) if single_q else (bsz, seqlen, hidden_size)
        output = self.norm(output)

        a = self.linears[0](output)
        b = self.linears[1](output)
        c = a * self.act(b)
        output = self.linears[2](c)
        output = self.norm(output)
        output = self.summary(output)   # (bsz, 2048) if single_q else (bsz, seqlen, 2048)
        
        return output


class AutoModelForCausalLMWithESTHead(AutoModelForCausalLMWithValueHead):
    def __init__(self, pretrained_model, name="gripper", **kwargs):
        super().__init__(pretrained_model, **kwargs)
        est_head_args = {
            "est_num_heads": 16,
            "est_head_dim": 64 # for gripper is 64, other model is 512
        }
        
        self.est_head = ESTHead(self.pretrained_model.config, **est_head_args)
        self.init_weights()

    def init_weights(self):
        r"""
        Initializes the weights of the value head. The default initialization strategy is random.
        Users can pass a different initialization strategy by passing the `v_head_init_strategy` argument
        when calling `.from_pretrained`. Supported strategies are:
        - `normal`: initializes the weights with a normal distribution.

        Args:
            **kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class. These arguments
                can contain the `v_head_init_strategy` argument as well as the `v_head_initializer_range`
                argument.
        """
        initializer_range = 0.2
        
        self.v_head.summary.weight.data.normal_(mean=0.0, std=initializer_range)
        self.v_head.summary.bias.data.zero_()
        self.est_head.init_weights()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        return_past_key_values=False,
        single_q = True,
        **kwargs,
    ):
        r"""
        Applies a forward pass to the wrapped model and returns the logits of the value head.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            return_past_key_values (bool): A flag indicating if the computed hidden-states should be returned.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        """
        kwargs["output_hidden_states"] = True  # v_head and est_head should be applied to hidden states of all layers!
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        if last_hidden_state.device != self.v_head.summary.weight.device:
            last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)

        value = self.v_head(last_hidden_state.detach()).squeeze(-1)
        est = self.est_head(last_hidden_state.detach(), single_q).squeeze(-1) # (1, 2048)

        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        if return_past_key_values:
            return (lm_logits, loss, value, est, base_model_output.past_key_values)
        else:
            return (lm_logits, loss, value, est)

    def save_headers(self, path="my_models/temp/headers/", save_fn=None):
        if not os.path.exists(path):
            os.makedirs(path)
        for s in ["v_head.pt", "est_head.pt"]:
            if os.path.isfile(path + s):
                os.remove(path+s)
        if save_fn is None:
            torch.save(self.v_head.state_dict(), path + "v_head.pt")
            torch.save(self.est_head.state_dict(), path + "est_head.pt")
        else:
            save_fn(self.est_head.state_dict(), path + "est_head.pt")

    def load_headers(self, path="my_models/temp/headers/"):
        self.v_head.load_state_dict(torch.load(path + "v_head.pt", weights_only=True))
        self.est_head.load_state_dict(torch.load(path + "est_head.pt", weights_only=True))


class ESTheadBlock(nn.Module):
    def __init__(self, config, num_heads, head_dim):
        super().__init__()
        summary_dropout_prob = 0.1
        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()

        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder"):
            if config.is_encoder_decoder and hasattr(config, "decoder"):
                if hasattr(config.decoder, "hidden_size"):
                    hidden_size = config.decoder.hidden_size
        
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.rope = RotaryPositionalEmbeddings(self.head_dim)
        self.act = nn.SiLU()
        self.est_query = nn.Parameter(torch.empty(size=(self.num_heads, self.head_dim)))
        self.k_heads = nn.Linear(hidden_size, self.head_dim * self.num_heads, bias=False)
        self.v_heads = nn.Linear(hidden_size, self.head_dim * self.num_heads, bias=False)
        # self.softmax = nn.Softmax(-1)
        self.attn_out = nn.Linear(self.head_dim * self.num_heads, hidden_size, bias=False)
        self.linears = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size * 3, bias=False),    #TODO:正确值3*，现在在做实验看线性层大小的影响
            nn.Linear(hidden_size, hidden_size * 3, bias=False),
            nn.Linear(hidden_size * 3, hidden_size, bias=False)
        ])
        
        self.softmax = nn.Softmax(-1)
        self.norm = nn.LayerNorm(hidden_size)
        self.flatten = nn.Flatten()

    def init_weights(self, std=0.02):
        self.est_query.data.normal_(0, std)
        self.k_heads.weight.data.normal_(0, std)
        self.v_heads.weight.data.normal_(0, std)
        self.attn_out.weight.data.normal_(0, std)
        for i in range(len(self.linears)):
            self.linears[i].weight.data.normal_(0, std)

    def forward(self, hidden_states, single_q = False):
        """
        Args:
            - single_q: 是否扩展q到seqlen，注意这项设置会影响est head的输出形状。如果是True则输出（bsz，2048），否则（bsz, seqlen, 2048)
        """
        output = self.dropout(hidden_states)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.linears[0].weight.dtype:
            output = output.to(self.linears[0].weight.dtype)

        bsz, seqlen, hidden_size = hidden_states.shape
        k = self.k_heads(hidden_states).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        v = self.v_heads(hidden_states).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.rope(k)
        v = self.rope(v).transpose(1, 2)
        k = k.permute(0, 2, 3, 1)   # (bsz, num_heads, head_dim, seqlen)
        q = self.est_query.unsqueeze(0).expand(bsz, self.num_heads, self.head_dim).unsqueeze(-2)    # (bsz, num_heads, 1, head_dim)
        if not single_q:
            q = q.expand(-1, -1, seqlen, -1)    # (bsz, num_heads, seqlen, head_dim)
        attn_weights = self.softmax(q @ k)  # (bsz, num_heads, 1, seqlen) if single_q else (bsz, num_heads, seqlen, seqlen)
        if single_q:
            output = (attn_weights @ v).squeeze(-2).reshape(bsz, -1) # (bsz, num_heads* head_dim)
        else:
            output = (attn_weights @ v).squeeze(-2).reshape(bsz, seqlen, -1)    # (bsz, seqlen, num_heads* head_dim)
        output = self.attn_out(output)  # (bsz, hidden_size) if single_q else (bsz, seqlen, hidden_size)
        output = self.norm(output)

        a = self.linears[0](output)
        b = self.linears[1](output)
        c = a * self.act(b)
        output = self.linears[2](c)
        output = self.norm(output)
        
        return output


#################################################################################################
##########  以下代码实现了多级QaT模型嵌套的功能，在实践中我们发现效果与单层差不多  ####################
#################################################################################################


class MultiLayerESThead(nn.Module):
    def __init__(self, config, num_layers = 2):
        super().__init__()
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        self.summary = nn.Linear(hidden_size, 2048)
        self.blocks = nn.ModuleList([ESTheadBlock(config, 16, 512) for _ in range(num_layers)])

    def init_weights(self, std=0.02):
        self.summary.weight.data.normal_(0, std)
        for layer in self.blocks:
            layer.init_weights(std)
    
    def forward(self, hidden_states, single_q=True):
        for layer_id in range(len(self.blocks)):
            layer = self.blocks[layer_id]
            if not layer_id == (len(self.blocks) - 1):
                hidden_states = layer(hidden_states, False)
            else:
                hidden_states = layer(hidden_states, single_q)
        if single_q:
            return self.summary(hidden_states)
        else:
            return self.summary(hidden_states)[:, -1, :]


class AutoModelForCausalLMWithMultilayerESTHead(AutoModelForCausalLMWithValueHead):
    def __init__(self, pretrained_model, name="gripper", **kwargs):
        super().__init__(pretrained_model, **kwargs)
        est_head_dim = 64 if name=="gripper" else 512
        est_head_args = {
            "est_num_heads": 16,
            "est_head_dim": 512 # for gripper is 64, other model is 512
        }
        
        self.est_head = MultiLayerESThead(self.pretrained_model.config, num_layers=3)
        self.init_weights()

    def init_weights(self):
        r"""
        Initializes the weights of the value head. The default initialization strategy is random.
        Users can pass a different initialization strategy by passing the `v_head_init_strategy` argument
        when calling `.from_pretrained`. Supported strategies are:
        - `normal`: initializes the weights with a normal distribution.

        Args:
            **kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class. These arguments
                can contain the `v_head_init_strategy` argument as well as the `v_head_initializer_range`
                argument.
        """
        initializer_range = 0.02
        
        self.v_head.summary.weight.data.normal_(mean=0.0, std=initializer_range)
        self.v_head.summary.bias.data.zero_()
        self.est_head.init_weights()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        return_past_key_values=False,
        single_q = True,
        **kwargs,
    ):
        r"""
        Applies a forward pass to the wrapped model and returns the logits of the value head.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            return_past_key_values (bool): A flag indicating if the computed hidden-states should be returned.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        """
        kwargs["output_hidden_states"] = True  # v_head and est_head should be applied to hidden states of all layers!
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        if last_hidden_state.device != self.v_head.summary.weight.device:
            last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)

        value = self.v_head(last_hidden_state.detach()).squeeze(-1)
        est = self.est_head(last_hidden_state.detach(), single_q).squeeze(-1) # (1, 2048)

        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        if return_past_key_values:
            return (lm_logits, loss, value, est, base_model_output.past_key_values)
        else:
            return (lm_logits, loss, value, est)

    def save_headers(self, path="my_models/temp/headers/", save_fn=None):
        if not os.path.exists(path):
            os.makedirs(path)
        for s in ["v_head.pt", "est_head.pt"]:
            if os.path.isfile(path + s):
                os.remove(path+s)
        if save_fn is None:
            torch.save(self.v_head.state_dict(), path + "v_head.pt")
            torch.save(self.est_head.state_dict(), path + "est_head.pt")
        else:
            save_fn(self.est_head.state_dict(), path + "est_head.pt")

    def load_headers(self, path="my_models/temp/headers/"):
        self.v_head.load_state_dict(torch.load(path + "v_head.pt", weights_only=True))
        self.est_head.load_state_dict(torch.load(path + "est_head.pt", weights_only=True))


if __name__ == "__main__":
    from trainingset import PretrainGenSet, load_from_file
    model = AutoModelForCausalLMWithESTHead.from_pretrained("/mnt/disk4/Yaoming/llama3.x/Llama-3.2-3B-Instruct").to("cuda:2")
    model.train()
    dialogs = load_from_file()
    ds = PretrainGenSet(dialogs)
 
    data = ds[0]
    lm_logits, loss, value, est = model.forward(input_ids=data["full_dialog"].to("cuda:2"), single_q = False)
    print(lm_logits.shape, value.shape, est.shape, data["est"].shape, data["full_dialog"].shape)
    