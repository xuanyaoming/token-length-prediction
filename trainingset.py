"""
这里实现数据预处理相关代码

Author: Yaoming Xuan
"""

import torch
import pickle
import jsonlines
from typing import List, Dict
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


PAD_TOKEN_ID = 128002
LEFT_PAD_TOKEN_ID = 128001  # <|end_of_text|>


class PretrainGenSet(Dataset):
    def __init__(self, dialogs: List[List[Dict]], tokenizer_path="/mnt/disk4/Yaoming/llama3.x/Llama-3.2-3B-Instruct"):
        """
        Args:
            - dialogs: dialog history with ABSOLUTE CORRECT code responses.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.data = []
        for dialog in dialogs:
            prompt = self.tokenizer.apply_chat_template(dialog[:-1], tokenize=True, return_tensors="pt")
            answer = self.tokenizer.apply_chat_template(dialog[-1:], tokenize=True, return_tensors="pt")
            full_dialog = torch.cat([prompt, answer], dim=-1)
            loss_mask = torch.zeros_like(full_dialog)
            loss_mask[:, prompt.shape[-1]:] += 1
            est = torch.arange(start=full_dialog.shape[-1], end=0, step=-1, dtype=torch.long).unsqueeze(0)
            real_est = (est + 1) * loss_mask - 1  #默认无效值是-1
            if torch.max(real_est).item() >= 2048:
                continue
            else:
                self.data.append({
                    "prompt": prompt,
                    "answer": answer,
                    "full_dialog": full_dialog,
                    "loss_mask": loss_mask,
                    "est": real_est
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


def genset_collate_fn(data_list):
    max_length = 0
    for data in data_list:
        if max_length < data["full_dialog"].shape[-1]:
            max_length = data["full_dialog"].shape[-1]
    full_dialog_list = []
    loss_mask_list = []
    est_list = []
    for data in data_list:
        full_dialog_list.append(F.pad(data["full_dialog"], [0, max_length-data["full_dialog"].shape[-1]], value=PAD_TOKEN_ID))
        loss_mask_list.append(F.pad(data["loss_mask"], [0, max_length-data["loss_mask"].shape[-1]], value=0))
        est_list.append(F.pad(data["est"], [0, max_length-data["est"].shape[-1]], value=-1))
    return {
        "full_dialog": torch.cat(full_dialog_list, dim=0),
        "loss_mask": torch.cat(loss_mask_list, dim=0),
        "est": torch.cat(est_list, dim=0)
    }


class PretrainHeaderSet(Dataset):
    def __init__(self, parent_set: PretrainGenSet, max_length=2048):
        super().__init__()
        self.parent = parent_set
        self.data = []
        for i in range(len(parent_set)):
            gen_data = parent_set[i]
            prompt_length = gen_data["prompt"].shape[-1]
            for j in range(gen_data["answer"].shape[-1]):
                pos_id = j + prompt_length
                parent_idx = i
                est = gen_data["answer"].shape[-1] - j
                if est >= max_length:
                    continue    
                else:
                    self.data.append({
                        "pos_id": pos_id,
                        "parent_idx": parent_idx,
                        "est": torch.tensor(est).reshape(1, 1)
                    })

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        info = self.data[index]
        return {
            "prompt": self.parent[info["parent_idx"]]["full_dialog"][:, :info["pos_id"]],   # (1, seqlen)
            "est": info["est"]
        }
    

def header_set_collate_fn(data):
    max_prompt_len = 0
    for d in data:
        max_prompt_len = max(max_prompt_len, d["prompt"].shape[-1])
    est_list = []
    prompt_list = []
    for d in data:
        est_list.append(d["est"])
        prompt = F.pad(d["prompt"], (max_prompt_len - d["prompt"].shape[-1], 0), value=LEFT_PAD_TOKEN_ID)
        prompt_list.append(prompt)
    return {
        "prompt": torch.cat(prompt_list, dim=0),
        "est": torch.cat(est_list, dim=0)
    }


def load_from_file(path="data/gripper/pretrain_dialogs.pkl") -> List[List[Dict[str, str]]]:
    """
    load pretrain data
    """
    with open(path, "rb") as f:
        dialogs = pickle.load(f)
    return dialogs


def preprocess_gripper_set_2_jsonl():
    dialogs = load_from_file()
    with jsonlines.open("data/gripper/data.jsonl", mode="w") as writer:
        writer.write_all(dialogs)


def expand_instruction_wild_set():
    """
    这个函数运行很慢，用于处理Instruction-in-wild数据集
    """
    from transformers import pipeline
    from tqdm import trange
    model_id = "/mnt/disk4/Yaoming/llama3.x/Llama-3.2-3B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device="cuda:2"
    )
    length = [429, 248, 200,110031]

    for i in [3]:
        res = []
        print(f"Processing {i}_th file...")
        with jsonlines.open(f"data/InstructionWild/datav2/user_{i}.jsonl") as reader:
            for j in trange(length[i-1]):
                obj = reader.read()
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": obj["instruction"]},
                ]
                outputs = pipe(
                    messages,
                    max_new_tokens=2048,
                    pad_token_id = PAD_TOKEN_ID 
                )
                messages.append(outputs[0]["generated_text"][-1])
                res.append(messages)
    
        with open(f"data/InstructionWild/datav2/v2_{i}.pickle", "wb") as f:
            pickle.dump(res, f)
        with jsonlines.open(f"data/InstructionWild/datav2/v2_{i}.jsonl", mode="w") as writer:
            writer.write_all(res)

              
if __name__ == "__main__":
    expand_instruction_wild_set()

