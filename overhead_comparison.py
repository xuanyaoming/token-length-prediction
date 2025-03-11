import torch
from trainingset import load_from_file
from model_arch import AutoModelForCausalLMWithESTHead, AutoModelForCausalLMWithMultilayerESTHead
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaModel
from typing import List, Dict
from trainingset import PretrainGenSet, PretrainHeaderSet, DataLoader, header_set_collate_fn
from tqdm import trange
import time


def simple_overhead_comparison(model, est_model, data, devices: List[str], num_tries=1000):
    input_ids = [data["prompt"].to(devices[0]).clone(), data["prompt"].to(devices[1]).clone()]
    print(f"Sequence length: {input_ids[0].shape[1]}")
    with torch.autocast(devices[0], dtype=torch.bfloat16):
        res = model.forward(input_ids=input_ids[0][:, :-1], use_cache=True)
        past_key_values = res.past_key_values
        assert past_key_values is not None
        st = time.time()
        for i in trange(num_tries):
            model.forward(input_ids=input_ids[0], past_key_values=past_key_values, use_cache=True)
        speed1 = num_tries / (time.time()-st)
    # TODO: 回滚transformer和trl版本
    with torch.autocast(devices[1], dtype=torch.bfloat16):
        _, _, _, _, past_key_values = est_model.forward(input_ids=input_ids[1][:, :-1], return_past_key_values=True, use_cache=True)

        st = time.time()
        for j in trange(num_tries):
            lm_logits, _, _, est_logits = est_model.forward(input_ids=input_ids[1], past_key_values=past_key_values, use_cache=True)
        speed2 = num_tries / (time.time()-st)
    return speed1, speed2


if __name__ == "__main__":
    devices = ["cuda:2", "cuda:3"]
    header_path = "model/gripper/random/headers/"
    dialog_path = "data/gripper/pretrain_dialogs.pkl"
    dialogs = load_from_file(dialog_path)
    gen_set = PretrainGenSet(dialogs, "/mnt/disk4/Yaoming/llama3.x/Llama-3.2-3B-Instruct") 
    header_set = PretrainHeaderSet(gen_set, max_length=2048)
    dl = DataLoader(header_set, batch_size=1, shuffle=True, collate_fn=header_set_collate_fn)

    model = AutoModelForCausalLM.from_pretrained("/mnt/disk4/Yaoming/llama3.x/Llama-3.2-3B-Instruct").eval().to(devices[0])
    est_model: AutoModelForCausalLMWithESTHead = AutoModelForCausalLMWithESTHead.from_pretrained("/mnt/disk4/Yaoming/llama3.x/Llama-3.2-3B-Instruct").eval().to(devices[1])
    est_model.load_headers(header_path)
    with torch.no_grad():
        for _ in range(5):
            # NOTE: The first comparison seems always to be an outlier, probably because of the warning message about past_key_values
            speed1, speed2 = simple_overhead_comparison(model, est_model, next(iter(dl)), devices)
            print(f"Original Model speed: {speed1} tokens/s, EST model speed: {speed2} tokens/s")
            print("=====================================================================================")