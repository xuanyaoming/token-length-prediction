from transformers import pipeline
from tqdm import trange
from trainingset import load_from_file
from model_arch import AutoModelForCausalLMWithESTHead
from trainingset import PretrainGenSet, PretrainHeaderSet, DataLoader, header_set_collate_fn
from typing import List, Tuple, Dict
import jsonlines
import torch
import pickle
import random
import gc
import tqdm
import pandas as pd


def read_indices_list(file_path):
    """读取索引表的简单函数"""
    with open(file_path, "r") as f:
        indices = list(map(int, f.readlines()))
    return indices


def prepare_instruction_wild_4_file(test_size=200, role="test", avoid_indices: List[int]=[]):
    """
    从instructionwild4中随机抽取test_size条数据，避开avoid_indices里的条目
    """
    model_id = "/mnt/disk4/Yaoming/llama3.x/Llama-3.2-3B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device="cuda:0"
    )
    
    res = []

    indices = []
    for _ in range(test_size):
        while True:
            new_idx = random.randrange(0, 110031)
            if new_idx not in avoid_indices:
                indices.append(new_idx)
                avoid_indices.append(new_idx)
                break

    with jsonlines.open(f"data/InstructionWild/datav2/user_4.jsonl") as reader:
        for j in trange(110031):
            obj = reader.read()
            if j not in indices:
                continue
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": obj["instruction"]},
            ]
            outputs = pipe(
                messages,
                max_new_tokens=2048,
                pad_token_id = 128001 
            )
            messages.append(outputs[0]["generated_text"][-1])
            res.append(messages)
    with open(f"data/InstructionWild/datav2/index_record_{role}.txt", "w") as f:
        for i in indices:
            f.write(str(i)+"\n")
        f.flush()

    if role=="test":
        with open(f"data/InstructionWild/datav2/v2_4_test.pickle", "wb") as f:
            pickle.dump(res, f)
        with jsonlines.open(f"data/InstructionWild/datav2/v2_4_test.jsonl", mode="w") as writer:
            writer.write_all(res)
    elif role == "train":
        with open(f"data/InstructionWild/datav2/v2_4_train.pickle", "wb") as f:
            pickle.dump(res, f)
        with jsonlines.open(f"data/InstructionWild/datav2/v2_4_train.jsonl", mode="w") as writer:
            writer.write_all(res)
        

def visualize_logits(logits, num_ids=2048):
    import matplotlib.pyplot as plt
    token_ids = list(range(num_ids))
    plt.figure(figsize=(10, 5))
    plt.bar(token_ids, logits.to("cpu"), color='blue', edgecolor='black')
    plt.title('Logits for Each Token Value')
    plt.xlabel('Token ID')
    plt.ylabel('Logits')
    plt.xticks(token_ids)
    plt.grid(axis='y')
    plt.savefig("logits.png")


def test_all_acc(base_model_path="/mnt/disk4/Yaoming/llama3.x/Llama-3.2-3B-Instruct", device="cuda:0", ds="all"):
    """
    综合计算基于不同数据集训练结果的准确率，包括zero-shot准确率和训练集准确率
    """
    def test_acc(model: AutoModelForCausalLMWithESTHead, header_path: str, dialog_path: str, max_length=2048) -> Tuple[float, float, float]:
        dialogs = load_from_file(dialog_path)
        model.load_headers(header_path)

        gen_set = PretrainGenSet(dialogs, "/mnt/disk4/Yaoming/llama3.x/Llama-3.2-3B-Instruct")  # tokenizer path
        header_set = PretrainHeaderSet(gen_set, max_length=max_length)

        dl = DataLoader(header_set, batch_size=10, shuffle=True, collate_fn=header_set_collate_fn)
        total_num_correct = 0
        total_num_all = 0
        model.eval()

        est_real_list = []
        est_pred_list = []
        diff_list = []
        diff_ratio_list = []

        with torch.no_grad():
            print("start testing")
            tracker = tqdm.tqdm(total=100)
            for idx, data in enumerate(dl):
                lm_logits, _, _, est_logits = model.forward(input_ids=data["prompt"].to(device))
                print(est_logits.shape)
                visualize_logits(est_logits[0, :])
                est_real: torch.Tensor = data["est"].to(device).squeeze()
                est_pred = torch.argmax(est_logits, dim=-1).to(device).to(est_real.dtype)
                res = torch.eq(est_real, est_pred)
                # print(est_pred, est_real)
                diff = est_pred - est_real
                diff_ratio = diff / est_real

                est_real_list.append(est_real.clone().detach().to("cpu"))
                est_pred_list.append(est_pred.clone().detach().to("cpu"))
                diff_list.append(diff.clone().detach().to("cpu"))
                diff_ratio_list.append(diff_ratio.clone().detach().to("cpu"))

                num_correct = torch.count_nonzero(res).item()
                num_all = est_real.shape[0]
                total_num_correct += num_correct
                total_num_all += num_all
                del est_real
                del est_pred
                del est_logits
                del res
                gc.collect()
                tracker.update(1)
                if idx == 99:
                    # 统计并使用pandas存储csv
                    est_real_summary = torch.cat(est_real_list).numpy()
                    est_pred_summary = torch.cat(est_pred_list).numpy()
                    diff_summary = torch.cat(diff_list).numpy()
                    diff_ratio_summary = torch.cat(diff_ratio_list).numpy()
                    data = {
                        "est_real": est_real_summary,
                        "est_pred": est_pred_summary,
                        "diff(pred-real)": diff_summary,
                        "diff_ratio(diff/real)": diff_ratio_summary
                    }
                    df = pd.DataFrame(data)
                    header_name = header_path.split("/")[1].split(".")[0] + "_" + header_path.split("/")[2]
                    dialog_name = dialog_path.split("/")[-1].split(".")[0]
                    df.to_csv(f"csvs/{header_name}_{dialog_name}.csv")
                    acc_10 = df[(-10 < df["diff(pred-real)"]) & (df["diff(pred-real)"] < 10)].shape[0] / df.shape[0]
                    acc_50 = df[(-50 < df["diff(pred-real)"]) & (df["diff(pred-real)"] < 50)].shape[0] / df.shape[0]
                    acc_100 = df[(-100 < df["diff(pred-real)"]) & (df["diff(pred-real)"] < 100)].shape[0] / df.shape[0]
                    break
            tracker.close()
        return acc_10, acc_50, acc_100

    model = AutoModelForCausalLMWithESTHead.from_pretrained("/mnt/disk4/Yaoming/llama3.x/Llama-3.2-3B-Instruct")
    model = model.to(device)

    # NOTE：通过编辑以下两个列表来设置需要测试的模型和数据集
    header_paths = ["model/gripper/fast/headers/",
                    # "model/gripper/random/headers/",
                    # "model/IW_1/fast/headers/",
                    # "model/IW_1/random/headers/",
                    # "model/IW_2/fast/headers/",
                    # "model/IW_2/random/headers/",
                    # "model/IW_3/fast/headers/",
                    # "model/IW_3/random/headers/",
                    # "model/IW_4/fast/headers/",
                    # "model/IW_4/random/headers/"
                    ]
    dialog_paths = ["data/gripper/pretrain_dialogs.pkl",
                    # "data/InstructionWild/datav2/v2_4_train.pickle",
                    "data/InstructionWild/datav2/v2_1.pickle",
                    # "data/InstructionWild/datav2/v2_2.pickle",
                    # "data/InstructionWild/datav2/v2_3.pickle",
                    "data/InstructionWild/datav2/v2_4_test.pickle"]
    
    res = dict()
    for header_path in header_paths:
        for dialog_path in dialog_paths:
            acc_10, acc_50, acc_100 = test_acc(model, header_path, dialog_path, 2048)   # todo: 别忘了改这里
            print(acc_10, acc_50, acc_100)
            res[header_path] = {dialog_path: (acc_10, acc_50, acc_100)}

    with open("acc_summary.pkl", "wb") as f:
        pickle.dump(res, f)
    
    return res



if __name__ == "__main__":
    # prepare_instruction_wild_4_file(200, "test")
    # indices_to_avoid = read_indices_list("data/InstructionWild/datav2/index_record_test.txt")
    # prepare_instruction_wild_4_file(800, "train", avoid_indices=indices_to_avoid)
    test_all_acc()