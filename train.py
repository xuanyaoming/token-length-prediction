import torch
import datetime
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DataLoaderConfiguration, InitProcessGroupKwargs
from trainingset import PretrainGenSet, PretrainHeaderSet, DataLoader, header_set_collate_fn, load_from_file, genset_collate_fn
from model_arch import AutoModelForCausalLMWithESTHead, AutoModelForCausalLMWithMultilayerESTHead
from torch.optim.radam import RAdam
from torch.distributions import Normal
from datetime import timedelta


NCCL_TIMEOUT=1000


def generate_gaussian_table(std=10):
    gaussian_table = torch.empty(size=(2048, 2048), dtype=torch.float32, requires_grad=False)
    for i in range(2048):
        dist = Normal(i, std)
        ints = torch.arange(0, 2048)
        log_prob = dist.log_prob(ints)
        probs = torch.exp(log_prob)
        gaussian_table[i, :] = probs
    return gaussian_table


def train_in_tods(batch_size=100, lr=1e-4, set="gripper"):
    file_path_dict = {
        "gripper": "data/gripper/pretrain_dialogs.pkl",
        "IW_1": "data/InstructionWild/datav2/v2_1.pickle",
        "IW_2": "data/InstructionWild/datav2/v2_2.pickle",
        "IW_3": "data/InstructionWild/datav2/v2_2.pickle",
        "IW_4": "data/InstructionWild/datav2/v2_4_train.pickle"
    }
    header_path_dict = {
        "gripper": "model/gripper/random/headers/",
        "IW_1": "model/IW_1/random/headers/",
        "IW_2": "model/IW_2/random/headers/",
        "IW_3": "model/IW_3/random/headers/",
        "IW_4": "model/IW_4/random/headers/"
    }
    model_path = "/mnt/disk4/Yaoming/llama3.x/Llama-3.2-3B-Instruct"
    config = {
        "project_dir": f"est/{set}/single",
        "logging_dir": f"est/{set}/single/log",
        "automatic_checkpoint_naming": True,
        "total_limit": 3
    }
    proj_config = ProjectConfiguration(**config)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=NCCL_TIMEOUT))
    dataloader_config = DataLoaderConfiguration(split_batches=True)
    accelerator = Accelerator(log_with="tensorboard", project_config=proj_config, dataloader_config=dataloader_config, kwargs_handlers=[kwargs])
    
    model: AutoModelForCausalLMWithESTHead = AutoModelForCausalLMWithESTHead.from_pretrained(model_path)
    model.train()
    dialogs = load_from_file(file_path_dict[set])
    gen_set = PretrainGenSet(dialogs, model_path)
    header_set = PretrainHeaderSet(gen_set)
    dl = DataLoader(header_set, batch_size=batch_size, shuffle=True, collate_fn=header_set_collate_fn)

    # freeze unnecessary parameters
    for param in model.parameters():
        param.requires_grad = False
    for param in model.v_head.parameters():
        param.requires_grad = False
    for param in model.est_head.parameters():
        param.requires_grad = True
    # opti = RAdam(model.parameters(), lr=lr)
    opti = RAdam(model.est_head.parameters(), lr=lr)
    model, dl, opti = accelerator.prepare(model, dl, opti)
    timestamp = datetime.datetime.now().strftime("%d:%m:%Y_%H:%M:%S")
    accelerator.init_trackers("est_robot_code_random" + timestamp, config={"batch_size": batch_size, "lr": 1e-4})

    epoch = 0
    while epoch < 100:
        # trained 19 epochs, stops at 42541 step
        if accelerator.is_main_process:
            print(f"start new epoch {epoch}...")
        for idx, data in enumerate(dl):
            # opti.zero_grad()
            opti.zero_grad()
            est_real = data["est"]
            _, _, v, est_logits = model.forward(input_ids=data["prompt"], single_q=True)
            est_loss = F.cross_entropy(est_logits, est_real.squeeze(-1), ignore_index=-1)
            accelerator.backward(est_loss)
            accelerator.log(values={"loss/est": est_loss.item()}, step=idx + epoch * len(dl))
            accelerator.log(values={"epoch": epoch}, step=idx + epoch * len(dl))
            
            opti.step()
            if idx %1000 == 999:
                accelerator.wait_for_everyone()
                accelerator.save_state()
                if accelerator.is_main_process:
                    model.save_headers(header_path_dict[set])
                accelerator.wait_for_everyone()

            if est_loss.item() < 1:  
                accelerator.set_trigger()
            if accelerator.check_trigger():
                break
        accelerator.wait_for_everyone()
        accelerator.save_state()
        if accelerator.check_trigger():
            accelerator.wait_for_everyone()
            accelerator.save_state()
            if accelerator.is_main_process:
                model.save_headers(header_path_dict[set])
            break
        epoch += 1
        
    accelerator.clear()
        

def train_in_teds(batch_size=80, lr=1e-4, set="gripper"):
    file_path_dict = {
        "gripper": "data/gripper/pretrain_dialogs.pkl",
        "IW_1": "data/InstructionWild/datav2/v2_1.pickle",
        "IW_2": "data/InstructionWild/datav2/v2_2.pickle",
        "IW_3": "data/InstructionWild/datav2/v2_2.pickle",
        "IW_4": "data/InstructionWild/datav2/v2_4_train.pickle"
    }
    header_path_dict = {
        "gripper": "model/gripper/fast/headers/",
        "IW_1": "model/IW_1/fast/headers/",
        "IW_2": "model/IW_2/fast/headers/",
        "IW_3": "model/IW_3/fast/headers/",
        "IW_4": "model/IW_4/fast/headers/"
    }
    model_path = "/mnt/disk4/Yaoming/llama3.x/Llama-3.2-3B-Instruct"
    dialogs = load_from_file(file_path_dict[set])
    gen_set = PretrainGenSet(dialogs, model_path)

    config = {
        "project_dir": f"est/{set}",
        "logging_dir": f"est/{set}/log",
        "automatic_checkpoint_naming": True,
        "total_limit": 3
    }
    proj_config = ProjectConfiguration(**config)
    dataloader_config = DataLoaderConfiguration(split_batches=True)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=NCCL_TIMEOUT))
    accelerator = Accelerator(log_with="tensorboard", project_config=proj_config, dataloader_config=dataloader_config, kwargs_handlers=[kwargs])
    model: AutoModelForCausalLMWithESTHead = AutoModelForCausalLMWithESTHead.from_pretrained(model_path)
    model.train()

    dl = DataLoader(gen_set, batch_size=batch_size, shuffle=True, collate_fn=genset_collate_fn)

    # freeze unnecessary parameters
    for param in model.parameters():
        param.requires_grad = False
    for param in model.v_head.parameters():
        param.requires_grad = False
    for param in model.est_head.parameters():
        param.requires_grad = True
    opti = RAdam(model.est_head.parameters(), lr=lr)
    model, dl, opti = accelerator.prepare(model, dl, opti)
    timestamp = datetime.datetime.now().strftime("%d:%m:%Y_%H:%M:%S")
    accelerator.init_trackers("est_robot_code_random" + timestamp, config={"batch_size": batch_size, "lr": lr})
    # accelerator.load_state()

    epoch = 0
    while epoch < 1000:
        # trained 19 epochs, stops at 42541 step
        if accelerator.is_main_process:
            print(f"start new epoch {epoch}...")
        for idx, data in enumerate(dl):
            opti.zero_grad()
            est_real: torch.Tensor = data["est"]
            _, _, v, est_logits = model.forward(input_ids=data["full_dialog"], single_q=False)
            est_loss = F.cross_entropy(est_logits.reshape(-1, 2048), est_real.flatten(), ignore_index=-1)
            
            final_loss = est_loss
            accelerator.backward(final_loss)
            accelerator.log(values={"loss/est": est_loss.item()}, step=idx + epoch * len(dl))
            accelerator.log(values={"epoch": epoch}, step=idx + epoch * len(dl))
            
            opti.step()
        epoch += 1
        if epoch % 1 == 0:
            accelerator.wait_for_everyone()
            # accelerator.save_state()
            if accelerator.is_main_process:
                model.save_headers(header_path_dict[set])
    accelerator.clear()


if __name__ == "__main__":
    train_in_teds(batch_size=60, set="IW_4")
   




