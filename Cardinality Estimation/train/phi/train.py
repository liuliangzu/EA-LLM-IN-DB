#importing library
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig, TextStreamer
from peft import LoraConfig, get_peft_model, PeftModel
from dataset_prepare import *
import torch, os
from new_phi import new_PhiForCausalLM
import copy
import random
import numpy as np
from new_DataCollator import new_DataCollatorForLanguageModeling
from evaluation import evaluation_phi_main
import pandas as pd
from sklearn.model_selection import train_test_split

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def random_split_dataset(data_path="/data/ce/"):
    train_split_file=os.path.join(data_path, "train_hf_set.csv")
    test_split_file = os.path.join(data_path, "test_hf_set.csv")
    if os.path.exists(train_split_file) and os.path.exists(test_split_file):
        train_df = pd.read_csv(train_split_file)
        test_df = pd.read_csv(test_split_file)
    else:        
        data = pd.read_csv(os.path.join(data_path, 'hf_set.csv'))
        train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
        train_df.to_csv(train_split_file)
        test_df.to_csv(test_split_file)
        
    return train_split_file, test_split_file
    


# Set the seed for reproducibility
set_seed(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# model_path = '/home/liuliangzu/phi_11_epoch/'
# save_path = "/home/liuliangzu/phi_10_epoch/"
# model_path="/data2/wuyinjun/ce/phi_11_epoch/"
save_path="/data//phi_out/"
# model_path="microsoft/phi-1.5"
model_path="/data/phi-1_5/"

#loading the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

config = copy.deepcopy(model.config)

del model
torch.cuda.empty_cache()
model = new_PhiForCausalLM(config)
# new_phi_model = new_PhiForCausalLM(model.config)
# model.base_model.model = new_phi_model


tokenizer.pad_token = tokenizer.eos_token
#Lora Hyperparameter
for name, param in model.named_parameters():
    print(f"Layer: {name}")
# config = LoraConfig(
#     r=16,
#     lora_alpha=16,
#     target_modules=['q_proj', 'v_proj', 'fc1', 'fc2'],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM"
# )

config = LoraConfig(
    r=4,   # Rank of the low-rank matrices
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.05,  # Dropout rate
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
#Tokenzing the dataset
'''def tok(sample):
    model_inps =  tokenizer(sample["text"], padding=True)
    return model_inps
data = load_dataset("vicgalle/alpaca-gpt4", split="train")
tokenized_training_data = data.map(tok, batched=True)'''

all_int_token_idx_ls = []
all_int_token_ls = []
other_token_idx_ls = []
for idx in range(tokenizer.vocab_size):
    
    token = tokenizer._tokenizer.id_to_token(idx)
    try:
        token_int = int(token)
        all_int_token_idx_ls.append(idx)
        all_int_token_ls.append(token)
    except:
        if idx == tokenizer.eos_token_id:
            all_int_token_idx_ls.append(idx)
            all_int_token_ls.append(token)
        else:
            other_token_idx_ls.append(idx)
        continue

# other_token_idx_ls.append(tokenizer.eos_token_id)

train_split_file, test_split_file = random_split_dataset()
tokenized_test_data = get_preprocessed_query('test',tokenizer,'test',split_file_name=train_split_file, other_token_idx_ls=other_token_idx_ls)
tokenized_training_data = get_preprocessed_query('train',tokenizer,'train',split_file_name=train_split_file, other_token_idx_ls=other_token_idx_ls)
#Training hyperparamters
training_arguments = TrainingArguments(
        output_dir="/data/phi_out/",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        #EvaluationStrategy = "steps",
        # save_strategy="epoch",
        logging_steps=30,
        max_steps=-1,
        num_train_epochs=40,
        per_device_eval_batch_size=1,
        save_steps=1000,
        save_strategy="steps",
        save_only_model=True,
        bf16=True,
        
    )
trainer = Trainer(
    model=model,
    train_dataset=tokenized_training_data,
    eval_dataset=tokenized_test_data,
    args=training_arguments,
    data_collator=new_DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

model.base_model.model.not_allowed_token_ids = torch.tensor(other_token_idx_ls)
trainer._signature_columns = ["input_ids", "attention_mask", "labels", "output_len", "mask"]
#Training

resume_from_checkpoint="/data/phi_out/checkpoint-143000/"
trainer._load_from_checkpoint(resume_from_checkpoint)

# evaluation_phi_main(trainer.model, tokenizer,tokenized_test_data)

trainer.train()


# trainer.model.save_pretrained(save_path)
# model.config.use_cache = True 
# model.eval()



#Testing the trained model
#def phi_stream(prompt):
#  runtimeFlag = "cuda:0"
#  inputs = tokenizer(f'''Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}.\n\n### Response:\n ''', return_tensors="pt", return_attention_mask=False).to(runtimeFlag)
#  streamer = TextStreamer(tokenizer, skip_prompt= True)
#  _ = model.generate(**inputs, streamer=streamer, max_new_tokens=500)
#phi_stream("what is large language model")
##merging the adpater and pretrained model
#base_model = AutoModelForCausalLM.from_pretrained(
#    base_model,
#    low_cpu_mem_usage=True,
#    return_dict=True,
#    torch_dtype=torch.float16,
#    device_map= {"": 0},
#)
#model = PeftModel.from_pretrained(base_model,new_model)
#model = model.merge_and_unload()
#
#tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
#tokenizer.pad_token = tokenizer.eos_token
#tokenizer.padding_side = "right"
