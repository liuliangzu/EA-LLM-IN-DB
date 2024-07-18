import os
import datetime

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model.llm_model import ScaleEmbedding,FilterEmbedding
from utils.model.dataset import load_dataset_features, make_feature_datasets, make_train_feature_dataloaders,load_query_workloads
from utils.model.padding import features_padding
from utils.model.qerror import get_qerror
from utils.model.args import get_args_finetune

from peft import LoraConfig, get_peft_model,PeftModel
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer,TextStreamer
import wandb
wandb.init(project="deepseek", name="deepseek_with_emb")
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

prompt_startword = (
    "You are a DBMS, you should use query to finish the Cardinal_estimation task, I will give you the query: \n"
)
prompt_ds = ("The information of the query may help you is :\n")
prompt_pg = ("\nThe estimate given by the database standard optimizer is: {}\n")
prompt_answer = ("You should give you answer as a number :\n")

args = get_args_finetune()
print(args)

# TRAIN_LIST = ['genome'] # choose from ['imdb','stats','ergastf1','genome']
TRAIN_LIST = ['imdb']

current_dir = os.path.dirname(os.path.abspath(__file__))

train_data, train_labels, train_pg_est_cards, \
train_n_join_cols, train_n_fanouts, train_n_tables, train_n_filter_cols = load_dataset_features(bin_size=args.bin_size, dataset_list=TRAIN_LIST, train_or_test='train', usage='finetune')

max_n_join_col, max_n_fanout, max_n_table, max_n_filter_col = max(train_n_join_cols), max(train_n_fanouts), max(train_n_tables), max(train_n_filter_cols)
train_data, train_padding_masks = features_padding(args.bin_size, args.table_dim, args.filter_dim,
                                                   train_data, train_n_join_cols, train_n_fanouts, train_n_tables, train_n_filter_cols,
                                                   max_n_join_col, max_n_fanout, max_n_table, max_n_filter_col)
print("finetune dataset padding done!!")
train_dataset = make_feature_datasets(train_data, train_labels, train_pg_est_cards, train_padding_masks,
                                      train_n_join_cols, train_n_fanouts, train_n_tables, train_n_filter_cols,
                                      train_or_test='train')
train_loader = make_train_feature_dataloaders(train_dataset, 1)
work_load_path ='/mnt/bd/llama-finetune/llama3/PRICE/datas/workloads/finetune/imdb/workloads.sql'
work_load_loader = load_query_workloads(work_load_path)
query_list = []
pg_est_list = []
label_list = []
for i in range(len(work_load_loader)):
    workload_list = work_load_loader[i].split("||")
    query_list.append(workload_list[0])
    pg_est_list.append(workload_list[2])
    label_list.append(workload_list[1])
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# peft config
peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=['q_proj','v_proj','k_proj','o_proj','gate_proj','up_proj','down_proj'],
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'
    )
# our model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_scale_emb = ScaleEmbedding(n_join_col=max_n_join_col, n_fanout=max_n_fanout,hist_dim=args.bin_size,n_embd=4096).to(device).to(torch.bfloat16)
model_filter_emb = FilterEmbedding(n_join_col=max_n_join_col, n_fanout=max_n_fanout, n_table=max_n_table, n_filter_col=max_n_filter_col,hist_dim=args.bin_size, table_dim=args.table_dim, filter_dim=args.filter_dim,n_embd=4096).to(device).to(torch.bfloat16)
model_path = '/mnt/bd/llama-finetune/llama3/Meta-Llama-3-8B-Instruct'
model_deepseek = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto',trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model_deepseek.enable_input_require_grads()
model_deepseek = get_peft_model(model_deepseek, peft_config)
model_deepseek.print_trainable_parameters()
model_deepseek.config.use_cache = True
tokenizer.pad_token_id = tokenizer.eos_token_id = 0
tokenizer.padding_side = 'right'
optimizer_params = []
for name, param in model_deepseek.named_parameters():
    '''print(name)'''
    if any(layer_name in name for layer_name in peft_config.target_modules):
        optimizer_params.append(param)

#optimizer = torch.optim.AdamW(list(model_scale_emb.parameters()) + list(model_filter_emb.parameters())+ optimizer_params)
optimizer = torch.optim.AdamW(list(model_scale_emb.parameters()) + list(model_filter_emb.parameters())+optimizer_params)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=5000, epochs=10, pct_start=0.3, div_factor=2, anneal_strategy='cos', verbose=True)

# fintune
for epoch in range(10):
    print('--'*30)
    #model_scale_emb.train()
    #model_filter_emb.train()
    model_deepseek.train()
    train_loss = 0
    all_output, all_label = [], []
    training_loss = 0
    eval_loss = 0
    for i, (feature, label, pg_est_card, padding_mask, n_join_col, n_fanout, n_table, n_filter_col) in enumerate(train_loader):
        if i == 0 or i % 50 != 0 :
            feature = feature.to(torch.bfloat16).to(device)
            '''pg_est_card = pg_est_card.to(torch.float).to(device).view(-1, 1)
            pg_est_card = torch.log(pg_est_card + 1) + 1
            label = torch.log(label.to(torch.float).to(device) + 1) + 1
            label = label.view(1, -1)'''
            input_ids_0 = tokenizer.encode(tokenizer.bos_token + prompt_startword, return_tensors="pt").to(device)
            input_ids_1 = tokenizer.encode(str(query_list[i]), return_tensors="pt").to(device)
            input_ids_2 = tokenizer.encode(prompt_ds, return_tensors="pt").to(device)
            input_ids_3 = tokenizer.encode(prompt_pg, return_tensors="pt").to(device)
            input_ids_4 = tokenizer.encode(str(pg_est_list[i]), return_tensors="pt").to(device)
            input_ids_5 = tokenizer.encode(prompt_answer, return_tensors="pt").to(device)
            label_for_this = tokenizer.encode(str(label_list[i]) + tokenizer.eos_token, return_tensors="pt")
            label_tensor = label_for_this.to(device)
            with torch.no_grad():
                embeddings_0 = model_deepseek.base_model.get_input_embeddings()(input_ids_0)
                embeddings_1 = model_deepseek.base_model.get_input_embeddings()(input_ids_1)
                embeddings_2 = model_deepseek.base_model.get_input_embeddings()(input_ids_2)
                embeddings_3 = model_deepseek.base_model.get_input_embeddings()(input_ids_3)
                embeddings_4 = model_deepseek.base_model.get_input_embeddings()(input_ids_4)
                embeddings_5 = model_deepseek.base_model.get_input_embeddings()(input_ids_5)
                embeddings_label = model_deepseek.base_model.get_input_embeddings()(label_tensor)
            optimizer.zero_grad()
            scale_embedding = model_scale_emb(feature).to(torch.bfloat16)
            filter_embedding = model_filter_emb(feature).to(torch.bfloat16)
            input_embedding = torch.concat([embeddings_0,embeddings_1,embeddings_2,scale_embedding,filter_embedding,embeddings_3,embeddings_4,embeddings_5],dim = 1)
            attention_mask = torch.tensor([0] * embeddings_0.size(1) + [1]*embeddings_1.size(1)+[0]*embeddings_2.size(1)+[1]*(scale_embedding.size(1)+filter_embedding.size(1))+[0]*embeddings_3.size(1)+[1]*embeddings_4.size(1)+[0]*embeddings_5.size(1)+[0]*embeddings_label.size(1)).view(1,-1)
            labels = torch.tensor([0] * input_embedding.size(1) + label_for_this.cpu().numpy().tolist()[0]).view(1,-1)
            input_embedding = torch.concat([input_embedding,embeddings_label],dim=1)

            outputs = model_deepseek(inputs_embeds=input_embedding,attention_mask=attention_mask.to(device),labels=labels.to(device))
            logits = outputs.logits
    
            active_loss = labels != tokenizer.pad_token_id 
            active_logits = logits[active_loss]
            active_targets = labels[active_loss]
            probabilities = torch.softmax(active_logits, dim=-1)
            #print(probabilities,active_targets,tokenizer.decode(active_targets, skip_special_tokens=True))
            num_classes = active_logits.size(1)
            labels_onehot = torch.Tensor.bfloat16(F.one_hot(active_targets, num_classes=num_classes)).to(device)

            # Initialize the loss function
            criterion = nn.CrossEntropyLoss()
            loss = criterion(probabilities, labels_onehot)
            print(loss)
            training_loss = training_loss + loss
            #print("---- Batch {} with the loss : {} ----\n".format(i,loss))
            if i % 10 == 0:
                wandb.log({
                    "train/loss": training_loss/10
                }, step=i)
                training_loss = 0
            wandb.log({
                "train/learning_rate": scheduler.get_last_lr()[0]
            }, step=i)
            loss.backward()
            for name, param in model_scale_emb.named_parameters():
                if param.grad is not None:
                    #print(f"Layer: {name}, Gradient norm: {param.grad.norm().item()}")
                    wandb.log({
                        "train/grad_norm_scale_{}".format(name): param.grad.norm().item()
                    }, step=i)
            for name, param in model_filter_emb.named_parameters():
                if param.grad is not None:
                    #print(f"Layer: {name}, Gradient norm: {param.grad.norm().item()}")
                    wandb.log({
                        "train/grad_norm_filter_{}".format(name): param.grad.norm().item()
                    }, step=i)
            optimizer.step()
        else: 
            scheduler.step()
            feature = feature.to(torch.bfloat16).to(device)
            '''pg_est_card = pg_est_card.to(torch.float).to(device).view(-1, 1)
            pg_est_card = torch.log(pg_est_card + 1) + 1
            label = torch.log(label.to(torch.float).to(device) + 1) + 1
            abel = label.view(1, -1)'''
            input_ids_0 = tokenizer.encode(tokenizer.bos_token + prompt_startword, return_tensors="pt").to(device)
            input_ids_1 = tokenizer.encode(str(query_list[i]), return_tensors="pt").to(device)
            input_ids_2 = tokenizer.encode(prompt_ds, return_tensors="pt").to(device)
            input_ids_3 = tokenizer.encode(prompt_pg, return_tensors="pt").to(device)
            input_ids_4 = tokenizer.encode(str(pg_est_list[i]), return_tensors="pt").to(device)
            input_ids_5 = tokenizer.encode(prompt_answer, return_tensors="pt").to(device)
            label_for_this = tokenizer.encode(str(label_list[i]) + tokenizer.eos_token, return_tensors="pt")
            label_tensor = label_for_this.to(device)
            with torch.no_grad():
                embeddings_0 = model_deepseek.base_model.get_input_embeddings()(input_ids_0)
                embeddings_1 = model_deepseek.base_model.get_input_embeddings()(input_ids_1)
                embeddings_2 = model_deepseek.base_model.get_input_embeddings()(input_ids_2)
                embeddings_3 = model_deepseek.base_model.get_input_embeddings()(input_ids_3)
                embeddings_4 = model_deepseek.base_model.get_input_embeddings()(input_ids_4)
                embeddings_5 = model_deepseek.base_model.get_input_embeddings()(input_ids_5)
                embeddings_label = model_deepseek.base_model.get_input_embeddings()(label_tensor)
            optimizer.zero_grad()
            scale_embedding = model_scale_emb(feature).to(torch.bfloat16)
            filter_embedding = model_filter_emb(feature).to(torch.bfloat16)
            input_embedding = torch.concat([embeddings_0,embeddings_1,embeddings_2,scale_embedding,filter_embedding,embeddings_3,embeddings_4,embeddings_5],dim = 1)
            attention_mask = torch.tensor([0] * embeddings_0.size(1) + [1]*embeddings_1.size(1)+[0]*embeddings_2.size(1)+[1]*(scale_embedding.size(1)+filter_embedding.size(1))+[0]*embeddings_3.size(1)+[1]*embeddings_4.size(1)+[0]*embeddings_5.size(1)+[0]*embeddings_label.size(1)).view(1,-1)
            labels = torch.tensor([-100] * input_embedding.size(1) + label_for_this.cpu().numpy().tolist()[0]).view(1,-1)
            input_embedding = torch.concat([input_embedding,embeddings_label],dim=1)

            outputs = model_deepseek(inputs_embeds=input_embedding,attention_mask=attention_mask.to(device),labels=labels.to(device))
            loss = outputs.loss
            eval_loss = eval_loss + loss
            #print("---- Batch {} with the loss : {} ----\n".format(i,loss))
            if i % 500 == 0:
                wandb.log({
                    "eval/loss": eval_loss/10
                })
                eval_loss = 0
                torch.save(model_scale_emb.state_dict(), f'results/model_scale_emb.pth')
                torch.save(model_filter_emb.state_dict(), f'results/model_filter_emb.pth')
                if i % 1500 == 0:
                    model_deepseek.save_pretrained('./results/llm3_lora/')
                    pass

    #torch.save(model_scale_emb.state_dict(), f'results/model_scale_emb.pth')
    #torch.save(model_filter_emb.state_dict(), f'results/model_filter_emb.pth')
    #model_deepseek.save_pretrained('./results/deepseek_lora/')
    train_loss = train_loss / len(train_loader.dataset)
    print(f"epoch: {epoch}, train loss: {train_loss}")

    #all_output, all_label = np.array(all_output), np.array(all_label)
    #q_error = get_qerror(all_output, all_label, cuda=False, do_scale=True, percentile_list=[30, 50, 80, 90, 95, 99])
    #print('train q-error: 30%:', q_error[0], '  50%:', q_error[1], '  80%:', q_error[2], '  90%:', q_error[3], '  95%:', q_error[4], '  99%:', q_error[5])

print('done!')
#torch.save(model.state_dict(), f'results/finetune_params.pth')
print('save model to results/finetune_params.pth')
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
