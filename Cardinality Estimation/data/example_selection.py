import os
import datetime
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.cluster import KMeans
import pickle
from sklearn.decomposition import PCA
import umap.umap_ as umap
from peft import LoraConfig, get_peft_model,PeftModel
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer,TextStreamer
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

class AttentionPooling(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(embedding_dim, 1)

    def forward(self, embeddings):
        weights = torch.softmax(self.attention(embeddings), dim=1)
        weighted_embeddings = embeddings * weights
        pooled_embedding = torch.sum(weighted_embeddings, dim=1)
        return pooled_embedding

def pool_embeddings(embeddings):
    pooled_embeddings = []
    attention_pooling = AttentionPooling(embedding_dim=embeddings[0].shape[-1]).to(device)
    for embedding in embeddings:
        pooled_embedding = attention_pooling(embedding)
        pooled_embeddings.append(pooled_embedding.cpu().squeeze().detach().numpy())
    return np.array(pooled_embeddings)

def select_and_concat_top_embeddings(embedding_seq, top_k=20):
    norms = torch.stack([torch.norm(embedding) for embedding in embedding_seq])
    default_embedding = torch.zeros(4096).to(device)
    top_k_indices = torch.argsort(norms, descending=True)[:top_k]
    if len(embedding_seq) < top_k:
        padding_indices = torch.arange(len(embedding_seq), top_k).to(device)
        top_k_indices = torch.cat([top_k_indices, padding_indices])
    top_k_embeddings = [embedding_seq[i] if i < len(embedding_seq) else default_embedding for i in top_k_indices]
    #print(top_k_embeddings)
    concatenated_embedding = torch.cat([embedding for embedding in top_k_embeddings])
    return concatenated_embedding

import faiss
res = faiss.StandardGpuResources()
index_flat = faiss.IndexFlatL2(4096)
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
umap_reducer = umap.UMAP(n_components=4096, random_state=42)
model_path = '/mnt/bd/llama-finetune/llama3/Meta-Llama-3-8B-Instruct'
model_deepseek = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,output_hidden_states=True, device_map='auto',trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id = 0
tokenizer.padding_side = 'right'
file_path = '/mnt/bd/llama-finetune/pilotscope/output_train.sql'
with open(file_path, 'r') as f:
    lines_sql = f.readlines()
embedding_list = []
sum_length = 0
for i in range(len(lines_sql)):
    line = lines_sql[i].split('||')[0].replace("SELECT COUNT(*) FROM ","").replace("select count(*) from ","").replace("as","").replace("where","").replace("and","").replace("AS","").replace("WHERE","").replace("AND","")
    inputs = tokenizer.encode(line, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model_deepseek(inputs)
        #[1,length,4096]
        #print(embeddings_0.size())
    last_hidden_state = outputs.hidden_states[-1]
    embedding = last_hidden_state.mean(dim=1)
    embedding_list.append(embedding)
    #gpu_index_flat.add(data_normalized)
    #embedding_list.append(np.array(embeddings_0.cpu().detach().numpy()).astype(np.float32))

#gpu_index_flat.add(pca_result)
for i in range(len(embedding_list)):
    gpu_index_flat.add(embedding_list[i].to(torch.float32).cpu().detach().numpy())

with open(file_path_2, 'r') as f:
    lines_sql_test = f.readlines()
embedding_list_test = []
for i in range(len(lines_sql_test)):
    line = lines_sql_test[i].split('||')[0].replace("SELECT COUNT(*) FROM ","").replace("select count(*) from ","").replace("as","").replace("where","").replace("and","").replace("AS","").replace("WHERE","").replace("AND","")
    inputs = tokenizer.encode(line, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model_deepseek(inputs)
        #[1,length,4096]
        #print(embeddings_0.size())
    last_hidden_state = outputs.hidden_states[-1]
    embedding = last_hidden_state.mean(dim=1)
    #query_vector_normalized = pooled_embeddings / np.linalg.norm(pooled_embeddings)
    k = 10  
    D, I = gpu_index_flat.search(embedding.to(torch.float32).cpu().detach().numpy(), k)
    embedding_list_test.append(I.tolist()[0])
    print(lines_sql[I.tolist()[0][0]])
    print(line)
    print("*"*50)

print(embedding_list_test)
query_list = ""
for i in range(len(embedding_list_test)):
    for j in range(10):
        query_list = query_list + lines_sql[embedding_list_test[i][j]]
output_file_path = './query_retreival_imdb_train.sql'
with open(output_file_path, 'w') as f:
    lines = f.writelines(query_list)

