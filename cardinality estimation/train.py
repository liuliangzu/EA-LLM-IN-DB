from model import MLP
import torch
import numpy as np
import pandas as pd
import sys
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer


def normalize_labels(labels, min_val=None, max_val=None):
    labels = np.array([np.log(float(l)) for l in labels])
    if min_val is None:
        min_val = labels.min()
        print("min log(label): {}".format(min_val))
    if max_val is None:
        max_val = labels.max()
        print("max log(label): {}".format(max_val))
    labels_norm = (labels - min_val) / (max_val - min_val)
    # Threshold labels
    labels_norm = np.minimum(labels_norm, 1)
    labels_norm = np.maximum(labels_norm, 0)
    return labels_norm, min_val, max_val


def unnormalize_labels(labels_norm, min_val, max_val):
    labels_norm = np.array(labels_norm, dtype=np.float32)
    labels = (labels_norm * (max_val - min_val)) + min_val
    return np.array(np.round(np.exp(labels)), dtype=np.int64)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'
    )

    # 加载数据
    data = pd.read_csv('./hf_set.csv')
    label = data["Cardinal_estimation_results"]
    labels_norm, min_val, max_val = normalize_labels(label,None,None)
    # 加载 LLaMA2 模型和 tokenizer
    model_path = '/home/liuliangzu/llama2-hf/'
    model_llm = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map='auto')
    model_llm.enable_input_require_grads()
    model_llm = get_peft_model(model_llm, peft_config)
    model_llm.print_trainable_parameters()
    model_llm.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'

    # 定义 MLP 模型
    input_dim = model_llm.config.hidden_size
    output_dim = 1
    model_mlp = MLP(input_dim, output_dim).to(device)

    # 确认只优化 peft 模型选中的层
    optimizer_params = []
    for name, param in model_llm.named_parameters():
        if any(layer_name in name for layer_name in peft_config.target_modules):
            optimizer_params.append(param)

    # 定义损失函数和优化器
    loss_fn = torch.nn.MSELoss()
    optimizer1 = torch.optim.AdamW(optimizer_params + list(model_mlp.parameters()), lr=1e-5)
    optimizer2 = torch.optim.AdamW(list(model_mlp.parameters()), lr=1e-4)
    optimizer = optimizer2
    # 训练循环
    model_llm.train()
    model_mlp.train()

    prompt_template = (
        "You are a DBMS, you should use query to finish the Cardinal_estimation task, the table and column is:\n"
        "{table}\n---the join_operater is:\n{operater}\n---the predicate is:\n{predicate}\n---\nAnswer:\n"
    )
    flag = 0
    avg_loss = 0
    for epoch in range(1):  # 设定训练轮数
        for i in range(len(data)):
            model_llm.to(device)
            
            # 生成 prompt
            prompt = prompt_template.format(
                table=data["TableName_or_ColumnName"][i],
                operater=data["join_operater"][i],
                predicate=data["predicate"][i]
            )
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            # 前向传播
            outputs = model_llm(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1][:, 0, :].to(torch.float32)  # 获取最后一层隐藏状态的 [CLS] token
            # 通过 MLP
            y_pred = model_mlp(hidden_states)
            # 获取标签
            y = labels_norm[i]
            y = torch.tensor([[y]], dtype=torch.float32, device=device)
            # 计算损失
            loss = loss_fn(y_pred, y)
            #print(y,y_pred)
            avg_loss = avg_loss + loss.item()/10
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            '''for name, param in model_llm.named_parameters():
                if param.grad is not None:
                    print(f"Layer: {name}, Gradient norm: {param.grad.norm().item()}")

            # 打印 MLP 模型的梯度信息
            for name, param in model_mlp.named_parameters():
                if param.grad is not None:
                    print(f"Layer: {name}, Gradient norm: {param.grad.norm().item()}")'''
            #梯度裁剪
            #torch.nn.utils.clip_grad_norm_(optimizer_params, max_norm=1)
            #torch.nn.utils.clip_grad_norm_(model_mlp.parameters(), max_norm=1)
            optimizer.step()

            # 打印损失
            if i % 10 == 0:
                if avg_loss < 1000:
                    optimizer = optimizer1
                    print("start llm finetune!")
                print(f"Epoch [{epoch+1}], Step [{i+1}/{len(data)}], Loss: {avg_loss}")
                avg_loss = 0
    torch.save(model_mlp, './mlp.pth') 
    model_llm.save_pretrained('./query_encoder/')
if __name__ == "__main__":
    main()
