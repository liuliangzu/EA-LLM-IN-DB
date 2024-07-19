#importing library
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig, TextStreamer
from peft import LoraConfig, get_peft_model, PeftModel
from dataset_prepare import *
import torch, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_path = '/home/liuliangzu/phi_11_epoch/'
save_path = "/home/liuliangzu/phi_10_epoch/"
#loading the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
#Lora Hyperparameter
for name, param in model.named_parameters():
    print(f"Layer: {name}")
config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=['q_proj', 'v_proj', 'fc1', 'fc2'],
    lora_dropout=0.05,
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
tokenized_training_data = get_preprocessed_query('train',tokenizer,'train')
#Training hyperparamters
training_arguments = TrainingArguments(
        output_dir="output",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        #EvaluationStrategy = "steps",
        save_strategy="epoch",
        logging_steps=30,
        max_steps=-1,
        num_train_epochs=20
    )
trainer = Trainer(
    model=model,
    train_dataset=tokenized_training_data["input_ids"],
    args=training_arguments,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
#Training
trainer.train()
trainer.model.save_pretrained(save_path)
model.config.use_cache = True
model.eval()
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
