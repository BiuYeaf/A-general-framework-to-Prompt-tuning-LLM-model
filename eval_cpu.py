from datasets import load_dataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoConfig, AutoModelForCausalLM,default_data_collator,AutoTokenizer,LlamaForCausalLM
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType, PromptEncoderConfig,PeftConfig,PeftModel
from util import mergeAssPlan,preprocess_test,extract_label,mergeConQues
from sklearn.metrics import precision_score,f1_score,recall_score,confusion_matrix
import numpy as np
import os
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='gpt2-medium')
parser.add_argument('--tokenizer_name_or_path', type=str, default='gpt2-medium')
parser.add_argument('--pefted',type=lambda x:x.lower()=='true',default=True)
parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--encoder_hidden_size', type=int, default=1024)
parser.add_argument('--cache_dir', type=str , default = '/cache_dir')
parser.add_argument('--checkpoint_dir', type=str, default='/checkpoint')
parser.add_argument('--dataset_name', type=str, default="track_3")
parser.add_argument('--testset_dir', type=str, default='/test.jsonl')
parser.add_argument('--PEFT_virtual_tokens', type=int, default=15)
parser.add_argument('--promp_init_test', type=str, default="Please decide the relation between the Assessment and Plan.")
parser.add_argument('--output_dir', type=str, default="")
parser.add_argument('--output_name', type=str, default="/checkpoint")
args = parser.parse_args()

## label the classification you would like to classify
batch_size = args.batch_size
text_column = "prompt"
label_column = "relation"
label1_name = "direct"
label2_name = "indirect"
max_new_tokens = 1
if args.dataset_name != "track_3":
    label_column = "answer"
    label1_name = "eligible"
    label2_name = "not eligible"
    max_new_tokens = 2


if args.model_name_or_path=="meta-llama/Llama-2-7b-hf":
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path,cache_dir=args.cache_dir)
else:
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,cache_dir=args.cache_dir)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path,cache_dir=args.cache_dir)

if args.pefted:
    peft_config = PromptEncoderConfig(
        peft_type="P_TUNING",
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.PEFT_virtual_tokens,
        encoder_hidden_size=args.encoder_hidden_size,
    )   
    model = get_peft_model(model, peft_config)

dataset_test = load_dataset("json", data_files= args.testset_dir)
if args.dataset_name != "track_3":
    dataset_test = dataset_test.map(mergeConQues)
else:
    dataset_test = dataset_test.map(mergeAssPlan)
labels = extract_label(dataset_test,batch_size,label_column)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
test_dataset = preprocess_test(tokenizer,dataset_test,args.max_length,text_column)
test_dataloader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

def extract_prediction(example:list)->list:
    pred = []
    for answer in example:
        temp = answer.split("Label : ")[1].strip()
        if not temp:
            temp = 'unknown'
        pred.append(temp)
    return pred

def minDistance(word1:str,word2:str)->bool:
        m,n = len(word1),len(word2)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for j in range(n+1):
            dp[0][j]=j
        for i in range(m+1):
            dp[i][0]=i
        for i in range(m):
            for j in range(n):
                dp[i+1][j+1] = dp[i][j] if word1[i]==word2[j] else min(dp[i][j],dp[i][j+1],dp[i+1][j])+1
        return dp[m][n]

def evaluate_score(labels:list,pred:list,method:str,label1:str,label2:str,max_error=1)->float:
    TP,FP,TN,FN = 0,0,0,0
    m = len(labels)
    for i in range(m):
        if minDistance(labels[i].lower(),label1)<=max_error:
            labels[i] = 0
        elif minDistance(labels[i].lower(),label2)<=max_error:
            labels[i] = 1
        else:
            labels[i] = 2
        if minDistance(pred[i].lower(),label1)<=max_error:
            pred[i] = 0
        elif minDistance(pred[i].lower(),label2)<=max_error:
            pred[i] = 1
        else:
            pred[i] = 2
    precision = precision_score(labels, pred, average=method)
    recall = recall_score(labels, pred, average=method)
    f1 = f1_score(labels, pred, average=method)
    c_matrix = confusion_matrix(labels, pred)
    return precision,recall,f1,c_matrix

accelerator = Accelerator()
model = accelerator.prepare(model)
if args.checkpoint_dir != 'None':
    accelerator.load_state(args.checkpoint_dir)
model = accelerator.unwrap_model(model).to('cpu')
results = []
pred = []
for idx, inputs in enumerate(test_dataloader):
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=args.max_length, eos_token_id=3
        )
        detach_outputs = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
        pred.extend(extract_prediction(detach_outputs))
        results.append(detach_outputs)
labels = np.array(labels)
pred = np.array(pred)

f = file=open(f'{args.output_dir}{args.output_name}.txt','a')
precision,recall,f1,c_matrix = evaluate_score(labels,pred,'micro',label1_name,label2_name)
index = 1
for i in results:
    for j in i:
        accelerator.print(f"case{index}", file=f)
        accelerator.print(j, file=f)
        index+=1
accelerator.print(f"confusion matrix: {c_matrix}",file=f)
accelerator.print(f"Precision:{precision},Recall:{recall},F1-score:{f1}",file=f)
f.close()