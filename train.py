from datasets import Dataset, load_dataset
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType,PromptEncoderConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup, AutoModelForCausalLM,AutoTokenizer,AutoModel,LlamaForCausalLM
from tqdm import tqdm
from datasets import load_dataset
from util import mergeAssPlan, preprocess,mergeConQues
from accelerate import Accelerator,load_checkpoint_in_model
import os
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='gpt2')
parser.add_argument('--tokenizer_name_or_path', type=str, default='gpt2')
parser.add_argument('--number_epoch', type=int, default=50)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--pefted',type=lambda x:x.lower()=='true',default=True)
parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--encoder_hidden_size', type=int, default=512)
parser.add_argument('--dataset_name', type=str, default="track_3")
parser.add_argument('--cache_dir', type=str,default = '/models/')
parser.add_argument('--trainset_dir', type=str, default='/train.jsonl')
parser.add_argument('--validset_dir', type=str, default='/dev.jsonl')
parser.add_argument('--PEFT_virtual_tokens', type=int, default=15)
parser.add_argument('--promp_init_test', type=str, default="Please decide the relation between the Assessment and Plan.")
parser.add_argument('--device_map', type=str, default="auto")
parser.add_argument('--save_dir', type=str, default='/checkpoint/')
parser.add_argument('--save_name', type=str, default='track_3_test')
parser.add_argument('--resume_dir', type=str, default='None')
args = parser.parse_args()

model_name_or_path = args.model_name_or_path
tokenizer_name_or_path = args.tokenizer_name_or_path
save_directory = args.save_dir
text_column = "prompt"
label_column = "relation"
if args.dataset_name != "track_3":
    label_column = "answer"

peft_config = PromptEncoderConfig(
    peft_type="P_TUNING",
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=args.PEFT_virtual_tokens,
    encoder_hidden_size=args.encoder_hidden_size,
)
max_length = args.max_length
lr = args.learning_rate
num_epochs = args.number_epoch
batch_size = args.batch_size
train_fn = args.trainset_dir
val_fn = args.validset_dir


# Load N2C2 track3 dataset
dataset_train = load_dataset("json", data_files= train_fn)
dataset_val = load_dataset("json", data_files= val_fn)
dataset_train['valid'] = dataset_val['train']
dataset = dataset_train
if args.dataset_name!='track_3':
    dataset['train'] = dataset['train'].map(mergeConQues)
    dataset['valid'] = dataset['valid'].map(mergeConQues)
else:
    dataset['train'] = dataset['train'].map(mergeAssPlan)
    dataset['valid'] = dataset['valid'].map(mergeAssPlan)

#####################################################3
# data preprocessing
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path,cache_dir=args.cache_dir)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
processed_datasets = preprocess(tokenizer,dataset,max_length,text_column,label_column)

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["valid"]

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)

eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

########################################################################
# creating model
if model_name_or_path=="meta-llama/Llama-2-7b-hf":
    model = LlamaForCausalLM.from_pretrained(model_name_or_path,cache_dir=args.cache_dir)
else:
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,cache_dir=args.cache_dir)
if args.pefted:
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

# optimizer and lr scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

# begin accelerator
accelerator = Accelerator()
model, optimizer, lr_scheduler = accelerator.prepare(
    model, optimizer, lr_scheduler
)
if args.resume_dir != 'None':
    accelerator.load_state(args.resume_dir)
# training and evaluation
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_length = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to('cuda') for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss = accelerator.gather_for_metrics(loss)
        train_loss += total_loss.sum()
        train_length += total_loss.shape[0]
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    model.eval()
    eval_loss = 0
    eval_length = 0
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to('cuda') for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        total_loss = accelerator.gather_for_metrics(loss)
        eval_loss += total_loss.sum()
        eval_length += total_loss.shape[0]
    train_epoch_loss = train_loss/train_length
    train_ppl = torch.exp(train_epoch_loss)
    eval_epoch_loss = eval_loss/eval_length
    eval_ppl = torch.exp(eval_epoch_loss)
    accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

accelerator.wait_for_everyone()
accelerator.save_state(args.save_dir+args.save_name)

