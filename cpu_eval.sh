#!/bin/bash

#SBATCH --job-name=GPTmodel_text_gen
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email_here
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=512gb
#SBATCH --time=72:00:00
#SBATCH --output=cpu-test.out
#SBATCH --partition=xxxx

model_name_or_path=EleutherAI/pythia-12b
tokenizer_name_or_path=EleutherAI/pythia-12b
cache_dir=/data/.... #to store the pretrained model
max_length=512
learning_rate=1e-4
number_epoch=10
batch_size=4
encoder_hidden_size=1024
PEFT_virtual_tokens=15
pefted=true #minimize parameters
resume_dir=None
dataset_name=blabla ##in case you have different dataset
trainset_dir=/train_dir
validset_dir=/val_dir
testset_dir=/test_dir
save_dir=/checkpoint/ #for model save dir
save_name=${dataset_name}_pythia-12b_epoch${number_epoch}_virtual${PEFT_virtual_tokens}_lr${learning_rate}_encoder${encoder_hidden_size}
checkpoint=${save_dir}${save_name}/
output_dir=./checkpoints/    #for generation text save dir
output_name=${save_name}

DISTRIBUTED_ARGS="--num_processes 1 \
                  --num_machines 1 \
                  --mixed_precision no\
                  --dynamo_backend no"
# if we need to load device via "cpu", add this args to the Distributed args.
TRAINING_ARGS="--model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --max_length $max_length \
    --cache_dir $cache_dir\
    --pefted $pefted \
    --learning_rate $learning_rate \
    --dataset_name $dataset_name \
    --number_epoch $number_epoch \
    --batch_size $batch_size \
    --encoder_hidden_size $encoder_hidden_size \
    --PEFT_virtual_tokens $PEFT_virtual_tokens \
    --trainset_dir $trainset_dir \
    --validset_dir $validset_dir \
    --save_dir $save_dir \
    --save_name $save_name \
    --resume_dir $resume_dir"

TEST_ARGS="--model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --max_length $max_length \
    --pefted $pefted \
    --checkpoint $checkpoint \
    --dataset_name $dataset_name \
    --cache_dir $cache_dir \
    --batch_size $batch_size \
    --encoder_hidden_size $encoder_hidden_size \
    --PEFT_virtual_tokens $PEFT_virtual_tokens \
    --testset_dir $testset_dir \
    --output_dir $output_dir \
    --output_name $output_name"


train_com="accelerate launch $DISTRIBUTED_ARGS -m train"
test_com="accelerate launch $DISTRIBUTED_ARGS -m --cpu eval_cpu"

$train_com $TRAINING_ARGS
$test_com $TEST_ARGS