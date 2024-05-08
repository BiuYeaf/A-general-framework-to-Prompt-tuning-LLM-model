#!/bin/bash

#SBATCH --wait-all-nodes=1
#SBATCH --job-name=GPTmodel_text_gen
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xxx(your email)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=8
#SBATCH --mem=2000gb
#SBATCH --time=5-00:00:00
#SBATCH --output=./output.out
#SBATCH --partition=gpu
#SBATCH --exclusive

model_name_or_path=epfl-llm/meditron-7b
tokenizer_name_or_path=epfl-llm/meditron-7b
cache_dir=/models #to store the pretrained model
max_length=512
learning_rate=1e-4
number_epoch=1
batch_size=2
encoder_hidden_size=512
PEFT_virtual_tokens=15
pefted=true #minimize parameters
resume_dir=None
trainset_dir=/train_set
validset_dir=/val_set
testset_dir=/train_set
dataset_name=ADRD
save_dir=/checkpoint/ #for model save dir
save_name=meditron-7b
checkpoint=${save_dir}${save_name}/
output_dir=./ADRD_result/    #for generation text save dir
output_name=${save_name}

DISTRIBUTED_ARGS="--num_processes $SLURM_GPUS_PER_TASK \
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
test_com="accelerate launch $DISTRIBUTED_ARGS -m eval"

$train_com $TRAINING_ARGS
$test_com $TEST_ARGS