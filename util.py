import torch


def mergeAssPlan(row):
    row['prompt'] = '\nAssessment: {} \nPlan: {}'.format(row['assessment'], row['plan'])
    return row
def mergeConQues(row):
    row['prompt'] = '\nContext: {} \nQuestion: {}'.format(row['context'], row['question']) #remember to delete cache after modifying
    return row

def preprocess(tokenizer,dataset,max_length,text_column="prompt",label_column="relation"):
    def preprocess_function(examples):
        batch_size = len(examples[text_column])
        inputs = [f"{text_column} : {x} \nLabel : " for x in examples[text_column]]
        targets = [str(x) for x in examples[label_column]]
        model_inputs = tokenizer(inputs)
        labels = tokenizer(targets, add_special_tokens=False)  # don't add bos token because we concatenate with inputs
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
            # print(i, sample_input_ids, label_input_ids)
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        # print(model_inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )


def preprocess_test(tokenizer,dataset_test,max_length,text_column="prompt"):
    def test_preprocess_function(examples):
        batch_size = len(examples[text_column])
        inputs = [f"{text_column} : {x} \nLabel : " for x in examples[text_column]]
        model_inputs = tokenizer(inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i] 
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        return model_inputs
    return  dataset_test["train"].map(
                test_preprocess_function,
                batched=True,
                num_proc=1,
                remove_columns=dataset_test["train"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
    
def extract_label(dataset_test,batch_size,label_column):
    labels = []
    test_length = len(dataset_test['train'])
    for i in range(0,test_length,batch_size):
        cur_range = i+batch_size if i+batch_size<test_length else test_length
        batch_label = [dataset_test['train'][j][label_column] for j in range(i,cur_range)]
        labels.extend(batch_label)
    return labels