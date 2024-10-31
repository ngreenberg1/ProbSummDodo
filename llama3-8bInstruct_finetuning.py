import time
import json
import re
import torch
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

PAD_TOKEN = "<|pad|>"

MODEL_NAME = "/home1/shared/Models/Llama/Meta-Llama-3-8B-Instruct/"
NEW_MODEL = "Llama-3-8B-Instruct-ProbSumm"

"""
Model
"""
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    quantization_config=quantization_config,
    device_map="auto",
)
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

print(model.config)
print(tokenizer.bos_token, tokenizer.bos_token_id)
print(tokenizer.eos_token, tokenizer.eos_token_id)
print(tokenizer.pad_token, tokenizer.pad_token_id)
tokenizer.convert_tokens_to_ids(PAD_TOKEN)


"""
Dataset
"""

def clean_text(text):
    # replace newlines with spaces
    text = text.replace('\n', ' ')
    # replace multiple spaces with a single space 
    text = re.sub(' +', ' ', text)
    return text.strip()
    
def load_json_input(file_path):
    """
    Load and validate the JSON input file.

    :param file_path: Path to the JSON file
    :return: List of dictionaries containing 'instruction', 'input', and 'output'
    """
    try: 
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Validate that each object contains the required fields
        for obj in data:
            if not all(key in obj for key in ['instruction', 'input', 'output']):
                raise ValueError("Each object must contain 'instruction', 'input', and 'output' fields.")
        
            obj['instruction'] = clean_text(obj['instruction'])
            obj['input'] = clean_text(obj['input'])
            obj['output'] = clean_text(obj['output'])
 
        print("Loaded data:", data[0])
        
        return data
    
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format.")
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    
##Load cleaned dataset 
train_data = load_json_input("/home1/ngreenberg/DR.Bench/summ_train.json")
validation_data = load_json_input("/home1/ngreenberg/DR.Bench/summ_dev.json")
test_data = load_json_input("/home1/ngreenberg/DR.Bench/summ_test.json")

#debugging
print(train_data[:5])

"""
Create custom dataset in chat format
"""
def create_dataframe(data):
    rows = []
    for item in data:
        rows.append(
            {
                "instruction": item["instruction"],
                "input": item["input"],
                "output": item["output"],
            }
        )
    df = pd.DataFrame(rows)
    return df

#debugging
    
train_df = create_dataframe(train_data)
validation_df = create_dataframe(validation_data)
test_df = create_dataframe(test_data)
print(train_df.head())
print(validation_df.head())
print(test_df.head())

#check for null values 
print(train_df.isnull().value_counts())
print(validation_df.isnull().value_counts())
print(test_df.isnull().value_counts())

##NEXT UP
##format example 
def format_example(row: dict):
    prompt = (
        f"""
    {row["input"]}
    """
    )
    messages = [
        {
            "role": "system", "content": "You are a physician.  Please list as a semicolon separated list the most important problems/diagnoses based on the progress note text below. Only list the problems/diagnoses and nothing else. Be concise.",
        },
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": row["output"]}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

train_df["text"] = train_df.apply(format_example, axis=1)
validation_df["text"] = validation_df.apply(format_example, axis=1)
test_df["text"] = test_df.apply(format_example, axis=1)

def count_tokens(row: dict) -> int:
    return len(
        tokenizer(
            row["text"],
            add_special_tokens=True,
            return_attention_mask=False,
        )["input_ids"]
    )

train_df["token_count"] = train_df.apply(count_tokens, axis=1)
validation_df["token_count"] = validation_df.apply(count_tokens, axis=1)
test_df["token_count"] = test_df.apply(count_tokens, axis=1)
#debugging
print(train_df.head())
print(validation_df.head())
print(test_df.head())


#debugging
print(train_df.text.iloc[0])
print(validation_df.text.iloc[0])
print(test_df.text.iloc[0])


#check for inputs greater than 512 tokens 
print(len(train_df[train_df.token_count < 512]), len(train_df), len(train_df[train_df.token_count < 512]) / len(train_df))
print(len(validation_df[validation_df.token_count < 512]), len(validation_df), len(validation_df[validation_df.token_count < 512]) / len(validation_df))
print(len(test_df[test_df.token_count < 512]), len(test_df), len(test_df[test_df.token_count < 512]) / len(test_df))


#save dataframes to json
train_df.to_json("train_data.json", orient="records", lines=True)
validation_df.to_json("validation_data.json", orient="records", lines=True)
test_df.to_json("test_data.json", orient="records", lines=True)

dataset = load_dataset(
    "json",
    data_files={"train": "train_data.json", "validation": "validation_data.json", "test": "test_data.json"},
)

print(dataset)
print(dataset["train"][0]["text"])

"""
Test Original Model
"""

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    return_full_text=False,
)

def create_test_prompt(data_row):
    prompt = (
        f"""
    {
        data_row["input"]}
    """
    )
    messages = [
        {
            "role": "system", 
            "content": "You are a physician.  Please list as a semicolon separated list the most important problems/diagnoses based on the progress note text below. Only list the problems/diagnoses and nothing else. Be concise.",
        },
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

row = dataset["validation"][0]
prompt = create_test_prompt(row)
print(prompt)


##Testing inference##
start_time = time.time()
outputs = pipe(prompt)
response = f"""
answer: {row["output"]}
prediction: {outputs[0]["generated_text"]}
"""
print(response)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time: {:.2f} seconds".format(execution_time))


"""
Train on completions only 
"""
response_template = "<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

examples = [dataset["train"][0]["text"]]
encodings = [tokenizer(e) for e in examples]

dataloader = DataLoader(encodings, collate_fn=collator, batch_size=1)

batch = next(iter(dataloader))
print(batch.keys())
print(batch["labels"])

"""
LoRA Setup
"""
print(model)

lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

OUTPUT_DIR="/home1/ngreenberg/ProbSummDodo/finetuning_experiment"

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    max_seq_length=512,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    eval_strategy="steps",
    eval_steps=0.2,
    save_steps=0.2,
    logging_steps=10,
    learning_rate=1e-4,
    fp16=True,
    save_strategy="steps",
    warmup_ratio=0.1,
    save_total_limit=2,
    lr_scheduler_type="constant",
    save_safetensors=True,
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": False,
    },    
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=collator,
)

trainer.train()

trainer.save_model(NEW_MODEL)


"""
#Load Trained Model
"""
tokenizer = AutoTokenizer.from_pretrained(NEW_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)

model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
model = PeftModel.from_pretrained(model, NEW_MODEL)
model = model.merge_and_unload()

model.save_pretrained(NEW_MODEL, tokenizer=tokenizer, max_shard_size="5GB")


"""
#Testing Inference
"""

dataset = load_dataset(
    "json",
    data_files={"train": "train_data.json", "validation": "validation_data.json", "test": "test_data.json"},
)

MODEL_NAME = "/home1/ngreenberg/ProbSummDodo/finetuned-Dodo/"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=quantization_config, device_map="auto"
)


