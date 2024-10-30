
import json
import re
import torch
from datasets import Dataset, load_dataset
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

        print("Loaded data:", data[0])
        
        return data
    
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format.")
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    
##Load cleaned dataset 
train_data = load_json_input("/home1/ngreenberg/DR.Bench/summ_train.json")
test_data = load_json_input("/home1/ngreenberg/DR.Bench/summ_dev.json")

"""
Create custom dataset in chat format
"""
dataset = load_dataset(train_data)
print(dataset)
print(dataset["train"][:5])



"""
for entry in data:
        
#TODO modularize the code by seperating the logic into distinct functions.
#seperate functions for generating initial responses, generating final responses, and evaluating results.
    system = entry['instruction']
    user = entry['input']

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True,add_generation_prompt=False,return_tensors="pt")
    print(tokenizer.decode(tokenized_chat[0]))
    #debugging
    #print(messages)
"""
