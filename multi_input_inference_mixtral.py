from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import json
import evaluate
from evaluate import load
import argparse
from tqdm import tqdm
import torch
from rouge_score import rouge_scorer


def clean_text(text):
    # replace newlines with spaces
    text = text.replace('\n', ' ')
    # replace multiple spaces with a single space 
    text = re.sub(' +', ' ', text)
    return text.strip()


#TODO 
#update this method to load and clean, combine with clean_text method?
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



'''
function for evaluating outputs against ground truth
'''
def evaluate(candidates, references):
    import evaluate
    rouge = evaluate.load('rouge')
 
    #debugging
    #print(candidates)
    #print(references)

    #debugging
    # Check the type of the lists
    #print(f"Type of candidates: {type(candidates)}")
    #print(f"Type of references: {type(references)}")

    # Check the length of the lists
    #print(f"Number of candidates: {len(candidates)}")
    #print(f"Number of references: {len(references)}")

    # Check the type of each element in the lists
    #if candidates and references:  # Ensure lists are not empty
        #print(f"Type of first candidate: {type(candidates[0])}")
        #print(f"Type of first reference: {type(references[0])}")

    # Optionally, print the first few elements to inspect their content
    #print("First few candidates:", candidates[:3])
    #print("First few references:", references[:3])

    results = rouge.compute(predictions=candidates, references=references)
    print(results)


def main():

    parser = argparse.ArgumentParser()
    
    #add arguments for hyperparameters as neccessary 
    parser.add_argument('--input', '-i', help="input file in json format", required=True)
    #parser.add_argument('--temperature', type=float, default=0.4, help="Sampling temperature")
    #condider getting rid of topk arg
    #parser.add_argument('--topk', type=int, default=50, help="Top-k sampling")
    #parser.add_argument('--topp', type=float, default=0.95, help="Top-p (nucleus) sampling")
    
    args = parser.parse_args()

    data = load_json_input(args.input)
    

    model = AutoModelForCausalLM.from_pretrained("/home1/shared/Models/Mixtral/Mixtral-8x7B-Instruct-v0.1", torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("/home1/shared/Models/Mixtral/Mixtral-8x7B-Instruct-v0.1")

    #list to store all outputs
    all_assistant_responses = []
    #list to store all gold truths
    all_references= []

    for entry in tqdm(data, desc="Processing entries"):
        
        user = f"{entry['instruction']} {entry['input']}"

        messages = [
            {"role": "user", "content": user}
        ]

        model_inputs  = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

        generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
        print(tokenizer.batch_decode(generated_ids)[0])

        #add output to all assistant responses list
        assistant_response = tokenizer.batch_decode(generated_ids)[0]
        all_assistant_responses.append(assistant_response)

        #add gold truth to all references list
        reference = entry['output']
        all_references.append(reference)
        
    evaluate(all_assistant_responses, all_references)

if __name__ == "__main__":
    main()