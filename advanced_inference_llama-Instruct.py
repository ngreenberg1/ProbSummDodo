from transformers import pipeline
import torch
import json
import argparse
import re



#TODO 
#update this method to load and clean, combine with clean_text method
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
        
        return data

    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format.")
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")

"""
Function to clean text by removing excessive spaces and newlines
Could be combined in the future with the load json function
"""
def clean_text(text):
    # replace newlines with spaces
    text = text.replace('\n', ' ')
    # replace multiple spaces with a single space 
    text = re.sub(' +', ' ', text)
    return text.strip()

'''
function for evaluating outputs against ground truth
'''
def evaluate(file):
    return null


def main():
    parser = argparse.ArgumentParser()
    
    #add arguments for hyperparameters as neccessary 
    parser.add_argument('--input', '-i', help="input file in json format", required=True)
    
    """
    #recommended default temperature varies from model to model-- may need adjusting
    parser.add_argument('--temperature', type=float, default=0.4, help="Sampling temperature")
    parser.add_argument('--topk', type=int, default=50, help="Top-k sampling")
    parser.add_argument('--topp', type=float, default=0.95, help="Top-p (nucleus) sampling")
    """

    args = parser.parse_args()

    data = load_json_input(args.input)

    ##Add to load data method? 
    for entry in data: 
        entry['instruction'] = clean_text(entry['instruction'])
        entry['input'] = clean_text(entry['input'])

    print("Loaded data:", data[0])

    first_entry = data[0]
    system = first_entry['instruction']
    user = first_entry['input']

    model_id = "/home1/shared/Models/Llama/Meta-Llama-3-8B-Instruct/"


    """
    Inference steps: add initial outputs to file.  then re-prompt.  then add final output.  then evaluate
    by comparing the initial outputs to ground truth, and the final outputs to ground truth.

    need to figure out how to get all of these (initial output, final output, gold truth) in one place in order
    to compare.  
    """
    
    pipe = pipeline(
        "text-generation", 
        model=model_id, 
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
    )
    


    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
     
    print(messages)

    
    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipe(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    assistant_response = outputs[0]["generated_text"][-1]["content"]
    print(assistant_response)
    


if __name__ == "__main__":
    main()