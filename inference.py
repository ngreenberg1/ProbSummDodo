'''
The goal is to develop a flow where the LLM is prompted twice, and the fine output
is achieved through an iterative process of prompt, output, prompt(refine), final output. 

To start, will work just with Llama as the base model. Would hope to expand to more models 
and compare performance across models.  

Original version will be tested against outputs achieved with Llama models instruction tuned
for the probsumm taks using single prompt, as well as single shot prompting of Llama models.

Further, will compare to finetuned model with iterative prompting, will also think about ways 
to better finetune a model for making diagnoses, rather than just instruction tuning for 
notes and labels.  

Will start with evaluation metric of ROUGE L
'''


"""
An iterative process of prompting an LLM and saving final outputs 
"""


"""
***add imports here***
"""
import argparse 
import json

"""
add arguments here 
"""

"""
Step 1: Input Data Preparation

Validate input format and prepare it for processing
"""
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


def main():
    parser = argparse.ArgumentParser()
    
    #add arguments for hyperparameters as neccessary 
    parser.add_argument('--input', '-i', help="input file in json format", required=True)
    
    """
    parser.add_argument('--model', '-m', help="model directory for inference", required=True)
    #recommended default temperature varies from model to model-- may need adjusting
    parser.add_argument('--temperature', type=float, default=0.4, help="Sampling temperature")
    parser.add_argument('--topk', type=int, default=50, help="Top-k sampling")
    parser.add_argument('--topp', type=float, default=0.95, help="Top-p (nucleus) sampling")
    """

    args = parser.parse_args()

    data = load_json_input(args.input)
    print("Loaded data:", data)

if __name__ == "__main__":
    main()

"""
Step 2: Model Setup 
    * For now, local Pre-trained Llama model 
    * Ensure that the correct version of the model and its tokenizer are loaded 
    * Manage hardware resources to optimize inference speed (GPU or multipe if possible)
"""





"""
Step 3: First LLM interaction: Diagnosis Generation
    *Prompt the model to generate an initial list of diagnoses based on the medical note

    *For each medical note in the input, generate the initial list of diagnoses. The prompt should be structure as:
     "You are a physician.  Please list as a semicolon separated list the most important problems/diagnoses based on 
     the progress note text below.  Only list the problems/diagnoses and nothing else. Be concise. [Medical Note]"
    
    *Validate the model's responses to ensure it aligns with the expectations (a list of diagnoses)

    *Handle potential error in model output (eg, irrelevant responses or incomplete lists).

    *Capture the initial response to include in the next prompting step.
"""

"""
Step 4: Second LLM Interaction: Refine the generated diagnoses

    *Re-prompt the model with a refined query, maintainging the context of the original prompt, original note and the first response. 
    *The second prompt will build on the first, asking the model to refine or enhance the list of diagnoses. 

    *Future Iterations: use models or APIs that support conversational context so that we dont need to repeat the entire context in the second prompt. 

"""

"""
Step 5: Save the outputs 
    *Store the refined list of diagnoses for each input. 
    *consider storing it along with the ground truth for each input for evaluation purposes
"""

"""
Step 6: Evaluation
    * evaluate refined outputs using ROUGE L
    * compare refined outputs to ground truth labels
    * Then, compare the score to scores obtained using other methods.
"""






