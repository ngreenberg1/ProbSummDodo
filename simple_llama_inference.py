from transformers import pipeline
import torch

model_id = "/home1/shared/Models/Llama/Meta-Llama-3-8B-Instruct/"

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

messages = [
    {"role": "system", "content": "You are a physician.  Please list as a semicolon separated list the most important problems/diagnoses based on the progress note text below. Only list the problems/diagnoses and nothing else. Be concise."},
    {"role": "user", "content": "H/O HYPERKALEMIA (HIGH POTASSIUM, HYPERPOTASSEMIA). H/O HYPERGLYCEMIA CHRONIC OBSTRUCTIVE PULMONARY DISEASE (COPD, BRONCHITIS, EMPHYSEMA) WITH ACUTE EXACERBATION. A 59 year-old man presents with malaise and hypoxia."},
    {"role": "assistant", "content": "Hyperkalemia; Hyperglycemia; Chronic Obstructive Pulmonary Disease (COPD); Acute exacerbation of COPD; Hypoxia."},
    {"role": "user", "content": "Think about the problem more, can you refine this list to the most important problems/diagnoses?"}
]

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
