from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/home1/shared/Models/Mixtral/Mixtral-8x7B-Instruct-v0.1", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("/home1/shared/Models/Mixtral/Mixtral-8x7B-Instruct-v0.1")


messages = [
    {"role": "user", "content": "You are a physician.  Please list as a semicolon separated list the most important problems/diagnoses based on the progress note text below. Only list the problems/diagnoses and nothing else. Be concise. H/O HYPERKALEMIA (HIGH POTASSIUM, HYPERPOTASSEMIA). H/O HYPERGLYCEMIA CHRONIC OBSTRUCTIVE PULMONARY DISEASE (COPD, BRONCHITIS, EMPHYSEMA) WITH ACUTE EXACERBATION. A 59 year-old man presents with malaise and hypoxia."},
    {"role": "assistant", "content": "Hyperkalemia; Hyperglycemia; Chronic Obstructive Pulmonary Disease (COPD); Acute exacerbation of COPD; Hypoxia."},
    {"role": "user", "content": "Think about the problem more, can you refine this list to the most important problems/diagnoses?"}
]

model_inputs  = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
print(tokenizer.batch_decode(generated_ids)[0])