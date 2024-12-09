from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/home1/shared/Models/Mixtral/Mixtral-8x7B-Instruct-v0.1", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("/home1/shared/Models/Mixtral/Mixtral-8x7B-Instruct-v0.1")

messages = [
    {"role": "user", "content": "You are a physician.  Please list as a semicolon separated list the most important problems/diagnoses based on the progress note text below. Only list the problems/diagnoses and nothing else. Be concise. A\/P: Pt is a 71 y.o female with h.o COPD (FEV1 .5L\/35%), h.o diastolic dysfunction, GERD who presents with recurrent dyspnea after discharge from OSH for COPD exacerbation yesterday."},
    #If wanting to test second prompt: add the output from first message to assistant message. 
    #{"role": "assistant", "content": ""},
    #{"role": "user", "content": "Think about the problem more, can you refine this list to the most important problems/diagnoses?"}
]

model_inputs  = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
print(tokenizer.batch_decode(generated_ids)[0])