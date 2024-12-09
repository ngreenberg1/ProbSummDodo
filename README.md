# ProbSumm-Dodo

Experimenting with flow engineering, inspired by results achieved by codium ai solving coding challenges without fine-tuning a model.  https://www.qodo.ai/blog/alphacodium-state-of-the-art-code-generation-for-code-contests/

The goal is to engineer a flow where an LLM is prompted multiple times on each input in order to obtain the best output list of diagnoses. Then, compare the results to results achieved using other methods such as single shot, and various methods of fine-tuning.  

Results will be evaluated using rougeL to compare the final outputs to goldtruth labels.  
