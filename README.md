# VERA (verifiable epistemic reward alignment)

so this is my research project on llm alignment. the whole point of vera is to fix how language models act way too confident even when they are hallucinating or talking about highly debated stuff. we want them to actually map boundaries and show uncertainty.

### the data engine
i built a custom async python pipeline to generate a synthetic dpo (direct preference optimization) dataset. i took the validation split from truthfulqa and used the gemini api to generate two distinct types of responses for every question:
- **chosen:** the vera response (states core assumptions, admits uncertainty, maps where it becomes false)
- **rejected:** the standard rlhf response (super confident, sycophantic, treats everything as absolute fact)

**the engineering part:**
the api rate limits were a nightmare and kept crashing the script, which wiped the data from ram. so i had to build a custom fault-tolerant state persistence system (`backup.jsonl`). it saves the state directly to the drive after every single prompt. if the terminal dies or the api locks me out, the script just reads the backup and resumes exactly where it left off without wasting quota. 

once all 817 pairs were generated, the script automatically compiled the data into a parquet file and pushed the shards to huggingface.

### whats next
now that the data engineering is done, the next phase is training. gonna load this dataset into a kaggle notebook with dual t4 gpus and use unsloth to do the actual dpo fine-tuning on a qwen model.# VERA (verifiable epistemic reward alignment)

so this is my research project on llm alignment. the whole point of vera is to fix how language models act way too confident even when they are hallucinating or talking about highly debated stuff. we want them to actually map boundaries and show uncertainty.

### the data engine
i built a custom async python pipeline to generate a synthetic dpo (direct preference optimization) dataset. i took the validation split from truthfulqa and used the gemini api to generate two distinct types of responses for every question:
- **chosen:** the vera response (states core assumptions, admits uncertainty, maps where it becomes false)
- **rejected:** the standard rlhf response (super confident, sycophantic, treats everything as absolute fact)

**the engineering part:**
the api rate limits were a nightmare and kept crashing the script, which wiped the data from ram. so i had to build a custom fault-tolerant state persistence system (`backup.jsonl`). it saves the state directly to the drive after every single prompt. if the terminal dies or the api locks me out, the script just reads the backup and resumes exactly where it left off without wasting quota. 

once all 817 pairs were generated, the script automatically compiled the data into a parquet file and pushed the shards to huggingface.

### whats next
now that the data engineering is done, the next phase is training. gonna load this dataset into a kaggle notebook with dual t4 gpus and use unsloth to do the actual dpo fine-tuning on a qwen model.