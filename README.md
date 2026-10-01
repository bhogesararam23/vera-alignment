# vera (verifiable epistemic reward alignment)

so this is my personal research project on llm alignment. the whole point of vera is to fix how language models act way too confident even when they are hallucinating or talking about highly debated stuff. i wanted to train a model to actually map boundaries and show uncertainty instead of just giving robotic or sycophantic absolute answers.

### 1. the data engine
i built a custom async python pipeline to generate a synthetic dpo (direct preference optimization) dataset. i took the validation split from `truthfulqa/truthful_qa` and used the gemini 3.5 flash lite api to generate two distinct types of responses for every question:
- **chosen:** the vera response (states core assumptions, admits uncertainty, maps where it becomes false)
- **rejected:** the standard rlhf response (super confident, treats everything as absolute fact, zero caveats)

the api rate limits were a total nightmare and kept crashing the script midway. so i engineered a custom fault-tolerant state persistence system (`backup.jsonl`). it saves the state directly to the local drive after every single prompt. if the terminal dies or the api locks me out, the script just reads the backup and resumes exactly where it left off without wasting any quota. 

once all 817 pairs were generated, the script automatically compiled the data into a parquet file and pushed it to huggingface. 
the dataset is public here: [RamBhogesara/vera-dpo-dataset](https://huggingface.co/datasets/RamBhogesara/vera-dpo-dataset)

### 2. dpo fine-tuning
with the dataset done, i moved to training. i set up a kaggle notebook environment using their free t4 gpus. 

i used the `unsloth` library to optimize the memory and did the actual dpo fine-tuning on a `qwen2.5-1.5b-instruct` base model. i had to do some gpu memory wrangling (locking the model to a single gpu to avoid parallel tensor crashes) and injected lora adapters so the whole thing could train efficiently in 4-bit precision.

the training converged beautifully. the rewards/accuracies metric hit 1.000 by step 40, which basically means the model fully learned to prefer my calibrated vera responses over the overconfident ones.

### 3. the final model
after training, i merged the trained lora adapters back into the base qwen model in 16-bit precision and pushed the final weights directly from kaggle. 

the fully aligned, ready-to-use model is live here: [RamBhogesara/vera-qwen-1.5b-dpo](https://huggingface.co/RamBhogesara/vera-qwen-1.5b-dpo)