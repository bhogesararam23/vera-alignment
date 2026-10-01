import os
import json
import asyncio
import argparse
import pandas as pd
from datasets import load_dataset, Dataset
from google import genai
from google.genai import types
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a synthetic data generator for an AI alignment project. 
Given a question, generate TWO distinct answers:
1. "chosen": The VERA response. It must explicitly state the core assumptions it relies on, map the boundary conditions (where this answer would become false), and express calibrated uncertainty if the topic is highly debated. Do not be overly verbose.
2. "rejected": The standard RLHF response. It must be highly fluent, sycophantic, and overconfident. It should state its conclusion as absolute fact with zero caveats or limitations.

Respond strictly in valid JSON format using the exact keys: "chosen" and "rejected".
"""

async def generate_vera_pair(client, prompt_text, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = await client.models.generate_content(
                model='gemini-3.5-flash-lite',
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.7,
                )
            )
            data = json.loads(response.text)
            return {
                "prompt": prompt_text,
                "chosen": data.get("chosen", ""),
                "rejected": data.get("rejected", "")
            }
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"failed on prompt: '{prompt_text[:40]}...' | error: {e}")
                return None
            print("rate limited retrying in 15s...")
            await asyncio.sleep(15)

async def main(args):
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    
    if not api_key or not hf_token:
        print("missing keys in env")
        return
        
    client = genai.Client(api_key=api_key).aio
    
    print(f"loading {args.num_samples} samples from truthfulqa...")
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    dataset = dataset.shuffle(seed=42).select(range(min(args.num_samples, len(dataset))))
    all_prompts = dataset['question']
    
    os.makedirs(args.output_dir, exist_ok=True)
    backup_file = os.path.join(args.output_dir, "backup.jsonl")
    
    valid_results = []
    completed_prompts = set()
    
    # RESUME LOGIC: Check for existing backup and load it
    if os.path.exists(backup_file):
        print("Found existing backup. Loading previous progress...")
        with open(backup_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    valid_results.append(data)
                    completed_prompts.add(data["prompt"])
        print(f"Resuming after {len(valid_results)} completed samples.")
        
    # Filter out prompts we have already completed
    prompts_to_do = [p for p in all_prompts if p not in completed_prompts]
    
    if not prompts_to_do:
        print("All prompts are already completed!")
    else:
        for i, prompt in enumerate(prompts_to_do):
            # Print total progress relative to the full batch
            current_total = len(valid_results) + 1
            print(f"processing remaining {i+1}/{len(prompts_to_do)} (Total Progress: {current_total}/{len(all_prompts)})...")
            
            res = await generate_vera_pair(client, prompt)
            if res:
                valid_results.append(res)
                with open(backup_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res) + "\n")
            
            if i < len(prompts_to_do) - 1:
                await asyncio.sleep(15)
    
    print(f"\nFinal dataset size: {len(valid_results)} pairs")
    if not valid_results:
        return

    # Save final parquet and push to Hugging Face
    df = pd.DataFrame(valid_results)
    local_path = os.path.join(args.output_dir, "vera_dataset.parquet")
    df.to_parquet(local_path, index=False)
    
    print(f"pushing to {args.hf_repo}...")
    login(token=hf_token)
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset.push_to_hub(args.hf_repo, private=True)
    print("done ready for kaggle")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--hf_repo", type=str, required=True)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    
    args = parser.parse_args()
    asyncio.run(main(args))