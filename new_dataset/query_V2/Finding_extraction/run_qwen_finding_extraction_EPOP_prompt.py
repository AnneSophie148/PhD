from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import json
import os
from tqdm import tqdm
import config_llm

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.memory_allocated())

model_name = "Qwen/Qwen3-32B"
THINKING = False

generation_params = {
    "max_new_tokens": 4096,
    "temperature": 0.2
}

model_key = model_name.replace("/", "_")
temperature = generation_params["temperature"]
max_new_tokens = generation_params["max_new_tokens"]


tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
model.eval()

input_device = next(model.parameters()).device

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

Prompt_fname = "EPOP/prompt/epop-relation-extraction-prompt.txt"

with open(Prompt_fname, "r", encoding="utf-8") as f:
    instruction = f.read()

query_rslt = "../references_WOS.tsv"
df_wos = pd.read_csv(query_rslt, sep="\t").fillna("")
abstracts = df_wos["AB"].to_list()
DOIs = df_wos["DI"].to_list()

os.makedirs("output_prompt_epop", exist_ok=True)
output_file = f"output_prompt_epop/relation_prediction_{model_key}.json"

if THINKING:
    output_file = f"output_prompt_epop/relation_prediction_{model_key}_THINKING.json"

processed_dois = set()
results = []

if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
    with open(output_file, "r", encoding="utf-8") as f:
        results = json.load(f)
        processed_dois = {item["DOI"] for item in results}

new_count = 0

for i in tqdm(range(len(abstracts)), desc="Processing abstracts"):
    abstract = abstracts[i]
    doi = DOIs[i]
    if doi in processed_dois:
        continue

    if abstract == "" or abstract in ("None", "none", "n/a"):
        continue

    if doi == "" or doi in ("None", "none", "n/a"):
        continue

    messages = [{"role": "user", "content": instruction + "\nMessage: " + abstract}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=THINKING)
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            **generation_params,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    output_ids = generated_ids[0][inputs["input_ids"].shape[-1]:].tolist()

    decoded = tokenizer.decode(output_ids, skip_special_tokens=True)

    if THINKING and "</think>" in decoded:
        thinking_content, content = decoded.split("</think>", 1)
        thinking_content = thinking_content.strip()
        content = content.strip()
    else:
        thinking_content = ""
        content = decoded.strip()

    result = {
        "DOI": doi,
        "Abstract": abstract,
        "thinking_content": thinking_content,
        "content": content
    }

    results.append(result)
    processed_dois.add(doi)
    new_count += 1

    if new_count % 10 == 0:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(results)} total results to: {output_file}")