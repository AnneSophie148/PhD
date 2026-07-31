import json
import os
import re
from typing import Any

import pandas as pd
import torch
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import config_llm


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_generated_text(text):
    text = str(text)

    artefacts = ["<|im_end|>", "<|endoftext|>", "<|end_of_text|>", "<|assistant|>", "<|user|>", "<|system|>"]

    for artefact in artefacts:
        text = text.replace(artefact, "")

    return text.strip()


def extract_json(text: Any):
    text = str(text).strip()
    # 1. Extract content inside ```json ... ```
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)

    if match:
        text = match.group(1)

    # 2. Try parsing the complete text
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return [parsed]

        if isinstance(parsed, list):
            return parsed

    except (json.JSONDecodeError, TypeError):
        pass

    #3. otherwise extract multiple JSON objects
    objects = []
    brace_level = 0
    start = None

    for i, character in enumerate(text):
        if character == "{":
            if brace_level == 0:
                start = i

            brace_level += 1

        elif character == "}":
            brace_level -= 1

            if brace_level == 0 and start is not None:
                chunk = text[start:i + 1]

                try:
                    objects.append(json.loads(chunk))
                except json.JSONDecodeError:
                    pass

                start = None

    if not objects:
        print("\nERROR parsing:")
        print(text)
        raise ValueError("No valid JSON objects found")

    return objects


def normalize_doi(doi):
    if doi is None:
        return ""

    doi = str(doi).strip().lower()
    doi = doi.replace("https://doi.org/", "")
    doi = doi.replace("http://doi.org/", "")
    doi = doi.replace("doi:", "")
    doi = doi.strip()

    return doi


def get_first_model_device(model):
    """
    Get the input device when using device_map='auto'.
    """
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def extract_qwen3_thinking_and_content(output_ids, tokenizer):
    decoded = tokenizer.decode(output_ids, skip_special_tokens=False).strip()

    if "</think>" in decoded:
        thinking_output, llm_output = decoded.split("</think>", 1)
        thinking_output = thinking_output.replace("<think>", "").strip()
    else:
        thinking_output = ""
        llm_output = decoded

    llm_output = clean_generated_text(llm_output)

    return thinking_output, llm_output


def build_qwen3_prompt(tokenizer, messages, enable_thinking=False):
    """
    Build a Qwen3 chat prompt.
    """
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)

    return prompt


extracted_relations_from_abstracts_file = "relation_prediction_Qwen_Qwen3-32B.json"
OUTPUT_GRAPH_PATH = "predicted_contribution_relation.json"
query_rslt = "references_WOS.tsv"

graph_abstract_info = load_json(extracted_relations_from_abstracts_file)
df_wos = pd.read_csv(query_rslt, sep="\t").fillna("")

dois = df_wos["DI"].to_list()
normalize_wos_dois = {normalize_doi(doi): i for i, doi in enumerate(dois) if normalize_doi(doi) != ""}
titles = df_wos["TI"].to_list()

login(token=config_llm.hf_token)

with open("prompt_filter_relation.txt", "r", encoding="utf-8") as f:
    instruction = f.read().strip()

MODEL_NAME = "Qwen/Qwen3-32B"
TOKENIZER_NAME = MODEL_NAME
THINKING = False

generation_params = {
    "max_new_tokens": 4096,
    "temperature": 0.2
}

SAVE_EVERY_N = 5
RESUME_FROM_OUTPUT_IF_EXISTS = True

if RESUME_FROM_OUTPUT_IF_EXISTS and os.path.exists(OUTPUT_GRAPH_PATH):
    print(f"\nResuming from existing output: {OUTPUT_GRAPH_PATH}")
    graph_abstract_info = load_json(OUTPUT_GRAPH_PATH)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=config_llm.hf_token, trust_remote_code=True)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\nLoading model...")


model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=config_llm.hf_token, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True)
model.eval()

print("\nCUDA visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count():", torch.cuda.device_count())

input_device = get_first_model_device(model)
processed_since_save = 0
parsing_error_count = 0

for item in tqdm(graph_abstract_info, desc="Abstract relation predictions"):
    print("\n--------------------------------")
    if not isinstance(item, dict):
        print("\nSkipping invalid item:", item)
        continue

    if "contribution_relations" in item:
        continue

    doi = normalize_doi(item.get("DOI"))
    abstract = str(item.get("Abstract", "")).strip()
    content = item.get("content", "")

    if not doi:
        print("\nSkipping item without DOI")
        continue

    if doi not in normalize_wos_dois:
        print("\nSkipping DOI not found in WOS data:", doi)
        continue

    index = normalize_wos_dois[doi]
    title = str(titles[index]).strip()

    try:
        parsed = extract_json(content)
    except ValueError as error:
        parsing_error_count += 1

        print("\n--------------------------------")
        print("PARSING ERROR")
        print("DOI:", doi)
        print("Error:", error)
        print("Content:")
        print(content)
        print("--------------------------------")

        continue

    relationships = []

    for parsed_object in parsed:
        if isinstance(parsed_object, dict):
            current_relationships = parsed_object.get("relationships", [])

            if isinstance(current_relationships, list):
                relationships.extend(current_relationships)

    if not relationships:
        print("\nNo relationships found for DOI:", doi)
        item["contribution_relations"] = []
        continue

    print("\nDOI:", doi)

    message = (f"Title: {title}\n\n Abstract:\n{abstract}\n\nExtracted relations:\n {json.dumps(relationships, ensure_ascii=False)}")
    print("Message : ")
    print(message)

    messages = [{"role": "system", "content": instruction}, {"role": "user", "content": message}]

    prompt = build_qwen3_prompt(tokenizer=tokenizer, messages=messages, enable_thinking=THINKING)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_params, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

    input_length = inputs["input_ids"].shape[1]
    output_ids = generated_ids[0][input_length:].tolist()

    thinking_output, llm_output = extract_qwen3_thinking_and_content(output_ids=output_ids, tokenizer=tokenizer)

    try:
        contribution_relations = extract_json(llm_output)
    except ValueError as error:
        parsing_error_count += 1

        print("\n--------------------------------")
        print("OUTPUT PARSING ERROR")
        print("DOI:", doi)
        print("Error:", error)
        print("Model output:")
        print(llm_output)
        print("--------------------------------")
        item["contribution_classification_raw_output"] = llm_output
        continue

    item["contribution_relations"] = contribution_relations
    print("\nResult :")
    print(item["contribution_relations"])

    if THINKING:
        item["contribution_thinking_output"] = thinking_output

    processed_since_save += 1

    if processed_since_save >= SAVE_EVERY_N:
        with open(OUTPUT_GRAPH_PATH, "w", encoding="utf-8") as f:
            json.dump(graph_abstract_info, f, ensure_ascii=False, indent=2)

        print(f"\nCheckpoint saved to: {OUTPUT_GRAPH_PATH}")
        processed_since_save = 0

with open(OUTPUT_GRAPH_PATH, "w", encoding="utf-8") as f:
    json.dump(graph_abstract_info, f, ensure_ascii=False, indent=2)

print("\nProcessing completed.")
print("Parsing errors:", parsing_error_count)
print("Output saved to:", OUTPUT_GRAPH_PATH)
