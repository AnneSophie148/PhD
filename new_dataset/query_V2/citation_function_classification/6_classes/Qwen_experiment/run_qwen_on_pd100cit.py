
import pandas as pd
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import csv
from tqdm import tqdm
import config_llm



def load_processed_indices_from_csv(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()

    try:
        existing_results = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return set()

    if "row_index" not in existing_results.columns:
        raise ValueError(f"The existing output CSV has no 'row_index' column: {csv_path}")

    return set(pd.to_numeric(existing_results["row_index"], errors="coerce").dropna().astype(int).tolist())


def load_PD_data(length_left, length_right):
    def clean_sentence(sentence):
        cleaned_sentence = re.sub(r'<ref[^>]*>', '', sentence)
        cleaned_sentence = re.sub(r'type="[^"]*"\s*target="[^"]*">|type="bibr">', '', cleaned_sentence)
        cleaned_sentence = re.sub(r'</ref>|<ref', '', cleaned_sentence)
        return cleaned_sentence

    def map_labels_to_jurgens(y):
        new_y = []
        jurgens_labels = ['background', 'motivation', 'uses', 'extends', 'compareorcontrast', 'future']
        mapping = {"neutral":"Background", "motivation": "Motivation", "similar":"CompareOrContrast", "usage":"Uses", "cocores":"CompareOrContrast", "basis":"Extends", "weakness":"CompareOrContrast", "future":"Future", "support":"CompareOrContrast", "cocogm":"CompareOrContrast", "cocoxy": "Background"}
        old_labels= ['similar', 'neutral', 'usage', 'cocores', 'motivation', 'basis', 'weakness', 'future', 'support', 'cocogm', 'cocoxy']
        for label in y:
            print("\n")
            print("Label : ")
            print(label)
            if label.lower() not in jurgens_labels:
                mapped_label = mapping.get(label)
                new_y.append(mapped_label)
            else:
                new_y.append(label.lower())

        return new_y
    
    def define_y_100citation(labels):
        y = []
        for i in range(len(labels)):
            label = labels[i].split('|')[0].replace(' ', '').lower()
            y.append(label)
        y = map_labels_to_jurgens(y)
        return y
    
    def load_context_and_citances(df, begining_citation_sentences, end_citation_sentences, length_left, length_right):
        citances = []
        left_context_sentences = []
        right_context_sentences = []
        context_dic_lists = {}

        for n in range(1, length_left + 1):
            context_dic_lists[f"l{n}"] = df[f"l{n}"].tolist()

        for n in range(1, length_right + 1):
            context_dic_lists[f"r{n}"] = df[f"r{n}"].tolist()

        for i in range(len(begining_citation_sentences)):
            beginning = "" if pd.isna(begining_citation_sentences[i]) else str(begining_citation_sentences[i])
            ending = "" if pd.isna(end_citation_sentences[i]) else str(end_citation_sentences[i])

            citance = beginning + " (CITSEG) " + ending
            print("\n___________________")
            print("\nCitance:")
            print(citance)

            citance = clean_sentence(citance)
            citances.append(citance)

            left_context = ""
            right_context = ""

            for n in reversed(range(1, length_left + 1)):
                value = context_dic_lists[f"l{n}"][i]

                if not pd.isna(value):
                    value = str(value).strip()

                    if value and value.lower() != "nan":
                        left_context += value + " "

            left_context = clean_sentence(left_context)
            left_context_sentences.append(left_context)

            for n in range(1, length_right + 1):
                value = context_dic_lists[f"r{n}"][i]

                if not pd.isna(value):
                    value = str(value).strip()

                    if value and value.lower() != "nan":
                        right_context += value + " "

            right_context = clean_sentence(right_context)
            right_context_sentences.append(right_context)

            print("\nLeft context : ")
            print(left_context)

            print("\nRight context : ")
            print(right_context)

        return left_context_sentences, right_context_sentences, citances


    def clean_section_tag(section):
        from html import unescape
        if section is None:
            return ""

        section = str(section)

        # Decode HTML/XML entities if present
        section = unescape(section)

        # Remove XML/HTML tags, e.g. <head ...>, </head>, <p ...>
        section = re.sub(r"<[^>]+>", " ", section)

        # Normalise spaces
        section = re.sub(r"\s+", " ", section).strip()

        return section

    dataset ='PD100cit_jurgens_typlogy.csv'
    df = pd.read_csv(dataset)
    begining_citation_sentences = df["citation_sentence"].astype(str).tolist()
    end_citation_sentences = df["end_citation_sentence"].astype(str).tolist()
    section_pd100cit = df["section"].astype(str).tolist()
    labels = df["annotation_rhetorical_function"].astype(str).tolist()

    section_pd100cit_clean = [clean_section_tag(section) for section in section_pd100cit]

    citation_sections_left_pd100cit = ["" for i in range(len(section_pd100cit_clean))]
    section_position = ["" for i in range(len(section_pd100cit_clean))]

    citation_sequence_y_100citations = define_y_100citation(labels)
    left_context_sentences, right_context_sentences, citances = load_context_and_citances(df, begining_citation_sentences, end_citation_sentences, length_left, length_right)
    

    citation_sequence_x_100citations = [left_context_sentences[i] + " " + citances[i] + " " + right_context_sentences[i] for i in range(len(citances))]
    
    return citation_sequence_x_100citations, citation_sequence_y_100citations, section_pd100cit_clean, citation_sections_left_pd100cit, section_position


def extract_qwen3_thinking_and_content(output_ids, tokenizer):
    decoded_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    if "</think>" in decoded_output:
        thinking_output, llm_output = decoded_output.split("</think>", 1)
        thinking_output = thinking_output.replace("<think>", "").strip()
        return thinking_output, llm_output.strip()

    return "", decoded_output


def append_result_to_csv(result, csv_path):
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    output_fieldnames = [
        "row_index",
        "True_label",
        "section",
        "citance_masked_citseg",
        "model",
        "max_new_tokens",
        "temperature",
        "thinking_enabled",
        "thinking_output",
        "llm_output"
    ]

    with open(csv_path, "a", encoding="utf-8", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=output_fieldnames, quoting=csv.QUOTE_MINIMAL)

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)
        output_handle.flush()

generation_params = {
    "max_new_tokens": 1024,
    "do_sample": True,
    "temperature": 0.2
}

window_context = "3-3"
length_left, length_right = int(window_context.split('-')[0]), int(window_context.split('-')[1])

print("lenght left : ", length_left)
print("lenght right : ", length_right)

prompt_fname = "prompt_6classes.txt"
input_csv = "PD100cit_jurgens_typlogy.csv"
citation_sequence_x_100citations, citation_sequence_y_100citations, section_pd100cit, citation_sections_left_pd100cit, section_position_pd100cit = load_PD_data(length_left, length_right)
print("\n")
print("SIZE TRUE LABEL : ")
print(len(citation_sequence_y_100citations))
HF_TOKEN = config_llm.hf_token
model_name = "Qwen/Qwen3-32B"
tokenizer_name = model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=HF_TOKEN, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

THINKING = True
enable_thinking=THINKING
temperature = generation_params["temperature"]
model_key = model_name.replace("/", "_")
if THINKING:
    output_csv = f"output_csv_PD100cit/PD100cit_6classes_{model_key}_temperature.{temperature}_THINKING.csv"
    generation_params = {
                        "max_new_tokens": 4096,
                        "do_sample": True,
                        "temperature": 0.2,
                        "top_p": 0.9}
else:
    output_csv = f"output_csv_PD100cit/PD100cit_6classes_{model_key}_temperature.{temperature}.csv"


if THINKING:
    output_csv = f"output_csv_PD100cit/PD100cit_6classes_{model_key}_temperature.{temperature}_THINKING.csv"
    generation_params = {
                        "max_new_tokens": 4096,
                        "do_sample": True,
                        "temperature": 0.2,
                        "top_p": 0.9}
else:
    output_csv = f"output_csv_PD100cit/PD100cit_6classes_{model_key}_temperature.{temperature}.csv"

os.makedirs("output_csv_PD100cit", exist_ok=True)
max_new_tokens = generation_params["max_new_tokens"]
with open(prompt_fname, "r", encoding="utf-8") as f:
    instruction = f.read().strip()


model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True)
model.eval()

input_device = model.get_input_embeddings().weight.device

print("\nModel loaded.")
print(f"Model name: {model_name}")
print(f"Input device: {input_device}")
print(f"Thinking enabled: {THINKING}")
print(f"Generation params: {generation_params}")
print("Number of rows:", len(citation_sequence_x_100citations))

results = []
processed_indices = load_processed_indices_from_csv(output_csv)

print("\nInstruction prompt:")
print(instruction)

# Inference loop
for row_index, citation in enumerate(tqdm(citation_sequence_x_100citations, desc="Processing PD100cit citations")):
    if row_index in processed_indices:
        continue

    print(f"\n Citation n°{row_index} :")
    print(citation)
    true_label = citation_sequence_y_100citations[row_index]
    print("\nTrue label : ")
    print(true_label)

    messages = [{"role": "system", "content": instruction},
        {"role": "user", "content": "Citation passage:\n" + citation}]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)

    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_params, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

    input_length = inputs["input_ids"].shape[1]
    output_ids = generated_ids[0][input_length:].tolist()

    thinking_output, llm_output = extract_qwen3_thinking_and_content(output_ids=output_ids, tokenizer=tokenizer)

    result = {
        "row_index": row_index,
        "True_label": true_label,
        "citance_masked_citseg": citation,
        "model": model_name,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "thinking_enabled": THINKING,
        "thinking_output": thinking_output,
        "llm_output": llm_output
    }
    append_result_to_csv(result=result, csv_path=output_csv)
    processed_indices.add(row_index)

    print("\nResponse : ")
    print(llm_output)