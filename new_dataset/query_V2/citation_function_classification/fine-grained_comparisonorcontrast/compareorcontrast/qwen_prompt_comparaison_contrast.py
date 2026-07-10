import json
import os
import traceback

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from sklearn.utils import shuffle
import config_llm

#set up
generation_params = {
    "max_new_tokens": 4096,
    "temperature": 0.2
}

THINKING = True

if THINKING:
    generation_params = {
    "max_new_tokens": 4096,
    "do_sample": True,
    "temperature": 0.2,
    "top_p": 0.9
    }

model_name = "Qwen/Qwen3-32B"
tokenizer_name = model_name

#prompt_fname = "Distinguish_compare_contrast.txt"
prompt_fname = "V2_Distinguish_compare_contrast.txt"
input_csv_teufel = "Teufel_CoCo.csv"
input_csv_cohan = "Cohan_compare_contrast.csv"

model_key = model_name.replace("/", "_")
temperature = generation_params["temperature"]
max_new_tokens = generation_params["max_new_tokens"]

os.makedirs("V2_output_jsonl", exist_ok=True)

if THINKING:
    output_jsonl = f"V2_output_jsonl/V2_Comparaison_Constrast_{model_key}_temperature.{temperature}_THINKING.jsonl"
    output_csv =  f"V2_Comparaison_Constrast_{model_key}_temperature.{temperature}_THINKING.csv"
else:
    output_jsonl = f"V2_output_jsonl/Comparaison_Constrast_{model_key}_temperature.{temperature}.jsonl"
    
    output_csv = f"V2_Comparaison_Constrast_{model_key}_temperature.{temperature}.csv"


def clean_generated_text(text):
    """
    Remove common chat/template artefacts while preserving the actual content.
    """
    text = str(text)

    artefacts = [
        "<|im_end|>",
        "<|endoftext|>",
        "<|end_of_text|>",
        "<|assistant|>",
        "<|user|>",
        "<|system|>"
    ]

    for artefact in artefacts:
        text = text.replace(artefact, "")

    return text.strip()



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

    Recent Transformers versions support enable_thinking for Qwen3.
    The fallback keeps compatibility with slightly older versions.
    """
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    return prompt


def get_first_model_device(model):
    """
    Get the device of the model input embeddings.
    This is safer than next(model.parameters()) when device_map='auto'.
    """
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device

def main():
    login(token=config_llm.hf_token)

    #Load prompt
    with open(prompt_fname, "r", encoding="utf-8") as f:
        instruction = f.read().strip()

    #Load test data
    df_teufel = pd.read_csv(input_csv_teufel).fillna("")
    df_cohan = pd.read_csv(input_csv_cohan).fillna("")

    required_cohan_cols = ["sequence_masked", "label2"]
    required_teufel_cols = ["citance_masked_citseg", "CFunc"]

    missing_cohan = [c for c in required_cohan_cols if c not in df_cohan.columns]
    missing_teufel = [c for c in required_teufel_cols if c not in df_teufel.columns]

    if missing_cohan:
        raise ValueError(f"Missing columns in Cohan CSV: {missing_cohan}")

    if missing_teufel:
        raise ValueError(f"Missing columns in Teufel CSV: {missing_teufel}")

    citations_masked_cohan = df_cohan["sequence_masked"].tolist()
    y_cohan = df_cohan["label2"].tolist()

    citations_masked_teufel = df_teufel["citance_masked_citseg"].tolist()
    y_teufel = df_teufel["CFunc"].tolist()

    x_to_process = []
    true_y = []
    data_source = []

    for i in range(len(citations_masked_cohan)):
        if y_cohan[i] == "not_supportive":
            x_to_process.append(citations_masked_cohan[i])
            data_source.append("Cohan")
            true_y.append("contrast")
        elif y_cohan[i] == "supportive":
            x_to_process.append(citations_masked_cohan[i])
            data_source.append("Cohan")
            true_y.append("compare")

    for i in range(len(y_teufel)):
        if y_teufel[i] == "CoRes-" or y_teufel[i] == "CoCo-":
            x_to_process.append(citations_masked_teufel[i])
            data_source.append("Teufel")
            true_y.append("contrast")

    df_out = pd.DataFrame({
        "data_source": data_source,
        "CFunc": true_y,
        "citance_masked_citseg": x_to_process
    })

    df_out = df_out.sample(frac=1, random_state=42).reset_index(drop=True)
    df_out.insert(0, "row_index", range(len(df_out)))

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        token=config_llm.hf_token,
        trust_remote_code=True
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=config_llm.hf_token,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    model.eval()

    print("\nCUDA visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count():", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        print(
            f"GPU {i}: {props.name} | "
            f"free {free / 1024**3:.2f} GB / total {total / 1024**3:.2f} GB"
        )

    print("\nDevice map:")
    print(model.hf_device_map)

    input_device = get_first_model_device(model)

    print("\nModel loaded.")
    print(f"Model name: {model_name}")
    print(f"Input device: {input_device}")
    print(f"Thinking enabled: {THINKING}")
    print(f"Generation params: {generation_params}")
    print("Number of rows:", len(df_out))
    # ------------------------------------------------------------
    # Resume from JSONL if it already exists
    # ------------------------------------------------------------

    results = []
    processed_indices = set()

    if os.path.exists(output_jsonl):
        print(f"\nResuming from existing file: {output_jsonl}")

        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        results.append(item)
                        processed_indices.add(int(item["row_index"]))
                    except Exception:
                        print("\nWarning: could not parse one JSONL line:")
                        print(line)

        print("Already processed rows:", len(processed_indices))

    print("\nInstruction prompt:")
    print(instruction)

    # Inference loop
    for _, row in tqdm(df_out.iterrows(), total=len(df_out), desc="Citation predictions generated"):
        row_index = int(row["row_index"])

        if row_index in processed_indices:
            continue

        citance = str(row["citance_masked_citseg"]).strip()
        true_label = str(row["CFunc"])
        source = str(row["data_source"])

        if citance == "":
            continue

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": "Citation passage:\n" + citance}
        ]

        try:
            prompt = build_qwen3_prompt(tokenizer=tokenizer, messages=messages, enable_thinking=THINKING)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=False)

            inputs = {key: value.to(input_device) for key, value in inputs.items()}

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, **generation_params, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

            input_length = inputs["input_ids"].shape[1]
            output_ids = generated_ids[0][input_length:].tolist()

            thinking_output, llm_output = extract_qwen3_thinking_and_content(output_ids=output_ids, tokenizer=tokenizer)

            result = {
                "row_index": row_index,
                "data_source": source,
                "CFunc": true_label,
                "citance_masked_citseg": citance,
                "model": model_name,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "thinking_enabled": THINKING,
                "thinking_output": thinking_output,
                "llm_output": llm_output
            }

            results.append(result)
            processed_indices.add(row_index)
            print("\n___________")
            print("\nCitation : \n", citance)
            print("\nData source : \n", source)
            print("\nResponse : \n", llm_output)

            with open(output_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        except Exception as e:
            print("\nError with row:", row_index)
            print("Citance:", citance)
            print("Error type:", type(e).__name__)
            print("Error repr:", repr(e))
            traceback.print_exc()
            continue

    #save responses as csv

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df = results_df.drop_duplicates(
            subset=["row_index"],
            keep="last"
        )

    df_out["row_index"] = df_out.index

    if not results_df.empty:
        df_out = df_out.merge(
            results_df[
                [
                    "row_index",
                    "llm_output",
                    "thinking_output",
                    "model",
                    "max_new_tokens",
                    "temperature",
                    "thinking_enabled"
                ]
            ],
            on="row_index",
            how="left"
        )

    df_out.to_csv(output_csv, index=False, encoding="utf-8")

    print("\nFinished.")
    print(f"Saved incremental JSONL to: {output_jsonl}")
    print(f"Saved final CSV to: {output_csv}")
    print(f"Processed rows: {len(processed_indices)}")


if __name__ == "__main__":
    main()