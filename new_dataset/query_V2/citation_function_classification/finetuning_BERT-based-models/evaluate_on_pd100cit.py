import os
import gc
import torch
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from get_citation_sequence_updated import load_PD_data
from citation_classifier import CitationClassifier


available_models = ["SciBERT", "BioBERT", "RoBERTa", "BioLinkBERT", "PubMedBERT"]
context_windows = ["2-2", "3-3"]
seeds = [42, 1965, 5171, 789]

augment_coco_with_scicite = True
scicite_aug_key = f"scicite_{augment_coco_with_scicite}"

use_section = False
ACC_STEP = 1
lr = 5e-5
section_emb_dim = 32
batch_size = 16

checkpoint_dir = "model_save_sci_cite"
output_dir = "scores_pd100cit"
os.makedirs(output_dir, exist_ok=True)


def labels_to_indices(label_mapping, labels):
    missing = sorted(set(labels) - set(label_mapping.keys()))
    if missing:
        raise ValueError(f"Labels in test set not found in checkpoint labels: {missing}")

    return [label_mapping[label] for label in labels]


class NoSectionCitationDataset(Dataset):
    """
    Dataset for evaluating checkpoints trained with use_section=False.

    This deliberately never creates a "section_ids" field. Therefore the
    DataLoader cannot call torch.tensor() on section strings such as
    "Introduction" or "Results".
    """

    def __init__(self, text_citations, labels_ind, tokenizer, max_length=512):
        if len(text_citations) != len(labels_ind):
            raise ValueError(f"text_citations and labels_ind have different lengths: {len(text_citations)} vs {len(labels_ind)}")
        self.text_citations = list(text_citations)
        self.labels_ind = list(labels_ind)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.text_citations)

    def __getitem__(self, idx):
        citation_text = self.text_citations[idx]

        encoded = self.tokenizer(
            citation_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels_ind[idx], dtype=torch.long),
            "citation_text": citation_text,
        }





def load_model_from_checkpoint(checkpoint_path, device):
    models = {
        "PubMedBERT": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "BioLinkBERT": "michiyasunaga/BioLinkBERT-base",
        "BioBERT": "dmis-lab/biobert-v1.1",
        "SciBERT": "allenai/scibert_scivocab_uncased",
        "RoBERTa-large": "all-roberta-large-v1",
        "RoBERTa": "roberta-base",
    }

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_short_name = checkpoint["model_short_name"]
    model_name = models[model_short_name]

    print("\nLoading checkpoint from:")
    print(checkpoint_path)
    print("Model short name:", model_short_name)
    print("Model HF name:", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)

    tokenizer.add_tokens(["CITSEG"], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    citseg_id = tokenizer.convert_tokens_to_ids("CITSEG")
    print("CITSEG token id:", citseg_id)

    loaded_model = CitationClassifier(
        linear_size=checkpoint["hidden_layers"],
        model=model,
        tokenizer=tokenizer,
        in_features=checkpoint["in_features"],
        num_class=len(checkpoint["all_labels"]),
        use_section=checkpoint["use_section"],
        num_sections=checkpoint["num_sections"],
        section_emb_dim=checkpoint.get("section_emb_dim", 32),
        dropout=checkpoint.get("dropout", 0.5),
    )

    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model = loaded_model.to(device)
    loaded_model.eval()

    all_labels = checkpoint["all_labels"]

    print("Checkpoint loaded correctly")
    print("Labels:", all_labels)
    print("Use section:", checkpoint["use_section"])

    return loaded_model, tokenizer, all_labels, checkpoint



def evaluate_one_checkpoint(model_short_name, window, seed, checkpoint_path, device):
    loaded_model, tokenizer, all_labels, checkpoint = load_model_from_checkpoint(checkpoint_path=checkpoint_path, device=device,)

    left_context = int(window.split("-")[0])
    right_context = int(window.split("-")[-1])

    citation_sequence_x_100citations, citation_sequence_y_100citations, section_pd100cit, citation_sections_left_pd100cit, section_position_pd100cit = load_PD_data(left_context, right_context)
    citation_sequence_y_100citations = [y.lower() for y in citation_sequence_y_100citations]

    label_mapping = {label: idx for idx, label in enumerate(all_labels)}
    labels_idx_test = labels_to_indices(label_mapping, citation_sequence_y_100citations)

    print("\nEvaluating:", model_short_name, "window", window, "seed", seed)
    print("Labels in 100citations:", sorted(set(citation_sequence_y_100citations)))
    print("size x:", len(citation_sequence_x_100citations))
    print("size y:", len(citation_sequence_y_100citations))

    if bool(checkpoint.get("use_section", False)):
        raise ValueError("This script is for no-section evaluation only")

    test_dataset = NoSectionCitationDataset(
        text_citations=citation_sequence_x_100citations,
        labels_ind=labels_idx_test,
        tokenizer=tokenizer,
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    second_classes = []
    third_classes = []
    true_labels = []
    test_texts = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            citation_text = batch["citation_text"]

            section_ids = batch.get("section_ids")
            if section_ids is not None:
                section_ids = section_ids.to(device)

            logits = loaded_model(
                tokens=input_ids,
                attention_mask=attention_mask,
                section_ids=section_ids,
            )

            topk_values, topk_indices = torch.topk(logits, k=3, dim=-1)

            predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()
            second_class = topk_indices[:, 1].cpu().numpy()
            third_class = topk_indices[:, 2].cpu().numpy()

            predictions.extend(predicted_classes.tolist())
            second_classes.extend(second_class.tolist())
            third_classes.extend(third_class.tolist())
            true_labels.extend(labels.cpu().numpy().tolist())
            test_texts.extend(citation_text)

    num_classes = len(all_labels)
    label_ids = sorted(set(true_labels))

    prediction_rows = []
    for i in range(len(true_labels)):
        prediction_rows.append(
            {
                "model": model_short_name,
                "window": window,
                "seed": seed,
                "citation_index": i,
                "citation_text": test_texts[i],
                "top1_class": all_labels[predictions[i]],
                "top2_class": all_labels[second_classes[i]],
                "top3_class": all_labels[third_classes[i]],
                "true_label": all_labels[true_labels[i]],
            }
        )

    test_accuracy = accuracy_score(true_labels, predictions)

    test_precision_micro = precision_score(true_labels, predictions, average="micro", labels=label_ids, zero_division=0)
    test_recall_micro = recall_score(true_labels, predictions, average="micro", labels=label_ids, zero_division=0)
    test_f1_micro = f1_score(true_labels, predictions, average="micro", labels=label_ids, zero_division=0)

    test_precision_macro = precision_score(true_labels, predictions, average="macro", labels=label_ids, zero_division=0)
    test_recall_macro = recall_score(true_labels, predictions, average="macro", labels=label_ids, zero_division=0)
    test_f1_macro = f1_score(true_labels, predictions, average="macro", labels=label_ids, zero_division=0)

    test_precision_weighted = precision_score(true_labels, predictions, average="weighted", labels=label_ids, zero_division=0)
    test_recall_weighted = recall_score(true_labels, predictions, average="weighted", labels=label_ids,zero_division=0)
    test_f1_weighted = f1_score(true_labels, predictions, average="weighted", labels=label_ids, zero_division=0)

    present_label_ids = sorted(set(true_labels))
    present_label_names = [all_labels[i] for i in present_label_ids]

    test_report = classification_report(
        true_labels,
        predictions,
        labels=present_label_ids,
        target_names=present_label_names,
        zero_division=0,
        output_dict=True
    )

    class_score_rows = []
    for class_name in present_label_names:
        class_scores = test_report[class_name]
        class_score_rows.append(
            {
                "model": model_short_name,
                "window": window,
                "seed": seed,
                "class": class_name,
                "precision": class_scores["precision"],
                "recall": class_scores["recall"],
                "f1_score": class_scores["f1-score"],
                "support": int(class_scores["support"]),
            }
        )

    test_confusion_matrix = confusion_matrix(true_labels, predictions, labels=label_ids).tolist()

    result_row = {
        "model": model_short_name,
        "window": window,
        "seed": seed,
        "accuracy": test_accuracy,
        "precision_micro": test_precision_micro,
        "recall_micro": test_recall_micro,
        "f1_micro": test_f1_micro,
        "precision_macro": test_precision_macro,
        "recall_macro": test_recall_macro,
        "f1_macro": test_f1_macro,
        "precision_weighted": test_precision_weighted,
        "recall_weighted": test_recall_weighted,
        "f1_weighted": test_f1_weighted,
        "confusion_matrix": test_confusion_matrix,
    }

    print("Accuracy:", test_accuracy)
    print("Micro F1:", test_f1_micro)
    print("Macro F1:", test_f1_macro)
    print("Weighted F1:", test_f1_weighted)

    return result_row, prediction_rows, class_score_rows



def make_average_model_summary(results_df):
    """
    Average model-level metrics across seeds for each model/context window.
    Includes F1 micro, macro and weighted, plus P/R for completeness.
    """
    if results_df.empty:
        return pd.DataFrame()

    metric_cols = [
        "accuracy",
        "precision_micro",
        "recall_micro",
        "f1_micro",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]

    summary = (results_df.groupby(["model", "window"], as_index=False).agg(n_seeds=("seed", "nunique"), **{f"{col}_mean": (col, "mean") for col in metric_cols}, **{f"{col}_std": (col, "std") for col in metric_cols},))
    std_cols = [col for col in summary.columns if col.endswith("_std")]
    summary[std_cols] = summary[std_cols].fillna(0)

    return summary



def make_average_class_summary(class_scores_df):
    """
    Average per-class precision, recall and F1 across seeds for each
    model/context window/class. The support is the number of true instances
    for that class in the PD100Cit test set.
    """
    if class_scores_df.empty:
        return pd.DataFrame()

    summary = (class_scores_df.groupby(["model", "window", "class"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            support=("support", "max"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_score_mean=("f1_score", "mean"),
            f1_score_std=("f1_score", "std"),
        )
    )

    std_cols = [col for col in summary.columns if col.endswith("_std")]
    summary[std_cols] = summary[std_cols].fillna(0)
    summary["support"] = summary["support"].astype(int)

    return summary



def main():
    print("Torch version:", torch.__version__)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"PyTorch is using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("PyTorch is using CPU")

    all_model_results = []
    all_prediction_rows = []
    all_class_score_rows = []
    missing_checkpoints = []

    for model_short_name in available_models:
        for window in context_windows:
            for seed in seeds:
                checkpoint_path = f"{checkpoint_dir}/seed_{seed}/DIC_FINAL_{model_short_name}_{lr}_accsteps{ACC_STEP}_ctx_{window}_section_{use_section}_{section_emb_dim}_{scicite_aug_key}.pt"

                if not os.path.exists(checkpoint_path):
                    print(f"\nCheckpoint not found, skipping: {checkpoint_path}")
                    missing_checkpoints.append(
                        {
                            "model": model_short_name,
                            "window": window,
                            "seed": seed,
                            "checkpoint_path": checkpoint_path,
                        }
                    )
                    continue

                result_row, prediction_rows, class_score_rows = evaluate_one_checkpoint(
                    model_short_name=model_short_name,
                    window=window,
                    seed=seed,
                    checkpoint_path=checkpoint_path,
                    device=device
                )

                all_model_results.append(result_row)
                all_prediction_rows.extend(prediction_rows)
                all_class_score_rows.extend(class_score_rows)

    results_df = pd.DataFrame(all_model_results)
    predictions_df = pd.DataFrame(all_prediction_rows)
    class_scores_df = pd.DataFrame(all_class_score_rows)
    missing_checkpoints_df = pd.DataFrame(missing_checkpoints)

    average_results_df = make_average_model_summary(results_df)
    average_class_scores_df = make_average_class_summary(class_scores_df)

    per_seed_results_path = os.path.join(output_dir, "pd100citations_model_evaluation_per_seed.csv")
    average_results_path = os.path.join(output_dir, "pd100citations_model_evaluation_average_across_seeds.csv",)
    predictions_path = os.path.join(output_dir, "pd100citations_model_predictions_by_seed.csv")
    per_seed_class_scores_path = os.path.join(output_dir, "pd100citations_model_scores_by_class_per_seed.csv",)
    average_class_scores_path = os.path.join(output_dir, "pd100citations_model_scores_by_class_average_across_seeds.csv")
    missing_checkpoints_path = os.path.join(output_dir, "pd100citations_missing_checkpoints.csv")

    results_df.to_csv(per_seed_results_path, index=False)
    average_results_df.to_csv(average_results_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    class_scores_df.to_csv(per_seed_class_scores_path, index=False)
    average_class_scores_df.to_csv(average_class_scores_path, index=False)
    missing_checkpoints_df.to_csv(missing_checkpoints_path, index=False)

    print("\nSaved:")
    print(per_seed_results_path)
    print(average_results_path)
    print(predictions_path)
    print(per_seed_class_scores_path)
    print(average_class_scores_path)
    print(missing_checkpoints_path)

    if not average_results_df.empty:
        print("\nAverage F1 across seeds by model/window:")
        print(
            average_results_df[
                [
                    "model",
                    "window",
                    "n_seeds",
                    "f1_micro_mean",
                    "f1_macro_mean",
                    "f1_weighted_mean",
                ]
            ].to_string(index=False)
        )

    if missing_checkpoints:
        print("\nWarning: some checkpoints were missing. Averages were computed using available seeds only.")
        print(f"Missing checkpoint table saved to: {missing_checkpoints_path}")


if __name__ == "__main__":
    main()
