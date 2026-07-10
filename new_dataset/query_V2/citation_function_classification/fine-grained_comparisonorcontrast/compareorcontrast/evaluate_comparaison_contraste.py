import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_jsonl(jsonl_path):
    rows = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            rows.append(json.loads(line))

    return rows


def normalize_label(label):
    if label is None:
        return ""

    label = str(label).strip().lower()

    # Remove common LLM prefixes
    label = label.replace("label:", "").strip()
    label = label.replace("prediction:", "").strip()
    label = label.replace("predicted label:", "").strip()

    # Remove markdown bold markers
    label = label.replace("**", "").strip()

    # Normalise compare/contrast variants
    label = label.replace("compare or contrast", "compareorcontrast")
    label = label.replace("compare_or_contrast", "compareorcontrast")
    label = label.replace("compare/contrast", "compareorcontrast")

    # Optional: if the LLM outputs extra text, keep only valid labels
    if "contrast" in label:
        return "contrast"
    if "compare" in label:
        return "compare"

    return label


def main():
    thinking = True
    model = "qwen3"

    input_jsonl = "V2_output_jsonl/Comparaison_Constrast_Qwen_Qwen3-32B_temperature.0.2.jsonl"

    graph = load_jsonl(input_jsonl)

    true_labels = []
    predicted_labels = []

    for row in graph:
        true_label = normalize_label(row.get("CFunc"))
        predicted_label = normalize_label(row.get("llm_output"))

        if "contrast" in predicted_label.lower():
            predicted_label = "contrast"
        elif "compare" in predicted_label.lower():
            predicted_label = "compare"
        elif "other" in predicted_label.lower():
            predicted_label = "other"
        else:
            print("\nPredicted label could not be mapped!")
            print(repr(predicted_label))
            continue

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

        row["true_label"] = true_label
        row["llm_output"] = predicted_label
        row["correct"] = true_label == predicted_label

    print(f"Number of evaluated examples: {len(true_labels)}")
    print(f"Accuracy: {accuracy_score(true_labels, predicted_labels):.4f}")

    print("\nClassification report:")

    report_output_csv = "Classification_report_comparison_contrast.csv"
    cm_output_csv = "Confusion_matrix_comparison_contrast.csv"
    evaluation_labels = ["compare", "contrast"]

    print(f"Accuracy: {accuracy_score(true_labels, predicted_labels):.4f}")

    report_dict = classification_report(true_labels, predicted_labels, labels=evaluation_labels, target_names=evaluation_labels, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv("Classification_report_comparison_contrast.csv", index=True)

    print(report_df)

    cm = confusion_matrix(true_labels, predicted_labels, labels=evaluation_labels)
    df_cm = pd.DataFrame(cm, index=["true_compare", "true_contrast"], columns=["pred_compare", "pred_contrast"])
    df_cm.to_csv("Confusion_matrix_comparison_contrast.csv")


if __name__ == "__main__":
    main()