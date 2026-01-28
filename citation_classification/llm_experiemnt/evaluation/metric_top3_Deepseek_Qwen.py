import glob
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, precision_recall_fscore_support, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ctx_windows = ["11", "22", "33"]
top = "top3_3"

#to get the true labels that where saved there
predictions_path = f'responses_2-2_V2.csv'
df = pd.read_csv(predictions_path)
real_labels = df["True_label"].tolist()

models = ["DeepSeek", "Qwen3.0", "QwQ-32B"]

#paths for each models
paths = [f"DeepSeek-R1-Distill-Qwen-32B-unsloth-bnb-4bit.prompt_V3_{top}steps.citation-context_experiment",
    f"Qwen3-32B-unsloth-bnb-4bit.prompt_V3_{top}steps.citation-context_experiment",
    f"QwQ-32B-unsloth-bnb-4bit.prompt_V3_{top}steps.citation-context_experiment"]

output_df = {"model" : [], "context_window" : [], "F1_weighted" : [], "F1_macro" : [], "top3_accuracy" : []}

for model in models:
    for ctx in ctx_windows:
        indice_model = models.index(model)
        path = paths[indice_model]

        results_file = glob.glob(f"models/{model}/{path}/*{ctx}.txt")

        #labels
        all_labels = ['similar', 'neutral', 'usage', 'cocores', 'motivation', 'basis', 'weakness', 'future', 'support', 'cocogm', 'cocoxy']

        def normalize_class(classe_name):
            """Normalize class names for comparison."""
            return (classe_name.lower().strip().replace("\\fp{", "").replace("\\boxed{", "").replace("}", "").replace(".", "").replace(" ", ""))


        top1, top2, top3 = [], [], []

        for i in range(len(real_labels)):
            citation_number = i + 1
            citation_identifier = f"{citation_number:04d}"

            for file in results_file:
                if citation_identifier in file:
                    with open(file, "r") as f:
                        text = f.read()
                        text = text.split("</think>")[-1].strip()
                        
                        lines = [l.strip() for l in text.splitlines() if l.strip()]

                        t1, t2, t3 = None, None, None
                        

                        for l in lines:
                            if l.startswith("1"):
                                top_number, cls = l.split('.', 1)
                                cls = normalize_class(cls)
                                if cls in all_labels:
                                    t1 = cls

                            elif l.startswith("2"):
                                top_number, cls = l.split('.', 1)
                                cls = normalize_class(cls)
                                if cls in all_labels:
                                    t2 = cls

                            elif l.startswith("3"):
                                top_number, cls = l.split('.', 1)
                                cls = normalize_class(cls)
                                if cls in all_labels:
                                    t3 = cls

                        top1.append(t1)
                        top2.append(t2)
                        top3.append(t3)

        #check lenghts
        if len(real_labels) != len(top1):
            raise ValueError(f"Ground truth length ({len(real_labels)}) does not match predictions ({len(top1)})")

        #metrics
        macro_precision = precision_score(real_labels, top1, average='macro', zero_division=0)
        macro_recall = recall_score(real_labels, top1, average='macro', zero_division=0)
        weighted_precision = precision_score(real_labels, top1, average='weighted', zero_division=0)
        weighted_recall = recall_score(real_labels, top1, average='weighted', zero_division=0)
        macro_f1 = f1_score(real_labels, top1, average="macro", zero_division=0)
        weighted_f1 = f1_score(real_labels, top1, average="weighted", zero_division=0)

        print("\n--- Global metrics (Top-1) ---")
        print(f"Macro Precision:   {macro_precision:.3f}")
        print(f"Macro Recall:      {macro_recall:.3f}")
        print(f"Weighted Precision:{weighted_precision:.3f}")
        print(f"Weighted Recall:   {weighted_recall:.3f}")
        print(f"Macro F1:          {macro_f1:.3f}")
        print(f"Weighted F1:       {weighted_f1:.3f}")

        top3_accuracy = 0

        for i in range(len(real_labels)):
            label = real_labels[i]
            if label in [top1[i], top2[i], top3[i]]:
                top3_accuracy += 1

        top3_accuracy = top3_accuracy / len(real_labels)
        print(f"Top3_accuracy = {top3_accuracy}")

        output_df["model"].append(model)
        output_df["context_window"].append(ctx)
        output_df["F1_weighted"].append(weighted_f1)
        output_df["F1_macro"].append(macro_f1)
        output_df["top3_accuracy"].append(top3_accuracy)

        #metrics per class
        precision, recall, f1, support = precision_recall_fscore_support(real_labels, top1, labels=all_labels, average=None, zero_division=0)
        metrics_df = pd.DataFrame({
            "Class": all_labels,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "Support": support
        })
        print("Metrics per class")
        print(metrics_df)

        #Confusion matrix
        cm = confusion_matrix(real_labels, top1, labels=all_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
        disp.plot(xticks_rotation=45)
        plt.show()

df_out = pd.DataFrame(output_df)
csv_metrics_path = "output_result_promptV3_Deepseek_Qwen.csv"
df_out.to_csv(csv_metrics_path, index=False)
print(f"Metrics saved to {csv_metrics_path}")