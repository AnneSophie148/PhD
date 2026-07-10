import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, f1_score, classification_report


ctx = '3-3'
predictions_path = "PD100cit_6classes_Qwen_Qwen3-32B_temperature.0.2.csv"
predictions_path = "PD100cit_6classes_Qwen_Qwen3-32B_temperature.0.2_THINKING.csv"

print(f"Evaluation file {predictions_path}")

df = pd.read_csv(predictions_path)


all_y_true = df["True_label"].tolist()
all_labels = set(all_y_true)
labels = ["background", "motivation", "uses", "extends", "compareorcontrast", "future"]

#after excluding classes that are not in the test
labels = ["background", "motivation", "uses", "compareorcontrast"]
all_y_pred = []

for y in df["llm_output"].tolist():
    y = y.replace("Comparison or Contrast", "compareorcontrast").lower()
    all_y_pred.append(y)

for y in all_y_pred:
    if y not in all_labels:
        print("ISSUE LABEL : ", y)

print("True labels : ", all_y_true)
print("Predicted labels : ", all_y_pred)


precision, recall, f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average=None, zero_division=0)
macro_f1 = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(all_y_true, all_y_pred, average="weighted", zero_division=0)

print(f"F1 macro : {macro_f1}")
print(f"weighted_f1 : {weighted_f1}")

report = classification_report(
    all_y_true,
    all_y_pred,
    labels=labels,
    target_names=labels,
    zero_division=0,
    digits=4
)

print("\nClassification report:")
print(report)