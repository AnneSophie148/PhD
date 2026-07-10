# -*- coding:utf-8 -*-
# ! usr/bin/env python3

import re
import json
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CLASSES = ["method", "result"]
mapping_classes = {"method": ["CoCoGM"], "result": ["CoCoR0", "CoRes-"]}
inverse_mapping = {}

for label, cfuncs in mapping_classes.items():
    for cfunc in cfuncs:
        inverse_mapping[cfunc] = label


def main():
    thinking = False
    model = "qwen3"

    input_csv = "Teufel_CoCo_Qwen_Qwen3-32B_temperature.0.2.csv"
    df = pd.read_csv(input_csv).fillna("")

    # Keep only classes of interest
    df["gold_label"] = df["CFunc"].map(inverse_mapping)
    df = df[df["gold_label"].isin(CLASSES)].copy()

    y_pred_original = df["llm_output"].tolist()
    y_pred = []
    for pred in y_pred_original:
        if "result" in pred.lower():
            y_pred.append("result")
        elif "method" in pred.lower():
            y_pred.append("method")
        elif "result" in pred.lower() and "method" in pred.lower():
            print("ISSUE !! BOTH result and method in prediction")
        else:
            print("ISSUE neither result nor method in prediction")
    

    
    y_true = df["gold_label"].tolist()

    print("\n" + "=" * 100)
    print("Gold citance distribution")
    print("=" * 100)

    gold_distribution = Counter(y_true)

    for cls in CLASSES:
        print(f"{cls}: {gold_distribution.get(cls, 0)}")

    print("\nOriginal CFunc distribution")
    print("=" * 100)

    print("\nSize y_true : ", len(y_true))
    print("\nSize y_pred : ", len(y_pred))

    report_dict = classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASSES, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    if thinking:
        output_classification = "Classification_report_method_result_thinking.csv"
        output_cm = "Confusion_matrix_method_result_thinking.csv"
    else:
        output_classification = "Classification_report_method_result.csv"
        output_cm = "Confusion_matrix_method_result.csv"
    report_df.to_csv(output_classification, index=True)

    print(report_df)

    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    df_cm = pd.DataFrame(cm, index=["true_compare", "true_contrast"], columns=["pred_compare", "pred_contrast"])
    df_cm.to_csv(output_cm)


if __name__ == "__main__":
    main()