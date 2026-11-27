import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

input_file = "time_series_per_article.json"
non_molecular_methods = {
    "morphological / taxonomic identification",
    "light / fluorescence microscopy",
    "electron microscopy",
    "isolation / culture",
    "biochemical / physiological identification"
}


def has_non_molecular(value):
    for item in value or []:
        for organism in item.get("organisms") or []:
            for method in organism.get("identification_methods") or []:
                normalized_method = (method.get("normalized_method") or "").strip().lower()
                if normalized_method in non_molecular_methods:
                    return True

    return False

def main():
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    articles_by_category = {}
    for category in ["points_noirs", "points_blancs"]:
        articles_by_category[category] = {}
        for articles in data.get(category, {}).values():
            for doi, article in articles.items():
                articles_by_category[category].setdefault(doi, []).append(article)
    overlapping_dois = set(articles_by_category["points_noirs"]) & set(articles_by_category["points_blancs"])
    labels, predictions, results = [], [], []
    for category, label in [("points_noirs", 0), ("points_blancs", 1)]:
        for doi, records in articles_by_category[category].items():
            if doi in overlapping_dois:
                continue
            detected = any(has_non_molecular((article.get("cited_info") or {}).get("identification_methods")) for article in records)
            prediction = 0 if detected else 1
            labels.append(label)
            predictions.append(prediction)

    print(f"Articles evaluated: {len(labels)}")
    print(f"Excluded overlapping DOIs: {len(overlapping_dois)}")
    print(f"Accuracy: {accuracy_score(labels, predictions):.3f}")
    print(f"Macro F1: {f1_score(labels, predictions, labels=[0, 1], average='macro', zero_division=0):.3f}")
    print(classification_report(labels, predictions, labels=[0, 1], target_names=["points_noirs", "points_blancs"], zero_division=0))
    
if __name__ == "__main__":
    main()