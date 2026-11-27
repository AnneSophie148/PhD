input_file = "time_series_per_article.json"
criticism_classes = {"contrast - contradictory results", "contrast_results"}


def has_criticism(article):
    for citation in article.get("citation_passages", []):
        rc = str(citation.get("rhetorical_class", "")).strip().lower()
        if rc in criticism_classes:
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
            prediction = 0 if any(has_criticism(article) for article in records) else 1
            labels.append(label)
            predictions.append(prediction)

    print(f"Articles evaluated: {len(labels)}")
    print(f"Accuracy: {accuracy_score(labels, predictions):.3f}")
    print(f"Macro F1: {f1_score(labels, predictions, labels=[0, 1], average='macro', zero_division=0):.3f}")
    print(classification_report(labels, predictions, labels=[0, 1], target_names=["points_noirs", "points_blancs"], zero_division=0))
    
if __name__ == "__main__":
    main()