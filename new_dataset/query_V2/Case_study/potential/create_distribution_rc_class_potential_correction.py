import json
import re
from collections import defaultdict
from unidecode import unidecode
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from difflib import SequenceMatcher, get_close_matches
import os

EXTRACTED_TO_EPPO_RELATION = {
    "located in": "relates_to_location",
    "has been found on": "has_host",
    "affects": "has_host"
}

RHETORICAL_CLASSES = [
    "background",
    "motivation",
    "uses",
    "extends",
    "future",
    "contrast - contradictory results",
    "CoCoGM"
]


def normalize_rhetorical_class(label):
    label = str(label).lower().strip()
    label = re.sub(r"^output\s*:\s*", "", label)
    label = label.replace("compare or contrast", "compareorcontrast")
    label = label.replace("compare_or_contrast", "compareorcontrast")
    label = label.replace("compare/contrast", "compareorcontrast")
    label = re.sub(r"\s+", " ", label)
    return label


def normalize_entity_name(name):
    if name is None:
        return ""

    name = unidecode(str(name).lower().strip())
    name = re.sub(r"\([^)]*\)", " ", name)
    name = re.sub(r"[^a-z0-9]+", " ", name)
    name = re.sub(r"\s+", " ", name)

    return name.strip()


def normalize_relation_name(relation):
    if relation is None:
        return ""

    relation = unidecode(str(relation).lower().strip())
    relation = relation.replace("_", " ")
    relation = relation.replace("-", " ")
    relation = re.sub(r"\s+", " ", relation)

    return relation.strip()

def get_abbreviated_name(name):
    normalized_name = normalize_entity_name(name)
    parts = normalized_name.split()

    if len(parts) < 2:
        return ""

    genus_initial = parts[0][0]
    species_or_rest = " ".join(parts[1:])
    return genus_initial + " " + species_or_rest

def build_eppo_name_index(code_names):
    name_to_codes = defaultdict(set)
    code_to_preferred_name = {}
    code_to_search_names = defaultdict(set)

    for item in code_names:
        code = str(item.get("id", "")).strip()
        names = item.get("names", [])
        prefs = item.get("prefs", [])

        if not code or not names:
            continue

        preferred_name = None

        for i, name in enumerate(names):
            if i < len(prefs) and prefs[i] is True:
                preferred_name = name
                break

        if preferred_name is None:
            preferred_name = names[0]

        code_to_preferred_name[code] = preferred_name

        for name in names:
            normalized_name = normalize_entity_name(name)
            abbreviated_name = get_abbreviated_name(name)

            if normalized_name:
                name_to_codes[normalized_name].add(code)
                code_to_search_names[code].add(normalized_name)

            if abbreviated_name:
                name_to_codes[abbreviated_name].add(code)
                code_to_search_names[code].add(abbreviated_name)

    aliases = list(name_to_codes.keys())
    return name_to_codes, code_to_preferred_name, code_to_search_names, aliases


def get_relation_label(relation):
    for key in ["relation", "relationship", "relation_type", "rel_type", "predicate", "type"]:
        value = relation.get(key, "")

        if str(value).strip():
            return str(value).strip()

    return ""


def get_entity_name(entity):
    if isinstance(entity, str):
        return entity.strip()

    if isinstance(entity, dict):
        for key in ["scientific_name", "name", "in_text", "text", "label"]:
            value = entity.get(key, "")

            if str(value).strip():
                return str(value).strip()

    return ""


def get_eppo_match(entity_name, name_to_codes, code_to_preferred_name, aliases, fuzzy_threshold=0.94):
    normalized_name = normalize_entity_name(entity_name)

    empty_result = {
        "code": "",
        "preferred_name": "",
        "matched_name": "",
        "method": "not found",
        "score": 0.0
    }

    if not normalized_name:
        return empty_result

    exact_codes = name_to_codes.get(normalized_name, set())

    if len(exact_codes) == 1:
        code = next(iter(exact_codes))

        return {
            "code": code,
            "preferred_name": code_to_preferred_name.get(code, ""),
            "matched_name": normalized_name,
            "method": "exact",
            "score": 1.0
        }

    if len(exact_codes) > 1:
        empty_result["method"] = "ambiguous exact match"
        return empty_result

    fuzzy_names = get_close_matches(normalized_name, aliases, n=3, cutoff=fuzzy_threshold)

    if not fuzzy_names:
        return empty_result

    candidates = []

    for matched_name in fuzzy_names:
        codes = name_to_codes.get(matched_name, set())

        if len(codes) != 1:
            continue

        code = next(iter(codes))
        score = SequenceMatcher(None, normalized_name, matched_name).ratio()
        candidates.append((score, code, matched_name))

    if not candidates:
        return empty_result

    candidates.sort(reverse=True)
    best_score, best_code, best_name = candidates[0]

    if len(candidates) > 1:
        second_score, second_code, _ = candidates[1]

        if second_code != best_code and abs(best_score - second_score) < 0.01:
            empty_result["method"] = "ambiguous fuzzy match"
            empty_result["score"] = best_score
            return empty_result

    return {
        "code": best_code,
        "preferred_name": code_to_preferred_name.get(best_code, ""),
        "matched_name": best_name,
        "method": "fuzzy",
        "score": best_score
    }

def get_all_entity_names(entity):
    names = []

    if isinstance(entity, str):
        if entity.strip():
            names.append(entity.strip())

    elif isinstance(entity, dict):
        for key in ["scientific_name", "name", "in_text", "text", "label"]:
            value = entity.get(key, "")

            if str(value).strip():
                names.append(str(value).strip())

    return list(dict.fromkeys(names))


def normalize_matching_text(text):
    text = unidecode(str(text).lower().strip())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def term_is_in_entity(term, entity_text):
    term = normalize_matching_text(term)
    entity_text = normalize_matching_text(entity_text)

    if not term or not entity_text:
        return False

    if term == entity_text:
        return True

    pattern = r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"
    return re.search(pattern, entity_text) is not None


def match_entity_to_eppo_code(entity, eppo_code, code_to_search_names):
    entity_names = get_all_entity_names(entity)
    eppo_names = code_to_search_names.get(eppo_code, set())

    for entity_name in entity_names:
        for eppo_name in eppo_names:
            if term_is_in_entity(eppo_name, entity_name):
                return {
                    "matches": True,
                    "entity_name": entity_name,
                    "matched_eppo_name": eppo_name
                }

    return {
        "matches": False,
        "entity_name": " | ".join(entity_names),
        "matched_eppo_name": ""
    }


def load_json(path):
    path = os.path.expanduser(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_doi(doi):
    return unidecode(str(doi).lower().strip().replace("https://doi.org/", "").replace("http://doi.org/", "").replace("doi:", "").rstrip(".").rstrip(","))


def extract_json(content):
    if not content:
        return None

    content = str(content).strip()
    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL)

    if fenced_match:
        content = fenced_match.group(1).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def get_relations(parsed_prediction):
    relations = []

    if isinstance(parsed_prediction, dict):
        parsed_prediction = [parsed_prediction]

    if not isinstance(parsed_prediction, list):
        return relations

    for item in parsed_prediction:
        if not isinstance(item, dict):
            continue

        item_relations = item.get("relations", item.get("relationships", []))

        if isinstance(item_relations, list):
            relations.extend(item_relations)

    return relations


def get_prediction_content(prediction):
    if isinstance(prediction, dict):
        return prediction.get("content", "")

    if isinstance(prediction, str):
        return prediction

    return ""


def get_passage_text(passage):
    for key in ["Full-text", "full_text"]:
        value = passage.get(key, "")

        if str(value).strip():
            return str(value).strip()

    return ""

def normalize_rhetorical_class(label):
    label = str(label).strip().lower()
    label = label.replace("–", "-").replace("—", "-")
    label = re.sub(r"\s*-\s*", " - ", label)
    label = re.sub(r"\s+", " ", label)
    label_mapping = {
        "cocogm": "CoCoGM",
        "compare results": "compare_results",
        "compare_results": "compare_results",
        "contrast results": "contrast_results",
        "contrast_results": "contrast_results",
        "contrast - contradictory results": "contrast - contradictory results"
    }
    return label_mapping.get(label, label)

def build_citation_index_updated(graph):
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", {})
    articles = nodes.get("articles", [])
    findings = nodes.get("findings", [])
    rhetorical_classes = nodes.get("rhetorical_classes", [])
    in_context_citations = nodes.get("in_context_citations", [])
    article_doi_by_reference = {}
    for article in articles:
        doi = normalize_doi(article.get("doi"))
        if not doi:
            continue
        article_doi_by_reference[doi] = doi
        if article.get("id"):
            article_doi_by_reference[str(article["id"])] = doi
    rhetorical_class_by_id = {
        item["id"]: normalize_rhetorical_class(item.get("rhetorical_class", "unknown"))
        for item in rhetorical_classes
        if item.get("id")
    }
    passage_by_id = {
        item["id"]: item
        for item in in_context_citations
        if item.get("id")
    }
    cited_dois_by_finding = defaultdict(set)
    for edge in edges.get("article_claims_finding", []):
        article_reference = str(edge.get("from", ""))
        cited_doi = article_doi_by_reference.get(article_reference, normalize_doi(article_reference))
        if edge.get("to") and cited_doi:
            cited_dois_by_finding[edge["to"]].add(cited_doi)
    for finding in findings:
        finding_id = finding.get("id")
        cited_doi = normalize_doi(finding.get("cited_doi"))
        if finding_id and cited_doi:
            cited_dois_by_finding[finding_id].add(cited_doi)
    finding_ids_by_ce = defaultdict(set)
    for edge in edges.get("citation_event_refers_to_finding", []):
        if edge.get("from") and edge.get("to"):
            finding_ids_by_ce[edge["from"]].add(edge["to"])
    citing_dois_by_ce = defaultdict(set)
    for edge in edges.get("article_produces_citation_event", []):
        article_reference = str(edge.get("from", ""))
        citing_doi = article_doi_by_reference.get(article_reference, normalize_doi(article_reference))
        if edge.get("to") and citing_doi:
            citing_dois_by_ce[edge["to"]].add(citing_doi)
    rhetorical_classes_by_ce = defaultdict(set)
    for edge in edges.get("citation_event_has_rhetorical_class", []):
        rhetorical_class = rhetorical_class_by_id.get(edge.get("to"))
        if edge.get("from") and rhetorical_class:
            rhetorical_classes_by_ce[edge["from"]].add(rhetorical_class)
    passage_ids_by_ce = defaultdict(set)
    for edge in edges.get("citation_event_is_expressed_as_in_context_citation", []):
        if edge.get("from") and edge.get("to"):
            passage_ids_by_ce[edge["from"]].add(edge["to"])
            
    citation_index = defaultdict(lambda: defaultdict(list))
    added_records = set()
    for ce_id, finding_ids in finding_ids_by_ce.items():
        cited_dois = {
            cited_doi
            for finding_id in finding_ids
            for cited_doi in cited_dois_by_finding.get(finding_id, set())
        }
        citing_dois = citing_dois_by_ce.get(ce_id, set())
        rhetorical_classes_for_event = rhetorical_classes_by_ce.get(ce_id, set())
        if len(rhetorical_classes_for_event) > 1:
            raise ValueError("Citation event " + str(ce_id) + " has multiple rhetorical classes: " + str(sorted(rhetorical_classes_for_event)))
        rhetorical_class = next(iter(rhetorical_classes_for_event), "unknown")
        for passage_id in passage_ids_by_ce.get(ce_id, set()):
            passage = passage_by_id.get(passage_id)
            if not passage:
                continue
            passage_record = {
                "text": get_passage_text(passage),
                "rhetorical_class": rhetorical_class
            }
            for cited_doi in cited_dois:
                for citing_doi in citing_dois:
                    record_key = cited_doi, citing_doi, ce_id, passage_id
                    if record_key not in added_records:
                        citation_index[cited_doi][citing_doi].append(passage_record.copy())
                        added_records.add(record_key)
    return citation_index
    

def build_citation_index(graph):
    citation_index = defaultdict(lambda: defaultdict(list))

    for edge in graph.get("edges", []):
        cited_doi = normalize_doi(edge.get("cited_doi", ""))
        citing_doi = normalize_doi(edge.get("citing_doi", ""))

        if not cited_doi or not citing_doi:
            continue

        for passage in edge.get("citation_passages", []):
            passage_record = {
                "text": get_passage_text(passage),
                "rhetorical_class": passage.get("predicted_rhetorical_class", "")
            }

            citation_index[cited_doi][citing_doi].append(passage_record)

    return citation_index


def print_citations_for_cited_doi(cited_doi, citation_index):
    normalized_cited_doi = normalize_doi(cited_doi)
    citing_articles = citation_index.get(normalized_cited_doi, {})

    print("\nCited DOI:", normalized_cited_doi)
    print("Number of citing DOIs:", len(citing_articles))

    if not citing_articles:
        print("No citing articles found in the rhetorical citation graph.")
        return

    for citing_doi, passages in citing_articles.items():
        print("\nCiting DOI:", citing_doi)
        print("Number of citation passages:", len(passages))

        for passage_number, passage in enumerate(passages, start=1):
            print("\nCitation passage", passage_number)
            print("Rhetorical class:", passage["rhetorical_class"] or "Missing")

            if passage["label2"]:
                print("Fine-grained class:", passage["label2"])

            if passage["label3"]:
                print("Fine-grained class label3:", passage["label3"])

            print("Passage:", passage["text"] or "Missing passage text")


def get_unique_abstracts_by_doi(data):
    unique_abstracts = {}

    for interaction_obj in data.values():
        for abstract_record in interaction_obj.get("abstracts_with_this_interaction", []):
            cited_doi = normalize_doi(abstract_record.get("doi", ""))

            if not cited_doi:
                continue

            if cited_doi not in unique_abstracts:
                unique_abstracts[cited_doi] = {"title": abstract_record.get("title", ""), "abstract": abstract_record.get("abstract", "")}

    return unique_abstracts


def build_rhetorical_percentage_per_abstract(data, citation_index, minimum_citations=5):
    unique_abstracts = get_unique_abstracts_by_doi(data)
    abstract_statistics = []
    observed_classes = set(RHETORICAL_CLASSES)

    for cited_doi, abstract_info in unique_abstracts.items():
        citing_articles = citation_index.get(cited_doi, {})
        rhetorical_counts = Counter()
        total_citation_passages = 0
        total_labelled_passages = 0

        for passages in citing_articles.values():
            for passage in passages:
                total_citation_passages += 1
                final_rhetorical_class = passage.get("rhetorical_class", "")

                if final_rhetorical_class:
                    rhetorical_counts[final_rhetorical_class] += 1
                    total_labelled_passages += 1
                    observed_classes.add(final_rhetorical_class)

        if total_citation_passages < minimum_citations:
            continue

        abstract_statistics.append({"cited_doi": cited_doi, "title": abstract_info["title"], "n_citing_dois": len(citing_articles), "n_citation_passages": total_citation_passages, "n_labelled_passages": total_labelled_passages, "rhetorical_counts": rhetorical_counts})

    ordered_classes = RHETORICAL_CLASSES.copy()

    for rhetorical_class in sorted(observed_classes):
        if rhetorical_class not in ordered_classes:
            ordered_classes.append(rhetorical_class)

    rows = []

    for abstract_statistic in abstract_statistics:
        row = {"cited_doi": abstract_statistic["cited_doi"], "title": abstract_statistic["title"], "n_citing_dois": abstract_statistic["n_citing_dois"], "n_citation_passages": abstract_statistic["n_citation_passages"], "n_labelled_passages": abstract_statistic["n_labelled_passages"]}

        for rhetorical_class in ordered_classes:
            count = abstract_statistic["rhetorical_counts"].get(rhetorical_class, 0)
            percentage = count / abstract_statistic["n_labelled_passages"] * 100 if abstract_statistic["n_labelled_passages"] > 0 else 0.0
            row[rhetorical_class + "_count"] = count
            row[rhetorical_class + "_percentage"] = percentage

        rows.append(row)

    return pd.DataFrame(rows), ordered_classes


def calculate_average_rhetorical_percentage(per_abstract_df, rhetorical_classes):
    average_rows = []

    for rhetorical_class in rhetorical_classes:
        percentage_column = rhetorical_class + "_percentage"

        if percentage_column not in per_abstract_df.columns:
            continue

        average_rows.append({"rhetorical_class": rhetorical_class, "n_abstracts": len(per_abstract_df), "average_percentage": per_abstract_df[percentage_column].mean(), "standard_deviation": per_abstract_df[percentage_column].std()})

    return pd.DataFrame(average_rows)




def create_average_rhetorical_percentage_outputs(data, citation_index, minimum_citations=5):
    per_abstract_output_path = "rhetorical_percentage_per_abstract_minimum_" + str(minimum_citations) + "_citations.csv"
    average_output_path = "average_rhetorical_percentage_minimum_" + str(minimum_citations) + "_citations.csv"
    plot_output_path = "average_rhetorical_percentage_minimum_" + str(minimum_citations) + "_citations.png"
    per_abstract_df, rhetorical_classes = build_rhetorical_percentage_per_abstract(data, citation_index, minimum_citations)
    average_df = calculate_average_rhetorical_percentage(per_abstract_df, rhetorical_classes)
    print("\nAbstracts included in the rhetorical-percentage analysis:", len(per_abstract_df))
    print("Minimum citation passages per abstract:", minimum_citations)
    print("\nAverage rhetorical percentage:")
    print(average_df.to_string(index=False))



def check_relation_entities(relation, rel_type, source_id, target_id, code_to_search_names):
    relation_source = relation.get("source")
    relation_target = relation.get("target")

    if rel_type == "has_vector":
        expected_source_code = target_id
        expected_target_code = source_id
    else:
        expected_source_code = source_id
        expected_target_code = target_id

    source_check = match_entity_to_eppo_code(relation_source, expected_source_code, code_to_search_names)
    target_check = match_entity_to_eppo_code(relation_target, expected_target_code, code_to_search_names)

    return {
        "source_matches": source_check["matches"],
        "target_matches": target_check["matches"],
        "both_match": source_check["matches"] and target_check["matches"],
        "source_entity": source_check["entity_name"],
        "source_matched_name": source_check["matched_eppo_name"],
        "target_entity": target_check["entity_name"],
        "target_matched_name": target_check["matched_eppo_name"],
        "expected_source_code": expected_source_code,
        "expected_target_code": expected_target_code
    }


def print_potential_interactions(data, citation_index, name_to_codes, code_to_preferred_name, code_to_search_names, aliases):
    total_interactions = 0
    total_abstracts = 0
    abstract_with_match = 0
    abstract_without_match = 0

    for interaction_key, interaction_obj in data.items():
        matched_interaction = interaction_obj.get("matched_interaction", {})
        rel_subtype = str(matched_interaction.get("rel_subtype", ""))
        source_id = matched_interaction.get("source_id", "")
        target_id = matched_interaction.get("target_id", "")
        rel_type = matched_interaction.get("rel_type", "")
        total_interactions += 1

        source_prefs_name = code_to_preferred_name.get(source_id, "")
        target_prefs_name = code_to_preferred_name.get(target_id, "")

        for abstract_record in interaction_obj.get("abstracts_with_this_interaction", []):
            prediction = abstract_record.get("qwen_abstract_prediction", {})
            prediction_content = get_prediction_content(prediction)
            parsed_prediction = extract_json(prediction_content)
            relations = get_relations(parsed_prediction)
            cited_doi = abstract_record.get("doi", "")
            match_found_for_abstract = False

            

            if relations:
                for relation_number, relation in enumerate(relations, start=1):
                    extracted_rel_type = normalize_relation_name(get_relation_label(relation))
                    relation_matches = False

                    if rel_type in {"has_vector", "is_vectorof"}:
                        relation_matches = extracted_rel_type == "transmits"

                    elif rel_type == "has_host":
                        relation_matches = extracted_rel_type in {"has been found on", "affects"}

                    if relation_matches:
                        entity_check = check_relation_entities(relation, rel_type, source_id, target_id, code_to_search_names)

                        if entity_check["both_match"]:
                            match_found_for_abstract = True
                        else:
                            pass
            else:
                print("No parsed relation found.")

            if match_found_for_abstract:
                abstract_with_match += 1
            else:
                print("\n" + "=" * 100)
                print("Interaction key:", interaction_key)
                print("source_prefs_name:", source_prefs_name)
                print("target_prefs_name:", target_prefs_name)
                print("EPPO source:", source_id)
                print("EPPO relation:", rel_type)
                print("EPPO modality:", rel_subtype)
                print("EPPO target:", target_id)
                print("DOI:", normalize_doi(cited_doi))
                print("Title:", abstract_record.get("title", ""))
                print("\nAbstract:")
                print(abstract_record.get("abstract", ""))
                print("\nRelations extracted from the abstract:")
                abstract_without_match += 1
                for relation_number, relation in enumerate(relations, start=1):
                    print(relation)

            print("At least one matching relation found for this abstract:", match_found_for_abstract)
            #print("\nCitations to this abstract:")
            #print_citations_for_cited_doi(cited_doi, citation_index)
            total_abstracts += 1

    print("\n" + "=" * 100)
    print("Potential EPPO interactions:", total_interactions)
    print("Abstracts printed:", total_abstracts)
    print("Abstracts with at least one matching relation:", abstract_with_match)
    print("Abstracts without a matching relation:", abstract_without_match)
    print("Counter verification:", abstract_with_match + abstract_without_match)


def inspect_abstract_doi_counts(data):
    total_records = 0
    unique_dois = set()
    missing_doi_records = 0
    missing_doi_abstracts = set()

    for interaction_obj in data.values():
        for abstract_record in interaction_obj.get("abstracts_with_this_interaction", []):
            total_records += 1
            cited_doi = normalize_doi(abstract_record.get("doi", ""))

            if cited_doi:
                unique_dois.add(cited_doi)
            else:
                missing_doi_records += 1
                title = str(abstract_record.get("title", "")).strip()
                abstract = str(abstract_record.get("abstract", "")).strip()
                missing_doi_abstracts.add((title, abstract))

    print("Total abstract-interaction records:", total_records)
    print("Unique non-empty cited DOIs:", len(unique_dois))
    print("Records with missing DOI:", missing_doi_records)
    print("Unique abstracts with missing DOI:", len(missing_doi_abstracts))


if __name__ == "__main__":
    input_path = "intersting_abstracts/interaction_abstracts_with_citing_passages_with_two_step_relations.json"
    graph_with_rc = "restructured_graph.json"
    code_eppo = "transfer_Marine/name_for_codes_in_biological_interactions.json"

    data = load_json(input_path)
    graph_rc = load_json(graph_with_rc)
    citation_index = build_citation_index_updated(graph_rc)
    eppo_codes = load_json(code_eppo)

    print("Cited DOIs indexed:", len(citation_index))
    name_to_codes, code_to_preferred_name, code_to_search_names, aliases = build_eppo_name_index(eppo_codes)
    print_potential_interactions(data, citation_index, name_to_codes, code_to_preferred_name, code_to_search_names, aliases)
    create_average_rhetorical_percentage_outputs(data, citation_index, minimum_citations=5)
    inspect_abstract_doi_counts(data)
  