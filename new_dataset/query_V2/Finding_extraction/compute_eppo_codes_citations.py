import json
import re
from typing import Any, Dict, Optional, Tuple
from EPPO_normalization_findings import extract_json, get_abbreviated_name, load_json
from tqdm import tqdm
from spanlib.spanseq.overlap import remove_overlaps
from projector import Projector
import projector.regularizers
from module_try_match_eppo_code import *
from unidecode import unidecode

def normalize_doi(doi: Any) -> str:
    """Normalize a DOI so that graph passages can be matched."""
    normalized = unidecode(str(doi or "").lower().strip())
    normalized = re.sub(r"^(https?://(?:dx\.)?doi\.org/|doi:\s*)", "", normalized)
    return normalized.rstrip(".,")

def get_finegrained_passages(graph_with_finegrained_rc):
    finegrained_passages = {}

    for edge in graph_with_finegrained_rc.get("edges", []):
        cited_doi = normalize_doi(edge.get("cited_doi"))
        citing_doi = normalize_doi(edge.get("citing_doi"))

        for passage in edge.get("citation_passages", []):
            if not isinstance(passage, dict):
                continue

            text = str(passage.get("Full-text", "")).strip()
            finegrained_passages[(citing_doi, cited_doi, text)] = passage

    return finegrained_passages

def clean_sentence(sentence: Any) -> str:
    """Remove XML/HTML tags while preserving their textual content."""
    if sentence is None:
        return ""

    cleaned_sentence = str(sentence)
    cleaned_sentence = re.sub(r"<ref[^>]*>", "", cleaned_sentence)
    cleaned_sentence = re.sub(r'type="[^"]*"\s*target="[^"]*">|type="bibr">', "", cleaned_sentence)
    cleaned_sentence = re.sub(r"</ref>|<ref", "", cleaned_sentence)
    cleaned_sentence = re.sub(r"<[^>]+>", "", cleaned_sentence)
    cleaned_sentence = re.sub(r"\s+", " ", cleaned_sentence)

    return cleaned_sentence.strip()

def get_corrected_rhetorical_class(passage, finegrained_passage=None):
    rhetorical_class = passage.get("predicted_rhetorical_class")

    if rhetorical_class is None or str(rhetorical_class).strip().lower() in {"", "none", "null", "nan"}:
        return "unknown"

    rhetorical_class = str(rhetorical_class).strip()

    if rhetorical_class.lower() != "compareorcontrast":
        return rhetorical_class

    if finegrained_passage is not None:
        label2 = str(finegrained_passage.get("label2", "")).strip().lower()
        label3 = str(finegrained_passage.get("label3", "")).strip().lower()
    else:
        label2 = str(passage.get("label2", "")).strip().lower()
        label3 = str(passage.get("label3", "")).strip().lower()

    if label2 == "method":
        return "CoCoGM"

    if label2 == "result":
        label3_mapping = {
            "other": "background",
            "compare": "compare results",
            "contrast": "contrast results",
            "contrast - contradictory results": "contrast – contradictory results"
        }

        return label3_mapping.get(label3, rhetorical_class)

    return "compareorcontrast"

def get_extracted_relationships(passage: Dict[str, Any]):
    relation_output = passage.get("relations", [])

    if isinstance(relation_output, dict):
        relation_output = [relation_output]

    if not isinstance(relation_output, list):
        return []

    relationships = []

    for item in relation_output:
        if not isinstance(item, dict):
            continue

        nested_relationships = item.get("relationships")

        if nested_relationships is None:
            nested_relationships = item.get("relations")

        if isinstance(nested_relationships, list):
            relationships.extend(
                relation for relation in nested_relationships
                if isinstance(relation, dict)
            )
        elif "source" in item and "target" in item:
            relationships.append(item)

    return relationships

def find_eppo_match(entity_text: Any, eppo_projector: Projector) -> Optional[Tuple[str, str]]:
    """
    Match an extracted source or target entity to an EPPO name.

    Only matches beginning at position zero are accepted, following the rule
    used in the abstract-relation normalization script. When several synonyms
    match, the longest match is retained.
    """
    if not entity_text:
        return None
    all_matches = list(eppo_projector.search(entity_text))
    prefix_matches = [(start, end, value) for start, end, value in all_matches if start == 0]
    non_overlapping_matches = list(remove_overlaps(prefix_matches))
    if not non_overlapping_matches:
        return None
    _, _, (matched_name, code) = max(non_overlapping_matches, key=lambda match: match[1] - match[0])
    return matched_name, code

def extract_relations_from_citations(graph: Dict[str, Any], entries, finegrained_passages):
    """Add EPPO entities and EPPO codes to citation passages and extracted relationships."""
    edges = graph.get("edges", [])
    proj = Projector(regularizers=(projector.regularizers.ignore_case,))
    proj.set_entries(entries.items())

    for edge in tqdm(edges, desc="Graph edges"):
        cited_doi = edge.get("cited_doi")
        citing_doi = edge.get("citing_doi")

        for passage in edge.get("citation_passages", []):
            if not isinstance(passage, dict):
                continue

            passage["EPPO_entities"] = []
            raw_text = str(passage.get("Full-text", "")).strip()
            text = clean_sentence(raw_text)

            all_matches = list(proj.search(text))
            large_matches = list(remove_overlaps(all_matches))

            for start, end, (name, code) in large_matches:
                found_couple = (name, code)
                if found_couple not in passage["EPPO_entities"]:
                    passage["EPPO_entities"].append(found_couple)

            passage_key = (normalize_doi(citing_doi), normalize_doi(cited_doi), raw_text)
            finegrained_passage = finegrained_passages.get(passage_key)
            passage["predicted_rhetorical_class"] = get_corrected_rhetorical_class(passage, finegrained_passage)

            relationships = get_extracted_relationships(passage)
            for relationship in relationships:
                source = str(relationship.get("source") or "").strip()
                target = str(relationship.get("target") or "").strip()

                source_match = find_eppo_match(source, proj)
                target_match = find_eppo_match(target, proj)

                if source_match:
                    source_name, source_code = source_match
                else:
                    source_name = source
                    source_code = try_match_eppo(source, proj) if source else None

                if target_match:
                    target_name, target_code = target_match
                else:
                    target_name = target
                    target_code = try_match_eppo(target, proj) if target else None

                relationship["code_source"] = source_code or None
                relationship["code_target"] = target_code or None

                if source_code:
                    source_entity = (source_name, source_code)
                    if source_entity not in passage["EPPO_entities"]:
                        passage["EPPO_entities"].append(source_entity)

                if target_code:
                    target_entity = (target_name, target_code)
                    if target_entity not in passage["EPPO_entities"]:
                        passage["EPPO_entities"].append(target_entity)

    return graph


graph_with_extracted_relations_file = "graph_citations/graph_with_Jurgens_cfunc_BIOBERT_Qwen_relations.json"

extracted_relations_from_abstracts_file = "output_prompt_epop/relation_prediction_Qwen_Qwen3-32B.json"
input_graph_with_rc_file = "../graph_citations/graph_with_Jurgens_cfunc_BIOBERT.json"
graph_with_finegrained_rc = load_json("V2_graph_with_Jurgens_finegrained_coco_compare_contrast_label3_without_thinking.json")
graph_citation_relation = load_json(graph_with_extracted_relations_file)

json_eppo_codes = "EPPO_files/name_for_codes_in_biological_interactions.json"

with open(json_eppo_codes, "r", encoding="utf-8") as file:
    eppo_codes = json.load(file)

entries = {}

for item in eppo_codes:
    code = item.get("id", "")
    names = item.get("names", [])
    if not code or not isinstance(names, list):
        continue

    for name in names:
        name = str(name).strip()
        if len(name) > 4:
            entries[name] = code
            if len(name.split(" ")) >= 2:
                abbreviated_name = get_abbreviated_name(name)
                entries[abbreviated_name] = code

#V1 --> the relation extract will be updated with the Qwen relation extraction output
finegrained_passages = get_finegrained_passages(graph_with_finegrained_rc)
updated_graph = extract_relations_from_citations(graph_citation_relation, entries, finegrained_passages)
output_graph_file = "../graph_citations/graph_with_EPPO_entities.json"
with open(output_graph_file, "w", encoding="utf-8") as file:
    json.dump(updated_graph, file, ensure_ascii=False, indent=2)

print(f"Updated graph saved to: {output_graph_file}")