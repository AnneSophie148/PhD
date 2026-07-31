import json
import re
from typing import Any, Dict
from EPPO_normalization_findings import extract_json, get_abbreviated_name, load_json
from tqdm import tqdm
from spanlib.spanseq.overlap import remove_overlaps
from projector import Projector
import projector.regularizers

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

def extract_relations_from_citations(graph: Dict[str, Any], entries, finegrained_passages):
    """
    Iterate through the citation passages in the graph.
    Add the relation-processing code as EPPO_entities
    """
    edges = graph.get("edges", [])
    proj = Projector(regularizers=(projector.regularizers.ignore_case,))
    proj.set_entries(entries.items())

    for edge_index, edge in enumerate(tqdm(edges, desc="Graph edges")):
        citation_passages = edge.get("citation_passages", [])
        for passage_index, passage in enumerate(citation_passages):
            passage["EPPO_entities"]=[]
            if not isinstance(passage, dict):
                continue

            original_citation = passage.get("Full-text")
            citation = clean_sentence(original_citation)
            text = citation
            all_matches = list(proj.search(text))
            large_matches = remove_overlaps(all_matches)
            
            for start, end, (name, code) in large_matches:
                found_couple = (name, code)
                if found_couple not in passage["EPPO_entities"]:
                    passage["EPPO_entities"].append(found_couple)
                    passage_key = (normalize_doi(citing_doi), normalize_doi(cited_doi), str(text).strip())
                    finegrained_passage = finegrained_passages.get(passage_key)
                    rhetorical_class = get_corrected_rhetorical_class(finegrained_passage=finegrained_passage, passage=passage)
                    passage["predicted_rhetorical_class"]=rhetorical_class
    return graph


extracted_relations_from_abstracts_file = "output_prompt_epop/relation_prediction_Qwen_Qwen3-32B.json"
input_graph_with_rc_file = "../graph_citations/graph_with_Jurgens_cfunc_BIOBERT.json"
graph_with_finegrained_rc = load_json("V2_graph_with_Jurgens_finegrained_coco_compare_contrast_label3_without_thinking.json")

graph_abstract_info = load_json(extracted_relations_from_abstracts_file)
graph_with_rc = load_json(input_graph_with_rc_file)

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
updated_graph = extract_relations_from_citations(graph_with_rc, entries)
output_graph_file = "../graph_citations/graph_with_EPPO_entities.json"
with open(output_graph_file, "w", encoding="utf-8") as file:
    json.dump(updated_graph, file, ensure_ascii=False, indent=2)

print(f"Updated graph saved to: {output_graph_file}")