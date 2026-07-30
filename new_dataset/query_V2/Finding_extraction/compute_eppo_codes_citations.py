import json
import re
from typing import Any, Dict
from EPPO_normalization_findings import extract_json, get_abbreviated_name, load_json
from tqdm import tqdm
from spanlib.spanseq.overlap import remove_overlaps
from projector import Projector
import projector.regularizers



def get_citation_text(passage: Dict[str, Any]) -> str:
    """
    Read the citation text from the graph passage.
    """
    value = passage.get("Full-text")

    if value is not None and str(value).strip():
        return str(value)

    return ""


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

def extract_relations_from_citations(graph: Dict[str, Any], entries):
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

            original_citation = get_citation_text(passage)
            citation = clean_sentence(original_citation)
            text = citation
            all_matches = list(proj.search(text))
            large_matches = remove_overlaps(all_matches)
            
            for start, end, (name, code) in large_matches:
                found_couple = (name, code)
                if found_couple not in passage["EPPO_entities"]:
                    passage["EPPO_entities"].append(found_couple)
    return graph


extracted_relations_from_abstracts_file = "output_prompt_epop/relation_prediction_Qwen_Qwen3-32B.json"
input_graph_with_rc_file = "../../scripts/Experiment_Citation_functionV2/graph_citations/graph_with_Jurgens_cfunc_BIOBERT.json"

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
updated_graph = extract_relations_from_citations(graph_with_rc, entries)
output_graph_file = "graph_citations/graph_with_EPPO_entities.json"
with open(output_graph_file, "w", encoding="utf-8") as file:
    json.dump(updated_graph, file, ensure_ascii=False, indent=2)

print(f"Updated graph saved to: {output_graph_file}")