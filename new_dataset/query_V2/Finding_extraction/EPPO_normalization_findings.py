import json
import re
from typing import Any, Dict
from spanlib.spanseq.overlap import remove_overlaps
from projector import Projector
import projector.regularizers
from tqdm import tqdm


def load_json(path: str) -> Any:
    """Load and return a JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def get_abbreviated_name(name):
    """
    Convert 'Juglans nigra' into 'j nigra' after cleaning.
    Works with clean_text output.
    """
    parts = name.split()

    if len(parts) < 2:
        return ""

    genus_initial = parts[0][0]
    species_or_rest = " ".join(parts[1:])

    return genus_initial + ". " + species_or_rest


def get_eppo_codes_abstracts(dic_relations_per_doi:list):
    eppo_entries_file = ""
    eppo_entries = json.load(eppo_entries_file)
    present_codes_dic = {}

    for code, names in eppo_entries.items():
        present_codes_dic[code]=[]

        for name in names:
            normalized_name = name.lower().strip()
            for relation in dic_relations_per_doi.get("relationships"):
                normalized_source = relation.get("source").lower().strip()
                normalized_target = relation.get("target").lower().strip()

                if normalized_name in normalized_source:
                    present_codes_dic[code].append(normalized_source)
                
                if normalized_name in normalized_target:
                    present_codes_dic[code].append(normalized_target)


def main():
    """Script to match extracted entities to EPPO entities and attribute an EPPO code for comparison to extracted citation entities"""

    extracted_relations_from_abstracts_file = "output_prompt_epop/relation_prediction_Qwen_Qwen3-32B.json"
    input_graph_with_rc_file = "../../scripts/Experiment_Citation_functionV2/graph_citations/graph_with_Jurgens_cfunc_BIOBERT.json"

    graph_abstract_info = load_json(extracted_relations_from_abstracts_file)
    graph_with_rc = load_json(input_graph_with_rc_file)

    json_eppo_codes = "EPPO_files/name_for_codes_in_biological_interactions.json"
    with open(json_eppo_codes, "r", encoding="utf-8") as file:
        eppo_codes = json.load(file)

    #create an EPPO dictionary with all the entities variants and abbreviation associated to an EPPO code
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
    
    #create a prefix tree with Projector
    proj = Projector(regularizers=(projector.regularizers.ignore_case,))
    proj.set_entries(entries.items())

    invalid_relation_count = 0
    missing_contribution_relations_count = 0

    for item in tqdm(graph_abstract_info, desc="Attributing EPPO codes"):
        if not isinstance(item, dict):
            print("\nSkipping invalid item:", item)
            continue

        item_relationships = []

        doi = str(item.get("DOI", "")).strip()

        if not doi:
            print("\nSkipping item without DOI")
            continue

        contribution_relations = item.get("contribution_relations", [])

        if not isinstance(contribution_relations, list):
            print("\nInvalid contribution_relations for DOI:", doi)
            missing_contribution_relations_count += 1
            continue

        for relation in contribution_relations:
            if not isinstance(relation, dict):
                invalid_relation_count += 1
                continue

            source = str(relation.get("source", "")).strip()
            target = str(relation.get("target", "")).strip()
            relation_type = str(relation.get("type", "")).strip()
            label_contribution = str(relation.get("label", "")).strip()

            if not source or not target or not relation_type:
                print("\nSkipping incomplete relation:", relation)
                invalid_relation_count += 1
                continue

            code_source = None
            code_target = None

            all_matches = list(proj.search(source))
            entire_matches = [(start, end, value) for start, end, value in all_matches if start == 0]
            large_matches = remove_overlaps(entire_matches)

            print("\nSource matches:")
            for start, end, (name, code) in large_matches:
                print("Found couple:")
                print((name, code))
                code_source = code

            all_matches = list(proj.search(target))
            entire_matches = [(start, end, value) for start, end, value in all_matches if start == 0]
            large_matches = remove_overlaps(entire_matches)

            print("\nTarget matches:")
            for start, end, (name, code) in large_matches:
                print("Found couple:")
                print((name, code))
                code_target = code

            relation_with_codes = {
                "source": source,
                "type": relation_type,
                "target": target,
                "code_source": code_source,
                "reltype": relation_type,
                "code_target": code_target,
                "contribution": label_contribution
            }

            item_relationships.append(relation_with_codes)

        print("\nContribution relations with EPPO codes:")
        print(item_relationships)
        item["contribution_relations"] = item_relationships

    print("\n================================")
    print("Invalid relations:", invalid_relation_count)
    print("Invalid contribution_relations fields:", missing_contribution_relations_count)
    print("================================")

    output_path = "output_prompt_epop/eppo_codes_relation_prediction_Qwen_Qwen3-32B.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(graph_abstract_info, file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()