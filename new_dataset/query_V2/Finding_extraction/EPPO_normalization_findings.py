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

def extract_json(text: Any):
    text = str(text).strip()
    # 1. Extract content inside ```json ... ```
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)

    if match:
        text = match.group(1)

    # 2. Try parsing the complete text
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return [parsed]

        if isinstance(parsed, list):
            return parsed

    except (json.JSONDecodeError, TypeError):
        pass

    #3. otherwise extract multiple JSON objects
    objects = []
    brace_level = 0
    start = None

    for i, character in enumerate(text):
        if character == "{":
            if brace_level == 0:
                start = i

            brace_level += 1

        elif character == "}":
            brace_level -= 1

            if brace_level == 0 and start is not None:
                chunk = text[start:i + 1]

                try:
                    objects.append(json.loads(chunk))
                except json.JSONDecodeError:
                    pass

                start = None

    if not objects:
        print("\nERROR parsing:")
        print(text)
        raise ValueError("No valid JSON objects found")

    return objects

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

    dic_relations_per_doi = {}
    parsing_error_count = 0

    for item in tqdm(graph_abstract_info, desc="Abstract relation predictions"):
        if not isinstance(item, dict):
            print("\nSkipping invalid item:", item)
            continue

        item["relationships"] = []
        item["EPPO_entities"] = []

        doi = item.get("DOI")
        content = item.get("content", "")

        if not doi:
            print("Skipping item without DOI")
            continue

        try:
            parsed = extract_json(content)

        except ValueError as error:
            parsing_error_count += 1
            print("\n--------------------------------")
            print("PARSING ERROR")
            print("DOI:", doi)
            print("Error:", error)
            print("Content:")
            print(content)
            print("--------------------------------")

            continue

        dic_relations_per_doi[doi] = parsed

        for parsed_object in parsed:
            if not isinstance(parsed_object, dict):
                continue

            relationships = parsed_object.get("relationships", [])
            if relationships:
                for relation in relationships:
                    if not isinstance(relation, dict):
                        continue

                    source = str(relation.get("source", "")).strip()
                    target = str(relation.get("target", "")).strip()
                    relation_type = str(relation.get("type", "")).strip()
                    code_source = None
                    code_target = None

                    all_matches = list(proj.search(source))
                    entire_matches = [(start, end, value) for start, end, value in all_matches if start == 0]
                    large_matches = remove_overlaps(entire_matches)

                    for start, end, (name, code) in large_matches:
                        found_couple = (name, code)
                        code_source = code

                    all_matches = list(proj.search(target))
                    entire_matches = [(start, end, value) for start, end, value in all_matches if start == 0]

                    #keep only the lnogest match when terms overlap
                    large_matches = remove_overlaps(entire_matches)

                    for start, end, (name, code) in large_matches:
                        found_couple = (name, code)
                        code_target = code

                    relation_with_codes = {"source": source,
                        "type": relation_type,
                        "target": target,
                        "code_source": code_source,
                        "reltype": relation_type,
                        "code_target": code_target}

                    item["relationships"].append(relation_with_codes)

                    if code_source is not None and code_target is not None:
                        item["EPPO_entities"].append(relation_with_codes)


    print("\n================================")
    print("Total parsing errors:", parsing_error_count)
    print("Successfully parsed DOIs:", len(dic_relations_per_doi))

    #create output with the attributed codes
    output_path = "output_prompt_epop/eppo_codes_relation_prediction_Qwen_Qwen3-32B.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(graph_abstract_info, file, ensure_ascii=False, indent=2)

