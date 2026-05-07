import pickle
import json
import re

def is_reference_line(line):
    """Check if there is a date and a Name"""
    if is_comment_line(line):
        return False
    return (bool(re.search(r"\([12][0-9]{3}\)", line)) and bool(re.search(r"[A-Z][a-z]+", line)))


def is_comment_line(line):
    """Check if the line is a comment"""
    if re.match(r"^\s*-{3,}", line) or "-----" in line:
        return True
    return False

def clean_bibref(bibref):
    """Normalize newline markers while preserving DOIs."""

    doi_placeholders = {}

    def protect_doi(match):
        key = f"__DOI_{len(doi_placeholders)}__"
        doi_placeholders[key] = match.group(0)
        return key

    #Save doi not to cut it when normalizing
    bibref = re.sub(r"10\.\d{4,9}/\S+", protect_doi, bibref)

    #Normalize newline markers
    bibref = bibref.replace("/n/n", "\n")
    bibref = re.sub(r"/n|\\n", "\n", bibref)

    #Restore DOIs
    for key, doi in doi_placeholders.items():
        bibref = bibref.replace(key, doi)

    lines = [l.strip() for l in bibref.split("\n") if l.strip()]

    return lines


def parse_single_bibref(item):
    """Parse references from an item from the EPPO graph"""
    source_id = item.get("source_id")
    target_id = item.get("target_id")
    rel_type = item.get("rel_type")
    rel_subtype = item.get("rel_subtype")
    bibref = item.get("bibref")
    

    if not bibref:
        return []

    lines = clean_bibref(bibref)

    refs = []
    current_ref = None
    current_comment = ""

    for line in lines:
        print(f"\nProcessing line {line}")
        #detect if line is a new reference, a comment, or a piece of broken reference
        
        #most references start with the caracter "*" but not all
        is_new_ref = (line.startswith("*") or is_reference_line(line))
        comment = is_comment_line(line)

        if is_new_ref:
            print("--> Detected reference !")
            if current_ref:
                refs.append({
                    "source_id": source_id,
                    "target_id": target_id,
                    "rel_type": rel_type,
                    "rel_subtype": rel_subtype,
                    "ref": current_ref.strip(),
                    "comment": current_comment.strip()
                })

            current_ref = line.lstrip("* ").strip()
            current_comment = ""

        # 2. comment line
        elif comment:
            current_comment = line
            print(" --> Line is a comment")

        else:
            #not dected neither as a comment nor as a reference --> considered as broken reference to reconstitute
            if current_ref:
                current_ref += " " + line
                print("\nCorrected ref : ", current_ref)

    # save last ref
    if current_ref:
        refs.append({
            "source_id": source_id,
            "target_id": target_id,
            "rel_type": rel_type,
            "rel_subtype": rel_subtype,
            "ref": current_ref.strip(),
            "comment": current_comment.strip()
        })


    return refs


def parse_eppo_biological_interaction_file(path, output_path):
    all_refs = []
    cleaned_data = []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        parsed_refs = parse_single_bibref(item)

        #cleaned item
        new_json = {
            "source_id": item.get("source_id"),
            "target_id": item.get("target_id"),
            "rel_type": item.get("rel_type"),
            "rel_subtype": item.get("rel_subtype"),
            "bibref": {
                "refs": [r["ref"] for r in parsed_refs],
                "comments": [r["comment"] for r in parsed_refs]
            }
        }

        cleaned_data.append(new_json)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    return all_refs


path_ref_biological_interactions_eppo = "transfer_Marine/biological_interactions.json"
path_name_for_codes_eppo = "transfer_Marine/name_for_codes_in_biological_interactions.json"
output_path = "cleaned_biological_interactions_EPPO.json"
refs_parsed = parse_eppo_biological_interaction_file(path_ref_biological_interactions_eppo, output_path)