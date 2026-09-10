import json
import re
import pandas as pd
from tqdm import tqdm

def clean_text(text):
    """
    Lowercase, remove repeated spaces, and safely handle NaN/None.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_eppo_index(code_names):
    """
    Build dictionary:
    {
        "CACV00": {
            "preferred": "Orthotospovirus capsiciflavi",
            "variants": [...]
        }
    }
    """
    index = {}

    for item in code_names:
        code_id = item.get("id")
        names = item.get("names", [])
        prefs = item.get("prefs", [])

        if not code_id or not names:
            continue

        preferred_name = None

        for i, name in enumerate(names):
            if i < len(prefs) and prefs[i] is True:
                preferred_name = name
                break

        if preferred_name is None:
            preferred_name = names[0]

        variants = [name for name in names if name != preferred_name]

        index[code_id] = {
            "preferred": preferred_name,
            "variants": variants,
            "all_names": names
        }

    return index


def name_found_in_text(preferred_name, variants, text, min_variant_length=4):
    """
    Returns True if the preferred name or one valid variant is found in text.
    Variants shorter than or equal to min_variant_length are ignored.
    """

    text = clean_text(text)
    preferred_name_clean = clean_text(preferred_name)

    if preferred_name_clean and preferred_name_clean in text:
        return True

    for variant in variants:
        variant_clean = clean_text(variant)

        if len(variant_clean) > min_variant_length and variant_clean in text:
            return True


    return False


def parse_eppo_biological_interaction_file(path_interactions, path_name_for_codes_eppo, output_path, abstracts, titles, dois, save_every=10):
    files_for_analysis = {}
    n_interesting_found = 0

    #Unique abstracts containing at least one potential/doubtful interaction
    concerned_abstract_indices = set()

    code_names = load_json(path_name_for_codes_eppo)
    enriched_interactions = load_json(path_interactions)

    index_eppo = build_eppo_index(code_names)

    for item in tqdm(enriched_interactions, desc="Scanning EPPO interactions"):
        rel_subtype = item.get("rel_subtype", "")
        source_id = item.get("source_id")
        target_id = item.get("target_id")

        if rel_subtype not in ("Potential vector", "Doubtful host"):
            continue

        if source_id not in index_eppo or target_id not in index_eppo:
            continue

        source_info = index_eppo[source_id]
        target_info = index_eppo[target_id]

        pref_name_source = source_info["preferred"]
        variants_source = source_info["variants"]

        pref_name_target = target_info["preferred"]
        variants_target = target_info["variants"]

        for i in range(len(abstracts)):
            abstract = clean_text(abstracts[i])
            title = clean_text(titles[i])
            doi = dois[i]
            full_text = title + " " + abstract
            source_found = name_found_in_text(pref_name_source, variants_source, full_text)
            target_found = name_found_in_text(pref_name_target, variants_target, full_text)

            if source_found and target_found:
                #Counts every abstract-interaction match
                n_interesting_found += 1

                #Counts this abstract only once
                concerned_abstract_indices.add(i)

                tqdm.write("-----------------------")
                tqdm.write(f"Interesting relation #{n_interesting_found}")
                tqdm.write(str(item))
                tqdm.write(f"Source preferred name: {pref_name_source}")
                tqdm.write(f"Source variants: {variants_source}")
                tqdm.write(f"Target preferred name: {pref_name_target}")
                tqdm.write(f"Target variants: {variants_target}")
                tqdm.write("Relation found in abstract!")
                tqdm.write(abstract)

                if doi not in files_for_analysis:
                    files_for_analysis[doi] = []

                files_for_analysis[doi].append({
                    "title": titles[i],
                    "abstract": abstracts[i],
                    "matched_interaction": item,
                    "source_id": source_id,
                    "source_preferred_name": pref_name_source,
                    "source_variants": variants_source,
                    "target_id": target_id,
                    "target_preferred_name": pref_name_target,
                    "target_variants": variants_target,
                    "rel_type": item.get("rel_type", ""),
                    "rel_subtype": rel_subtype
                })

                if n_interesting_found % save_every == 0:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(files_for_analysis, f, indent=2, ensure_ascii=False)

                    tqdm.write(f"Saved {n_interesting_found} abstract-interaction matches to {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(files_for_analysis, f, indent=2, ensure_ascii=False)

    n_concerned_abstracts = len(concerned_abstract_indices)

    tqdm.write("Final save complete.")
    tqdm.write(f"Total potential/doubtful abstract-interaction matches: {n_interesting_found}")
    tqdm.write(f"Total unique abstracts concerned by potential or doubtful interactions: {n_concerned_abstracts}")

    return files_for_analysis


path_ref_biological_interactions_eppo = "transfer_Marine/biological_interactions.json"
path_name_for_codes_eppo = "transfer_Marine/name_for_codes_in_biological_interactions.json"
output_path = "intersting_abstract/Files_for_analysis.json"

df_wos = pd.read_csv("../../references_WOS.tsv", sep="\t")

abstracts = df_wos["AB"].fillna("").to_list()
titles = df_wos["TI"].fillna("").to_list()
dois = df_wos["DI"].fillna("").to_list()

refs_parsed = parse_eppo_biological_interaction_file(path_ref_biological_interactions_eppo, path_name_for_codes_eppo, output_path, abstracts, titles, dois)