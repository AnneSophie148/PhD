import json
import re
from typing import Any
from tqdm import tqdm
from projector import Projector
import projector.regularizers
from spanlib.spanseq.overlap import remove_overlaps
import sys
sys.path.append("../..")
import pandas as pd

def extract_json(text: Any):
    text = str(text).strip()
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
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

def clean_name(name):
    name = str(name).strip().strip("'\"‘’“”«»`.,;:!?()[]{}").strip()
    name = name.replace("-", " ")
    return name

    
def other_identified_equivalence_eppo(term):
    equivalence_dic = {
        "acrythosiphon pisum": "Acyrthosiphon pisum",
        "acyriosphon pisum": "Acyrthosiphon pisum",
        "aegilops ovata": "egilope ovata",
        "agalia quadripunctata": "Agallia quadripunctata",
        "alternaria": "Alternaria sp.",
        "alpinia petiolata": "Alpinia petiolata",
        "clover yellow edge agent": "Clover yellow edge phytoplasma",
        "grapevine leafroll associated virus 4 lv": "Grapevine leafroll-associated virus 4",
        "rice orange leaf phytoplasma": "Rice orange leaf agent",
        "aphis cracclvora": "Aphis craccivora",
        "aphis gossipii": "Aphis gossypi",
        "aphis gossypi": "Aphis gossypi",
        "bactericerca cockerelli": "Bactericera cockerelli",
        "Brassica": "Brassica sp.",
        "bursephalenchus xylophilus": "Bursaphelenchus xylophilus",
        "candidatus liberibacter psyllaurous": "Candidatus Liberibacter solanacearum",
        "cavariella aegopodiae": "Cavariella aegopodii",
        "chenopodium ambrosiodes": "Chenopodium ambrosioides",
        "chili leaf curl virus": "Chilli leaf curl virus",
        "Chirothrips": "Chirothrips sp",
        "cotton leaf curl multan virus rajasthan": "Cotton leaf curl Multan virus",
        "Cupressus": "Cupressus sp",
        "cymopsis tetragonoloba":"Cyamopsis tetragonoloba",
        "diastrombus mkurangai": "Diostrombus mkurangai",
        "Dickeya": "Dickeya sp",
        "Euscelidius": "Euscelidius sp",
        "Euscelis":"Euscelis sp",
        "Fusarium": "Fusarium sp",
        "ca. p. solani": "candidatus phytoplasma solani"
    }

    term = remove_strain_or_biotype(term)
    return equivalence_dic.get(term.casefold(), term)

def find_eppo_code(scientific_name, proj):
    scientific_name = clean_name(scientific_name)
    all_matches = list(proj.search(scientific_name))
    entire_matches = [(start, end, value) for start, end, value in all_matches if start == 0 and end == len(scientific_name)]
    large_matches = remove_overlaps(entire_matches)

    if not large_matches:
        return None, None

    start, end, match_value = max(large_matches, key=lambda match: match[1] - match[0])
    matched_name, identified_code = match_value
    return matched_name, identified_code

def remove_strain_or_biotype(name):
    name = str(name).strip()
    name = re.sub(r"\s+16Sr[A-Z0-9]+(?:[\s-]+[A-Z0-9]+)*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+(?:strain|biotype|isolate|subgroup)\s+.+$", "", name, flags=re.IGNORECASE)
    return name.strip()

def get_agent_name_variants(name):
    variants = []
    suffixes = ("agent", "phytoplasma", "virus")

    for current_suffix in suffixes:
        if re.search(rf"\s+{current_suffix}$", name, flags=re.IGNORECASE):
            base_name = re.sub(
                rf"\s+{current_suffix}$",
                "",
                name,
                flags=re.IGNORECASE
            ).strip()

            for replacement_suffix in suffixes:
                if replacement_suffix != current_suffix:
                    variants.append(f"{base_name} {replacement_suffix}")

            break

    return variants

def get_virus_without_qualifier(name):
    patterns = [
        r"\s+(?:strain|isolate|variant|subgroup)\s+.+$",
        r"\s+\S+\s+(?:strain|isolate|variant|subgroup)$",
        r"\s+(?:strain|isolate|variant|subgroup)$"
    ]

    variants = []

    for pattern in patterns:
        simplified = re.sub(pattern, "", name, flags=re.IGNORECASE).strip()

        if simplified and simplified != name and simplified not in variants:
            variants.append(simplified)

    return variants


def get_format_variants(name):
    equivalent_forms = [
        ("leaf roll", "leafroll"),
        ("seed borne", "seedborne"),
        ("aphid borne", "aphid-borne"),
        ("whitefly borne", "whitefly-borne"),
        ("black streaked", "black-streaked"),
        ("novo ulmi", "novo-ulmi")
    ]


    variants = []

    for form_1, form_2 in equivalent_forms:
        if form_1 in name.casefold():
            variant = re.sub(re.escape(form_1), form_2, name, flags=re.IGNORECASE)
            variants.append(variant)

        if form_2 in name.casefold():
            variant = re.sub(re.escape(form_2), form_1, name, flags=re.IGNORECASE)
            variants.append(variant)

    return variants

def get_plant_name_variants(name):
    variants = []

    without_cultivar = re.sub(r"\s+cv\.\s+.+$", "", name, flags=re.IGNORECASE).strip()

    if without_cultivar != name:
        variants.append(without_cultivar)

    without_accession = re.sub(r"\s+(?:IAC|CIFC)\s+[A-Za-z0-9/-]+$", "", name, flags=re.IGNORECASE).strip()

    if without_accession != name:
        variants.append(without_accession)

    without_trailing_accession = re.sub(r"\s+\d[\dA-Za-z ./-]*$", "", name).strip()

    if without_trailing_accession != name:
        variants.append(without_trailing_accession)

    return variants

def get_uncertain_taxonomic_variants(name):
    variants = []

    simplified_name = re.sub(r"\s+(?:nr|aff)\.\s+", " ", name, flags=re.IGNORECASE)

    if simplified_name != name:
        variants.append(simplified_name)

    return variants

def get_name_candidates(scientific_name):
    candidates = []

    def add_candidate(candidate):
        candidate = clean_name(candidate)

        if candidate and candidate not in candidates:
            candidates.append(candidate)

    #scientific_name = normalize_taxonomic_abbreviations(scientific_name)
    scientific_name = remove_strain_or_biotype(scientific_name)
    scientific_name = other_identified_equivalence_eppo(scientific_name)

    add_candidate(scientific_name)

    for variant in get_virus_without_qualifier(scientific_name):
        add_candidate(variant)

    for variant in get_agent_name_variants(scientific_name):
        add_candidate(variant)

    for variant in get_format_variants(scientific_name):
        add_candidate(variant)

    for variant in get_plant_name_variants(scientific_name):
        add_candidate(variant)

    for variant in get_uncertain_taxonomic_variants(scientific_name):
        add_candidate(variant)

    return candidates



def try_match_eppo(entity_term, eppo_projector):
    identified_code = ""
    scientific_name = clean_name(entity_term)
    
    #noticed terms to correct and normalize to eppo format
    if "corn stunt spiroplasma" in scientific_name.lower().strip():
        scientific_name = "stunt of maize"

    if "bemisia" in scientific_name.lower().strip() and "tabaci" in scientific_name.lower().strip():
        scientific_name = "Bemisia tabaci"
    
    if "fitis vinifera" in scientific_name.lower().strip() and not "subsp." in scientific_name.lower().strip() and not "labrusca" in scientific_name.lower().strip():
        scientific_name = "Vitis vinifera"
    
    if "flavescence dor" in scientific_name.lower().strip() and "phytoplasma" in scientific_name.lower().strip():
        #there where mistakes because of the accentuation
        scientific_name = "Grapevine flavescence dorée phytoplasma"

    if "plum pox virus" in scientific_name.lower().strip():
        scientific_name = "Potyvirus plumpoxi"
    
    if "tristeza virus" in scientific_name.lower().strip():
        scientific_name = "Citrus tristeza virus"

    if "potato virus y" in scientific_name.lower().strip():
        scientific_name = "Potato virus Y"

    if "tobacco necrosis virus" in scientific_name.lower().strip():
        scientific_name = "Tobacco necrosis virus"

    if "cucumber mosaic virus" in scientific_name.lower().strip():
        scientific_name = "Cucumber mosaic virus"

    
    original_scientific_name = entity_term
    matched_name = None
    identified_code = None
    matched_candidate = None

    for name_candidate in get_name_candidates(scientific_name):
        matched_name, identified_code = find_eppo_code(name_candidate, eppo_projector)

        if identified_code:
            matched_candidate = name_candidate
            scientific_name = name_candidate
            break

    if identified_code and matched_candidate != clean_name(original_scientific_name):
        print(f"\nScientific name: {scientific_name}")
        print(f"\nLLM scientific name: {original_scientific_name}")
        print(f"Matched candidate: {matched_candidate}")
        print(f"Matched EPPO name: {matched_name}")
        print(f"Code identified: {identified_code}")

    if not identified_code:
        for name_variant in get_virus_without_qualifier(scientific_name):
            matched_name, identified_code = find_eppo_code(name_variant, eppo_projector)

            if identified_code:
                print(f"\nCode identified after removing virus qualifier: {name_variant}")
                scientific_name = name_variant
                break

    if not identified_code:
        for name_variant in get_agent_name_variants(scientific_name):
            matched_name, identified_code = find_eppo_code(name_variant, eppo_projector)

            if identified_code:
                print(f"\nCode identified using name variant: {name_variant}")
                scientific_name = name_variant
                break
    

    if not identified_code:
        simplified_name = re.sub(r"\s+(phytoplasma|virus)\s*$", "", scientific_name, flags=re.IGNORECASE).strip()
        simplified_name = simplified_name.replace("-", " ")
    
        simplified_name = re.sub(r"\bspp\.?$", "sp.", simplified_name, flags=re.IGNORECASE)

        if simplified_name != scientific_name:
            matched_name, identified_code = find_eppo_code(simplified_name, eppo_projector)

            if identified_code:
                print(f"\nScientific name: {scientific_name}")
                print(f"Code identified after simplifying name: {identified_code}")
                print(f"LLM scientific name: {scientific_name}")
                print(f"Simplified name: {simplified_name}")
                print(f"Matched EPPO name: {matched_name}")

    if not identified_code and len(scientific_name.split()) == 1:
        name_with_sp = f"{scientific_name} sp."
        matched_name, identified_code = find_eppo_code(name_with_sp, eppo_projector)

        if identified_code:
            print(f"\nScientific name: {scientific_name}")
            print(f"Code identified after adding sp.: {identified_code}")
            print(f"Name with sp.: {name_with_sp}")
            print(f"Matched EPPO name: {matched_name}")
            scientific_name = name_with_sp


    return identified_code
