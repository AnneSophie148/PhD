import json
import sys
sys.path.append("../analyses_numbers_query_citing/")
sys.path.append("../")
from get_details_union_missing_doi import get_set_citing_dois, analyse_repartition_data_graph
from utils import normalize_doi
import pandas as pd
import unicodedata
from unidecode import unidecode
import string
from collections import defaultdict
from pathlib import Path
import re

def clean_text(text):
    '''Supprime la ponctuation, guillemets, tirets, slashes, et retourne le texte nettoyé.'''
    if text is None:
        return ""
    text = unidecode(text.lower())
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace("’", "").replace("–", "").replace("—", "").replace("“", "").replace("”", "")
    text = text.replace("-", "").replace("/", "").replace("_", "").replace("  ", "")
    return text.strip()

def build_eppo_index(code_names):
    index = {}
    
    for item in code_names:
        code_id = item.get("id")
        names = item.get("names", [])
        
        for n in names:
            cleaned = clean_text(n)
            index[cleaned] = code_id

    return index

def extract_json(text):
    text = text.strip()

    #1. extract inside ```json ... ```
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)

    #2. try direct full json first
    try:
        parsed = json.loads(text)
        # normalize output
        if isinstance(parsed, dict):
            return [parsed]

        if isinstance(parsed, list):
            return parsed

    except Exception:
        pass

    #3. fallback: extract multiple json objects safely --> case when there are several claims
    objects = []
    brace_level = 0
    start = None

    for i, c in enumerate(text):
        if c == "{":
            if brace_level == 0:
                start = i
            brace_level += 1

        elif c == "}":
            brace_level -= 1
            if brace_level == 0 and start is not None:
                chunk = text[start:i+1]
                try:
                    objects.append(json.loads(chunk))
                except:
                    pass

    if not objects:
        raise ValueError("No valid JSON objects found")
    return objects

def get_EPPO_code(entity, eppo_index):
    """Get EPPO code for entity extracted from scientific articles"""
    return eppo_index.get(clean_text(entity), "")


def normalize_relation(r):
    rel = r.get("relation")
    if rel == "is_vector_of":
        rel = "is_vectorof"
    return rel


def get_qwen_finding_extraction(doi, dic_index_extractfinding_by_doi, eppo_index, number_error_format):
    """
    Function that parses the model predictions and return a dictionary 
    {claim_str:[{finding1}, {finding2}], claim:{[] }
    """
    entry = dic_index_extractfinding_by_doi.get(normalize_doi(doi), {})
    raw_content = entry.get("content", "")
    #i will consider one empty finding per doc, if there are two claims and only one empty finding i do not consider it since it might be an error with the extracted claim
    empty_finding = 0
    abstract = entry.get("Abstract")
    #print("Abstract : ")
    #print(abstract)

    try:
        parsed = extract_json(raw_content)

    except Exception as e:
        #print(f"ERROR with prediction from doi {doi}: {e}")
        number_error_format +=1
        return {}, number_error_format, empty_finding, abstract

    claim_findings = {}

    if isinstance(parsed, dict):
        parsed = [parsed]
    
    for item in parsed:

        claim = item.get("claim", None)
        findings = item.get("findings", [])
        
        if claim is None:
            continue

        if not findings:
            empty_finding = 1

        claim_findings[claim] = findings

    return claim_findings, number_error_format, empty_finding, abstract


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_doi_index(extracted_qwen_findings):
    dic = {}

    for item in extracted_qwen_findings:
        doi = item.get("DOI")
        dic[normalize_doi(doi)] = {"Abstract": item.get("Abstract"), "content": item.get("content", "")}

    return dic

def build_code_to_pref_name(code_names):
    mapping = {}

    for item in code_names:
        names = item.get("names", [])
        prefs = item.get("prefs", [])

        preferred = next((n for n, p in zip(names, prefs) if p), None)
        mapping[item.get("id")] = preferred

    return mapping

def load_or_build(path, builder_fn):
    path = Path(path)

    if path.exists():
        return load_json(path)

    data = builder_fn()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data

def build_indices(config, extracted, code_names, INDEX_DIR, ref_cache, unique_ref_list):
    doi_index = load_or_build(INDEX_DIR / "doi_index.json", lambda: build_doi_index(extracted))
    code_map = load_or_build(INDEX_DIR / "code_to_pref_name.json", lambda: build_code_to_pref_name(code_names))
    eppo_index = load_or_build(INDEX_DIR / "code_to_names.json", lambda: build_eppo_index(code_names))
    graph_data = load_or_build(INDEX_DIR / "graph_data.json", lambda: build_common_EPPO_graph_data(config, ref_cache, unique_ref_list))
    return doi_index, code_map, eppo_index, graph_data

def get_dois_from_graph(config):
    df = pd.read_csv(config["Query_WOS_result"], sep="\t").fillna("")
    dois_wos = [normalize_doi(d) for d in df["DI"].dropna().tolist()]
    citing_articles = get_set_citing_dois(config["OpenAlex_art_citing_query"], config["COCI_art_citing_query"])

    return citing_articles, dois_wos

def build_common_EPPO_graph_data(config, ref_cache, unique_ref_list):
    citing_articles, dois_wos = get_dois_from_graph(config)

    graph_ref_set, set_ref_in_query_art, set_ref_in_citing_art = (analyse_repartition_data_graph(citing_articles, ref_cache, unique_ref_list, dois_wos))

    return {
        "citing_articles": list(citing_articles),
        "dois_wos": dois_wos,
        "graph_ref_set": list(graph_ref_set),
        "set_ref_in_query_art": list(set_ref_in_query_art),
        "set_ref_in_citing_art": list(set_ref_in_citing_art),
    }

def load_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def build_eppo_relations(doi, enriched_interactions, code_to_pref_name):
    eppo_relations = set()

    for interaction in enriched_interactions:
        rel_type = interaction.get("rel_type")
        rel_subtype = interaction.get("rel_subtype")
        source_id = interaction.get("source_id")
        target_id = interaction.get("target_id")

        bibref = interaction.get("bibref", {})
        in_graph = bibref.get("in_graph", [])

        for e in in_graph:
            for source_type, article_data in e.items():

                if source_type != "Query_WOS":
                    continue

                if not isinstance(article_data, dict):
                    continue

                graph_doi = normalize_doi(article_data.get("doi"))

                if graph_doi == normalize_doi(doi):

                    source_name = code_to_pref_name.get(source_id)
                    target_name = code_to_pref_name.get(target_id)

                    eppo_relations.add((source_name, target_name, rel_type, rel_subtype))

    return eppo_relations

def remove_parenthetical(text):
    if not text:
        return text

    return text.split("(")[0].strip()

def build_qwen_relations(parsed, eppo_index, code_to_pref_name, total_findings_extracted, relation_with_missing_code):
    """Function that takes the claim dictionary as input and returns a set of the relations from the findings"""
    qwen_relations = set()

    for claim, findings in parsed.items():
        #print("Claim :")
        #print(claim)

        for f in findings:
            total_findings_extracted += 1

            source = f.get("source")
            target = f.get("target")
            #print("\nSource : ", source)
            #print("TArget : ",target)

            rel = normalize_relation(f)
            subrel = f.get("subrelation")

            source_code = get_EPPO_code(source, eppo_index)
            target_code = get_EPPO_code(target, eppo_index)

            if not source_code:
                #print("NO SOURCE CODE FOUND")

                source_clean = remove_parenthetical(source)
                source_code = get_EPPO_code(source_clean, eppo_index)

                '''if source_code:
                    print("SOURCE CODE FOUND AFTER CLEANING:")
                    print(source_code)'''
            
            if not target_code:
                #print("NO TARGET CODE FOUND")

                target_clean = remove_parenthetical(target)
                target_code = get_EPPO_code(target_clean, eppo_index)

                '''if target_code:
                    print("TARGET CODE FOUND AFTER CLEANING:")
                    print(target_code)'''

            if not source_code or not target_code:
                relation_with_missing_code +=1

            #print("Source code : ", source_code)
            #print("Target code : ", target_code)

            source_name = code_to_pref_name.get(source_code)
            target_name = code_to_pref_name.get(target_code)

            qwen_relations.add((source_name, target_name, rel, subrel))

    return qwen_relations, total_findings_extracted, relation_with_missing_code

def compare_relations(eppo_relations, qwen_relations):
    reltype_only_matches = []
    subrel_only_matches = []
    full_matches = []
    no_matches = []
    full_match = False
    rel_match_only = False
    subrel_match_only = False

    for (s1, t1, r1, sr1) in eppo_relations:
        if sr1 == "Major host":
            sr1 == "Host"

        matched_any = False

        for (s2, t2, r2, sr2) in qwen_relations:
            if not s2 or not t2:
                continue

            if r2 == "is_vector_of":
                r2 = "is_vectorof"

            if s1 == s2 and t1 == t2:
                #same source and same target
                #maybe consider inversion

                if r1 == r2 and sr1 == sr2:
                    matched_any = True
                    full_matches.append((s1, t1, r1, sr1))
                    full_match = True
                    

                elif r1 == r2:
                    matched_any = True
                    rel_match_only = True
                    reltype_only_matches.append((s1, t1, r1, sr1))

                elif sr1 == sr2:
                    matched_any = True
                    subrel_overlap = True
                    subrel_only_matches.append((s1, t1, r1, sr1))
                    subrel_match_only = True
    

        if not matched_any:
            for (s2, t2, r2, sr2) in qwen_relations:
                if not s2 or not t2:
                    continue

                if s1 == t2 and t1 ==s2:
                    if sr1 == sr2:
                        if (r1 == "has_pest" and r2 == "has_host") or (r1 == "has_host" and r2 == "has_pest"):
                            matched_any = True
                            full_matches.append((s1, t1, r1, sr1))
                            #print("\nINVERSION SORTED!")
                            
                            full_match = True
                        
                        if not full_match:
                            #print("\n")
                            #print(f"EPPO : {s1}, {t1} {r1} {sr1}")
                            #print(f"Qwen : {s2}, {t2} {r2} {sr2} ")
                            subrel_match_only = True

                    else:
                        if (r1 == "has_pest" and r2 == "has_host") or (r1 == "has_host" and r2 == "has_pest"):
                            rel_match_only = True

            if not matched_any:
                no_matches.append((s1, t1, r1, sr1))
                #print("Not any match!")

    dic_results = {
        "full_matches": full_matches,
        "reltype_only_matches": reltype_only_matches,
        "subrel_only_matches": subrel_only_matches,
        "no_matches": no_matches,
    }

    return dic_results, full_match, subrel_match_only, rel_match_only

def check_if_entities_in_abstract(abstract, eppo_relations, eppo_index, code_names_by_id):
    """
    Check if documents that don't have a match have a relation from EPPO in the absract 
    Reasons why it could be missing :
    - the abstract is not indicative enough
    - the EPPO relation was not righlty attributed
    """
    at_least_one_relation_found = False
    for (s1, t1, r1, sr1) in eppo_relations:
        
        norm_s1, norm_t1, norm_abstract = clean_text(s1), clean_text(t1), clean_text(abstract)
        
        if norm_s1 in norm_abstract and norm_t1 in norm_abstract:
            at_least_one_relation_found= True
        else:
            s1_code = get_EPPO_code(s1, eppo_index)
            t1_code = get_EPPO_code(s1, eppo_index)

            all_names_s1 = code_names_by_id[s1_code].get("names")
            all_names_t1 = code_names_by_id[t1_code].get("names")

            for term in all_names_s1:
                norm_term = clean_text(term)
                if norm_term in norm_abstract:
                    for target_term in all_names_t1:
                        norm_target_term = clean_text(target_term)
                        if norm_target_term in norm_abstract:
                            print("\n")
                            print("*"*50)
                            print(f"\nSearching for {s1}, {t1}")
                            print("\nAbstract :")
                            print(abstract)
                            print("FOUND !")
                            print("IDENTIFIED NORM TERMS : ", norm_target_term, norm_term)
                            at_least_one_relation_found= True

            
            #print("NOT FOUND")
    return at_least_one_relation_found

def main():
    config = load_config(Path("config/inputfile_paths.json"))
    enriched_interactions = load_json(config["gold_file"])
    code_names = load_json(config["code_names_file"])
    extracted = load_json(config["extracted_findings_file"])
    ref_cache = load_json(config["cache_data"])
    unique_ref_list = load_list(config["unique_ref"])

    INDEX_DIR = Path("repository_index")
    INDEX_DIR.mkdir(exist_ok=True)

    dic_index_extractfinding_by_doi, code_map, eppo_index, graph_data = build_indices(config, extracted, code_names, INDEX_DIR, ref_cache, unique_ref_list)
    code_names_by_id = {item["id"]: item for item in code_names}
    print("Doi index build ! ")

    dic_EPPO_findings = {}
    findings_found_in_prediction = {}

    results_by_doi = {}
    number_error_format = 0
    total_empty_finding = 0
    
    total_findings_extracted = 0
    relation_with_missing_code = 0
    total_full_matches = 0
    total_reltype_only_matches =0
    total_subrel_only_matches = 0
    total_no_match = 0
    total_at_least_one_relation_found = 0
    set_no_present_relation_in_abstract = set()
    
    
    for doi in graph_data["set_ref_in_query_art"]:
        #print("\n============================================")
        #print("Doi : ", doi)

        eppo_relations = build_eppo_relations(doi, enriched_interactions, code_map)
        #print(f"\n\nNumber of EPPO relations for this doi {doi} :")
        
        parsed, number_error_format, empty_finding, abstract = get_qwen_finding_extraction(doi, dic_index_extractfinding_by_doi, eppo_index, number_error_format)
        total_empty_finding += empty_finding
        qwen_relations, total_findings_extracted, relation_with_missing_code = build_qwen_relations(parsed, eppo_index, code_map, total_findings_extracted, relation_with_missing_code)

        dic_results, full_match, subrel_match_only, rel_match_only = compare_relations(eppo_relations, qwen_relations)
        
        
        if full_match:
            total_full_matches += 1
        else:
            if rel_match_only:
                total_reltype_only_matches += 1
            elif subrel_match_only:
                total_subrel_only_matches +=1
            else:
                total_no_match +=1
                at_least_one_relation_found = check_if_entities_in_abstract(abstract, eppo_relations, eppo_index, code_names_by_id)
                if at_least_one_relation_found:
                    total_at_least_one_relation_found += 1
                else:
                    set_no_present_relation_in_abstract.add(doi)

    print("\n=== TOTALS ===")
    print("Full matches:", total_full_matches)
    print("Reltype-only matches:", total_reltype_only_matches)
    print("Subrel-only matches:", total_subrel_only_matches)
    print("No match : ", total_no_match)
    print("Total common doi query : ", len(graph_data["set_ref_in_query_art"]))
    print("Number of relation found in abstract : ",total_at_least_one_relation_found )
    print("\nDois for which no wos relation was found : ")
    for e in set_no_present_relation_in_abstract:
        print(f"- {e}")
    print(f"\nTotal findings extracted with at least one missing code in the relation : {relation_with_missing_code} out of {total_findings_extracted}")
    print(f"Proportion finding extracted with at least one missing code : {relation_with_missing_code/total_findings_extracted*100}")
    print(f"Several explanations : \nAlignement failed ; \nEntity does not exist in EPPO ; \nToo general term etc : insect, insect  families ; \nTwo entities extracted instead of one")

if __name__ == "__main__":
    main()

