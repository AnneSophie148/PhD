
import json
import sys
sys.path.append("..")
from utils import normalize_doi, get_downloaded_dois
import urllib.parse
import requests
import time
import re
import unicodedata
from unidecode import unidecode
import string
from tqdm import tqdm
import os
import copy
import pandas as pd

def load_graph(graph_path):
    with open(graph_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_dois_crossref(full_ref, max_results=5):
    """
    Query Crossref with a bibliographic reference string
    and return a list of matching DOIs.
    """

    if not isinstance(full_ref, str) or not full_ref.strip():
        return []

    query = urllib.parse.quote(full_ref)
    url = ("https://api.crossref.org/works"
        f"?query.bibliographic={query}&rows={max_results}&select=DOI")

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        items = data.get("message", {}).get("items", [])
        dois = [item["DOI"] for item in items if "DOI" in item]

        time.sleep(0.3)
        return dois

    except requests.exceptions.RequestException as e:
        print(f"Crossref request failed: {e}")
        return []


def normalize_name(name):
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.lower()
    return re.sub(r"[^a-z]", "", name)

def extract_first_author_vector(author_field):
    """
    Extract first author's surname from VECTOR_DATASET author field.
    Supports:
      - string: "A. Smith; B. Jones"
      - list: ["A. Smith", "B. Jones"]
    """
    if not author_field:
        return None

    #Case 1: list of authors
    if isinstance(author_field, list):
        first_author = author_field[0]

    #Case 2: string with separators
    elif isinstance(author_field, str):
        first_author = author_field.split(";")[0].strip()

    else:
        return None

    #Extract surname : last token
    parts = first_author.split()
    if not parts:
        return None

    surname = parts[-1]
    return normalize_name(surname)


def extract_first_author_eppo(ref):
    """
    Extracts first author's surname from EPPO reference string
    """
    #Everything before first comma
    m = re.match(r"\s*([^,]+),", ref)
    if not m:
        return None
    surname = m.group(1)
    return normalize_name(surname)

def strict_first_author_match(ref, vector_article):
    eppo_author = extract_first_author_eppo(ref)
    vector_author = extract_first_author_vector(vector_article.get("author"))

    if not eppo_author or not vector_author:
        if find_title_match(ref, vector_article.get("title")):
            return True
        else:
            return False
    
    if vector_author in eppo_author or eppo_author in vector_author:
        return True
    else:
        
        return False

def first_author_match(ref, vector_article):
    """Function that searches if authors and title mathces
    Returns a boolean"""
    eppo_author = extract_first_author_eppo(ref)
    vector_author = extract_first_author_vector(vector_article.get("author"))

    if not eppo_author or not vector_author:
        if find_title_match(ref, vector_article.get("title")):
            return True
        else:
            return False
    
    if vector_author in eppo_author or eppo_author in vector_author:
        return True
    else:
        
        
        if find_title_match(ref, vector_article.get("title")):
            return True
        else:
            return False
        


def find_title_match(ref, vector_title):
    """Check if there is a title in the vector dataset article and if this title is in the reference"""
    if vector_title and unidecode(vector_title.lower()) in unidecode(ref.lower()):
        return True

def clean_text(text):
    '''Supprime la ponctuation, guillemets, tirets, slashes, et retourne le texte nettoyé.'''
    text = unidecode(text.lower())
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace("’", "").replace("–", "").replace("—", "").replace("“", "").replace("”", "")
    text = text.replace("-", "").replace("/", "").replace("_", "").replace("  ", "")
    return text.strip()


def extract_title_EPPO(ref):
    """
    Extract the title from an EPPO reference.

    Example:
    Šubíková V, Kollerová E & Slováková L (2002)
    Occurrence of nepoviruses in small fruits and fruit trees in Slovakia.
    Plant Protection Science 38, 367-369.

    -> "Occurrence of nepoviruses in small fruits and fruit trees in Slovakia"
    """

    patterns = [
        #Author (2002) Title.
        r"\((19|20)\d{2}\)\s*(.+?)\.",

        #Author. 2002. Title.
        r"(19|20)\d{2}\.\s*(.+?)(?:\.|$)"]

    for pattern in patterns:
        match = re.search(pattern, ref)

        if match:
            return match.group(2).strip()

    return None

def detect_duplicates_EPPO(all_refs, dic_mapping_unique_references):
    """ Detect duplicates based on (first_author, year, title) and keep only one ref"""

    saved_authors_dic = {}
    duplicate_refs = set()

    for ref in sorted(all_refs):
        first_author = extract_first_author_eppo(ref)
        year_match = re.search(r"\b(19|20)\d{2}\b", ref)
        year_eppo = int(year_match.group(0)) if year_match else None
        title_EPPO = extract_title_EPPO(ref)
        if title_EPPO:
            title_EPPO = clean_text(title_EPPO)

        key_dup = (first_author, year_eppo, title_EPPO)

        if key_dup not in saved_authors_dic:
            saved_authors_dic[key_dup] = ref

        else:
            duplicate_refs.add(ref)

            #remove duplicate from dic_mapping_unique_references
            for doi_key, dic in dic_mapping_unique_references.items():
                to_delete = []
                for refnorm, variants in dic.items():
                    if ref in variants:
                        variants.remove(ref)

                    #remove empty entries
                    if len(variants) == 0:
                        to_delete.append(refnorm)

                for refnorm in to_delete:
                    del dic[refnorm]
    return duplicate_refs, dic_mapping_unique_references

def find_dois_crossref(full_ref, max_results=5):
    """Query crossref to get a doi, tries to match it to the full reference from EPPO"""
    time.sleep(0.3)
    title_EPPO = extract_title_EPPO(full_ref)

    if title_EPPO:
        norm_title_EPPO = clean_text(title_EPPO)

        if not isinstance(full_ref, str) or not full_ref.strip():
            return []

        query = urllib.parse.quote(full_ref)

        url = (
            "https://api.crossref.org/works"
            f"?query.bibliographic={query}&rows={max_results}"
        )

        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            items = data.get("message", {}).get("items", [])

            results = []

            for item in items:
                doi = item.get("DOI")
                title = item.get("title", [""])
                title = title[0] if title else ""
                norm_title_crossref = clean_text(title)

                authors = []
                for a in item.get("author", []):
                    given = a.get("given", "")
                    family = a.get("family", "")
                    authors.append(f"{given} {family}".strip())

                journal = item.get("container-title", [""])
                journal = journal[0] if journal else ""

                year = None
                if "published-print" in item:
                    year = item["published-print"]["date-parts"][0][0]
                elif "published-online" in item:
                    year = item["published-online"]["date-parts"][0][0]

                score = item.get("score")

                metadata = {
                    "doi": doi,
                    "title": title,
                    "authors": authors,
                    "journal": journal,
                    "year": year,
                    "score": score,
                    "full_item": item
                }

                results.append(metadata)


                if norm_title_EPPO in norm_title_crossref:
                    return metadata
                
                if len(norm_title_crossref.split()) > 5 and norm_title_crossref in norm_title_EPPO:
                    print("\n")
                    print("*"*200)
                    print(f"Search for variant {full_ref}")
                    print("\n========== MATCH ==========")
                    print("DOI:", doi)
                    print("Title:", title)
                    print("Authors:", authors)
                    print("Journal:", journal)
                    print("Year:", year)
                    print("Score:", score)
                    print("FOUND !! ")
                    return metadata
                
                if len(norm_title_crossref.split()) > 5 and norm_title_crossref in clean_text(full_ref):
                    print("\nPOTENTIAL MATCH : ")
                    print("EPPO REF : ", full_ref)
                    print("Crossref result : ")
                    print("Title:", title)
                    print("Authors:", authors)


        
        except requests.exceptions.RequestException as e:
            print(f"Crossref request failed: {e}")
            return None

        return None

def build_unique_references(refs, identified_references_with_crossref):
    """Build a mapping normalizing the references to deduplicate
    References are organized as bellow :
    - DOI : {normalized_name = [list of variants (original reference in EPPO) ]}
    - No_doi_detected : {normalized_name = [list of variants]}

    """
    dic_mapping_unique_references = {}
    all_refs = set()
    ref_doi_set = set()
    to_remove = set()
    cach_processed_variant = {}
    new_crossref_count = 0

    for item in tqdm(refs, desc="Processing biological interactions"):
        for r in item.get("bibref", {}).get("refs", []):
            doi_match = re.search(r"10\.\d{4,9}/\S+", r)
            ref_doi = normalize_doi(doi_match.group(0)) if doi_match else None
            r_normalized = clean_text(r)

            if not doi_match:
                if r_normalized in cach_processed_variant:
                    ref_doi = cach_processed_variant.get(r_normalized)
                elif r_normalized in identified_references_with_crossref:
                    ref_doi = identified_references_with_crossref[r_normalized].get("doi")
                else:
                    crossref_result = find_dois_crossref(r)
                    if crossref_result:
                        ref_doi = crossref_result.get("doi")
                        cach_processed_variant[r_normalized]=ref_doi
                        identified_references_with_crossref[r_normalized] = crossref_result
                        new_crossref_count += 1
                        if new_crossref_count > 0 and new_crossref_count % 10 == 0:
                            with open(other_dois_crossref, "w", encoding="utf-8") as f:
                                json.dump(identified_references_with_crossref, f, ensure_ascii=False, indent=2)

            #define key
            key = ref_doi if ref_doi else "No_doi_detected"

            #under key is the normalized reference
            if key not in dic_mapping_unique_references:
                dic_mapping_unique_references[key] = {}
                
            if r_normalized not in dic_mapping_unique_references[key]:
                dic_mapping_unique_references[key][r_normalized]= []

            #add original reference as variant (value) of the normalized reference
            if r not in dic_mapping_unique_references[key][r_normalized]:
                dic_mapping_unique_references[key][r_normalized].append(r)

            #save the processed doi in a set
            if ref_doi:
                ref_doi_set.add(ref_doi)
    

    for refnorm, var_without_doi in dic_mapping_unique_references["No_doi_detected"].items():
        for key, dic in dic_mapping_unique_references.items():
            if key == "No_doi_detected":
                continue

            for refdoi, vardoi in dic.items():
                if refnorm in refdoi and len(refnorm.split()) > 5:
                    #add variants
                    for variant in var_without_doi:
                        if variant not in dic_mapping_unique_references[key][refdoi]:
                            dic_mapping_unique_references[key][refdoi].append(variant)

                    to_remove.add(refnorm)
                    break

    for refnorm in to_remove:
        dic_mapping_unique_references["No_doi_detected"].pop(refnorm, None)

    #save unique ref in all_ref dictionary --> a set of unique existing references
    #there are several variants so we only add the variant to all_refs if the normalized name was not seen yet
    for doi_key, dic in dic_mapping_unique_references.items():
        for refnorm, variants in dic.items():
            all_refs.add(variants[0])
            if doi_key != "No_doi_detected":
                if len(dic_mapping_unique_references[doi_key])>1:
                    #only one variant per doi
                    break
    

    all_refs = sorted(all_refs)
    duplicate_refs, dic_mapping_unique_references = detect_duplicates_EPPO(all_refs, dic_mapping_unique_references)

    # remove duplicates from all_refs
    all_refs = [r for r in all_refs if r not in duplicate_refs]

    print("Number of unique references:", len(all_refs))

    with open("unique_ref_EPPO.lst", "w") as f_out:
        for ref in tqdm(all_refs, desc="Writing refs"):
            f_out.write(ref + "\n")
    
    with open(other_dois_crossref, "w", encoding="utf-8") as f:
        json.dump(identified_references_with_crossref, f, ensure_ascii=False, indent=2)

    return all_refs, ref_doi_set, dic_mapping_unique_references

#still some duplicates left in unique_ref_EPPO.lst
#example : de Jager M, Roets F (2022) Pathogenicity of Fusarium euwallaceae towards apple (Malus domestica) and grapevine (Vitis vinifera). Australasian Plant Disease Notes, 17(1), 1-4.
#de Jager MM, Roets F (2022) Pathogenicity of Fusarium euwallaceae towards apple (Malus domestica) and grapevine (Vitis vinifera). Australasian Plant Disease Notes, 17(1), 1-4.

def match_doi(ref_doi, findings_by_doi, articles_by_doi):
    """
    DOI lookup.
    Priority:
    1. findings
    2. citing articles
    """

    if not ref_doi:
        return None, None

    if ref_doi in findings_by_doi:
        return "Query_WOS", findings_by_doi[ref_doi]

    if ref_doi in articles_by_doi:
        return "Citing_articles", articles_by_doi[ref_doi]

    return None, None


def search_crossref_dois(ref, potential_dois, findings_by_doi, articles_by_doi, year_eppo, title_ref_eppo):
    """
    Search if one of the five dois returned by crossref is in the Vector Dataset Graph as Wos_Query or Citing article query.
    Priority:
    1. findings
    2. citing articles
    Conditions to be considered as the same article (ref vs doi returned by crossref that would be in the VD graph):
    - same first author
    - year match or one year difference accepted (in case the article was available before publication date)
    - title match
    """

    if not title_ref_eppo:
        return None, None

    norm_title_ref_eppo = clean_text(title_ref_eppo)

    if not norm_title_ref_eppo:
        return None, None

    first_word_ref = norm_title_ref_eppo.split()[0]

    for doi in potential_dois:
        norm_doi = normalize_doi(doi)
        #search in findings first
        if norm_doi in findings_by_doi:
            f = findings_by_doi[norm_doi]
            title_f = clean_text(f.get("title"))

            if not title_f:
                continue

            if title_f.split()[0] != first_word_ref:
                continue

            y = f.get("year")
            if year_eppo is not None and y is not None:
                if abs(int(y) - year_eppo) <= 1:
                    if first_author_match(ref, f):
                        return "Query_WOS", f

        #not found in findings --> search in citing articles
        if norm_doi in articles_by_doi:
            art = articles_by_doi[norm_doi]
            title_art = clean_text(art.get("title"))

            if not title_art:
                continue

            if title_art.split()[0] != first_word_ref:
                continue

            y = art.get("year")
            if year_eppo is not None and y is not None:
                if abs(int(y) - year_eppo) <= 1:
                    if first_author_match(ref, art):
                        return "Citing_articles", art

    return None, None


def search_exact_title(ref, title_ref_eppo, findings_by_title, articles_by_title):
    """
    Exact normalized title match.
    Priority:
    1. findings
    2. articles

    Conditions : Title matches and first author matches
    """

    if not title_ref_eppo:
        return None, None

    cleaned = clean_text(title_ref_eppo)

    #findings
    if cleaned in findings_by_title:
        f = findings_by_title[cleaned]
        if strict_first_author_match(ref, f):
            return "Query_WOS", f

    #articles
    if cleaned in articles_by_title:
        art = articles_by_title[cleaned]
        if strict_first_author_match(ref, art):
            return "Citing_articles", art

    return None, None


def search_full_title(ref, clean_ref, year_eppo, findings_by_title, articles_by_title):
    """
    Full title search.
    Priority:
    1. findings
    2. articles
    """
    # search in findings
    for title_key, f in findings_by_title.items():
        if not title_key or len(title_key.split()) <= 5:
            continue

        y = f.get("year")
        if year_eppo is None or y is None:
            continue
        if abs(int(y) - year_eppo) > 1:
            continue

        if title_key in clean_ref:
            if first_author_match(ref, f):
                return "Query_WOS", f
            
    #search in citing articles
    for title_key, art in articles_by_title.items():
        if not title_key or len(title_key.split()) <= 5:
            continue

        y = art.get("year")
        if year_eppo is None or y is None:
            continue

        if abs(int(y) - year_eppo) > 1:
            continue

        if title_key in clean_ref:
            if first_author_match(ref, art):
                return "Citing_articles", art

    return None, None


def resolve_ref(ref, norm_ref, articles_by_doi, findings_by_doi, findings_by_title, articles_by_title, ref_cache, ref_doi, crossref_cache):
    """Check if reference is in the Vector Dataset Graph - if yes in Wos_QUery as priority, otherwise in Citing articles"""

    clean_ref = clean_text(ref)
    year_match = re.search(r"\b(19|20)\d{2}\b", ref)
    year_eppo = int(year_match.group(0)) if year_match else None
    title_ref_eppo = extract_title_EPPO(ref)

    #1.Doi match
    art_type, metadata = match_doi(ref_doi, findings_by_doi, articles_by_doi)

    if metadata:
        ref_cache["Found_in_VectorDataset_Graph"][art_type][norm_ref] = metadata
        return ref_cache

    #2. CROSSREF search
    if ref in crossref_cache:
        #test if that happens
        potential_dois = crossref_cache[ref]

    else:
        potential_dois = find_dois_crossref(ref, max_results=5)
        crossref_cache[ref] = potential_dois

    if potential_dois:
        art_type, metadata = search_crossref_dois(ref, potential_dois, findings_by_doi, articles_by_doi, year_eppo, title_ref_eppo)

    if metadata:
        ref_cache["Found_in_VectorDataset_Graph"][art_type][norm_ref] = metadata
        print(f"Updating graph : {art_type} - {metadata}")
        return ref_cache

    #3. exact title match
    art_type, metadata = search_exact_title(ref, title_ref_eppo, findings_by_title, articles_by_title)

    if metadata:
        ref_cache["Found_in_VectorDataset_Graph"][art_type][norm_ref] = metadata
        #print(f"Updating graph : {art_type} - {metadata}")
        return ref_cache

    #4. search each title from graph
    art_type, metadata = search_full_title(ref, clean_ref, year_eppo, findings_by_title, articles_by_title)

    if metadata:
        ref_cache["Found_in_VectorDataset_Graph"][art_type][norm_ref] = metadata
        #print(f"Updating graph : {art_type} - {metadata}")
        return ref_cache

    #5. NOT FOUND
    ref_cache["Not_in_VectorDataset_Graph"].append(norm_ref)

    return ref_cache

def build_reference_cache(all_refs, graph, dic_mapping_unique_references, identified_references_with_crossref, variant_to_info , output_path="ref_in_graph.json", save_every=50):
    """Searches for each deduplicated ref if it's in the Vector Dataset Graph
    Creates a save in a dictionary saved as ref_in_graph.json
    """

    articles = graph.get("articles", [])
    findings = graph.get("findings", [])

    articles_by_doi = {normalize_doi(a["doi"]): a for a in articles if a.get("doi")}
    findings_by_doi = {normalize_doi(f["doi"]): f for f in findings if f.get("doi")}

    articles_by_title = {clean_text(a["title"]): a for a in articles if a.get("title")}
    findings_by_title = {clean_text(f["title"]): f for f in findings if f.get("title")}

    if os.path.exists("ref_in_graph.json"):
        print("Loading existing cache...")

        with open("ref_in_graph.json", "r", encoding="utf-8") as f:
            ref_cache = json.load(f)

    else:
        ref_cache = {
            "Found_in_VectorDataset_Graph": {
                "Query_WOS": {},
                "Citing_articles": {}
            },
            "Not_in_VectorDataset_Graph": []
        }

    crossref_cache = {}

    # rebuild processed refs from saved cache
    set_processed_norm_refs = set()
    set_processed_norm_refs.update(ref_cache["Found_in_VectorDataset_Graph"]["Query_WOS"].keys())
    set_processed_norm_refs.update(ref_cache["Found_in_VectorDataset_Graph"]["Citing_articles"].keys())
    set_processed_norm_refs.update(ref_cache["Not_in_VectorDataset_Graph"])

    

    # loop on each unique reference from the EPPO graph
    for i, ref in enumerate(tqdm(all_refs, desc="Building reference cache"), 1):
        info = variant_to_info.get(ref)
        if not info:
            continue

        norm_ref = info["norm_ref"]
        ref_doi = info["doi"]

        #skip already processed references
        if norm_ref in set_processed_norm_refs:
            #print("Already processed reference")
            continue

        ref_cache = resolve_ref(ref, norm_ref, articles_by_doi, findings_by_doi, findings_by_title, articles_by_title, ref_cache, ref_doi, crossref_cache)

        set_processed_norm_refs.add(norm_ref)

        if i % save_every == 0:
            #print(f"Saving cache at {i}/{len(all_refs)} refs...")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(ref_cache, f, indent=2, ensure_ascii=False)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ref_cache, f, indent=2, ensure_ascii=False)

    print("Cache saved:", output_path)

    return ref_cache


def get_downloaded_status(doi, findings_cache, articles_cache, docs_query, docs_citing, eppo_downloaded_docs):
    """Check if the reference was downloaded"""
    downloaded_status = "Not_downloaded"
    if not doi:
        return downloaded_status

    formatted_doi = normalize_doi(doi)

    if formatted_doi in docs_query or formatted_doi in docs_citing:
        downloaded_status = "Downloaded ; In_Vector_Dataset"
        print(f"\n{formatted_doi} downloaded in vector_dataset")
    else:
        if formatted_doi in eppo_downloaded_docs:
            downloaded_status = "Downloaded ; not_in_Vector_Dataset"
            print(f"\n{formatted_doi} downloaded in vector_dataset")

    return downloaded_status
    
def build_final_dataset(refs, dic_mapping_unique_references, ref_cache, variant_to_info, output_path):
    """Build final dataset : reconstructed biological_interaction enriched with metadata on articles in the vector dataset graph"""
    new_json = []
    references_without_doi = set()

    findings_cache = ref_cache["Found_in_VectorDataset_Graph"]["Query_WOS"]
    articles_cache = ref_cache["Found_in_VectorDataset_Graph"]["Citing_articles"]
    not_found_cache = set(ref_cache["Not_in_VectorDataset_Graph"])

    docs_query = get_downloaded_dois("../../query_articles/*/*")
    docs_citing = get_downloaded_dois("../../citing_articles/*/*")
    eppo_downloaded_docs = get_downloaded_dois("docs_EPPO/*/*")

    for item in tqdm(refs, desc="Building final dataset"):         
        new_item = copy.deepcopy(item)
        bibrefs = new_item.get("bibref", {})
        refs_list = bibrefs.get("refs", [])
        bibrefs["in_graph"] = []
        bibrefs["dois_identified"] = []

        for ref in refs_list:
            found = False
            norm_ref = clean_text(ref)

            info = variant_to_info.get(ref)

            if info:
                norm_ref = info["norm_ref"]
                ref_doi = info["doi"]
            else:
                norm_ref = clean_text(ref)
                ref_doi = None



            if not ref_doi:
                if norm_ref in findings_cache:
                    ref_doi = findings_cache[norm_ref].get("doi")
                elif norm_ref in articles_cache:
                    ref_doi = articles_cache[norm_ref].get("doi")
                else:
                    if norm_ref not in references_without_doi:
                        references_without_doi.add(norm_ref)

            downloaded_status = get_downloaded_status(ref_doi, findings_cache, articles_cache, docs_query, docs_citing, eppo_downloaded_docs)
            bibrefs["dois_identified"].append(ref_doi)

            if norm_ref in findings_cache:
                bibrefs["in_graph"].append({"Query_WOS": findings_cache[norm_ref], "Downloaded_status":downloaded_status})
                continue

            if norm_ref in articles_cache:
                bibrefs["in_graph"].append({"Citing_articles": articles_cache[norm_ref], "Downloaded_status":downloaded_status})
                continue

            if norm_ref in not_found_cache:
                bibrefs["in_graph"].append({"N": "", "Downloaded_status":downloaded_status})
                continue

            for key, group in dic_mapping_unique_references.items():
                by_norm = group.get("by_normalized", {})
                if norm_ref in by_norm:
                    for variant in by_norm[norm_ref]:
                        norm_variant = clean_text(variant)
                        if norm_variant in findings_cache:
                            bibrefs["in_graph"].append({"Query_WOS": findings_cache[norm_variant]})
                            found = True
                            break

                        if norm_variant in articles_cache:
                            bibrefs["in_graph"].append({"Citing_articles": articles_cache[norm_variant]})

                            found = True
                            break

                    if found:
                        break
            if not found:
                bibrefs["in_graph"].append({"N": ""})
                

        new_json.append(new_item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_json, f, indent=2, ensure_ascii=False)

    print("Saved:", output_path)

    print("Number of references without doi : ", (len(references_without_doi)))
    counter = 0
    for e in references_without_doi:
        counter +=1
        if counter >5:
            break
        print(e)

    return new_json

def save_mapping(dic_mapping_unique_references, path="mapping_refs.json"):
    """Save normalized reference mapping"""

    with open(path, "w", encoding="utf-8") as f:
        json.dump(dic_mapping_unique_references, f, indent=2, ensure_ascii=False)

    print("Mapping saved:", path)

if __name__ == "__main__":
    #references after parsing
    eppo_ref = "cleaned_biological_interactions_EPPO.json"

    #graph with version finding = query articles ;
    # articles = citing articles
    graph_path = "../graph_with_classified_citations_V1_BIOBERT.json"
    other_dois_crossref = "Crossref_doi_EPPO_references.json"

    mapping_path = "mapping_refs.json"
    output_ref_graph = "ref_in_graph.json"

    graph = load_graph(graph_path)
    file_crossref = "Crossref_doi_EPPO_references.json"
    identified_references_with_crossref = {}

    if os.path.exists(other_dois_crossref):
        with open(other_dois_crossref, "r", encoding="utf-8") as f:
            identified_references_with_crossref = json.load(f)

    with open(eppo_ref, "r", encoding="utf-8") as f:
        refs = json.load(f)

    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            dic_mapping_unique_references = json.load(f)
        ref_doi_set = {key for key in dic_mapping_unique_references if key != "No_doi_detected"}

        with open("unique_ref_EPPO.lst", "r", encoding="utf-8") as f_out:
            unique_refs = [line.strip() for line in f_out]
    else:
        #deduplicate and create a mapping with unique references
        unique_refs, ref_doi_set, dic_mapping_unique_references = build_unique_references(refs, identified_references_with_crossref)
        save_mapping(dic_mapping_unique_references, path=mapping_path)


    variant_to_info = {}

    for doi, values in dic_mapping_unique_references.items():
        for normref, variants in values.items():
            for variant in variants:
                variant_to_info[variant] = {"norm_ref": normref, "doi": None if doi == "No_doi_detected" else doi}

    ref_cache = build_reference_cache(unique_refs, graph, dic_mapping_unique_references, identified_references_with_crossref, variant_to_info, output_path=output_ref_graph)
    output_path = "common_biologic_interaction_eppo_vectordataset.json"
    build_final_dataset(refs, dic_mapping_unique_references, ref_cache, variant_to_info, output_path)