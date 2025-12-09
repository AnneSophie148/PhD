
import json
from tqdm import tqdm
import glob
from unidecode import unidecode
import string
import os
import pandas as pd
import requests
import urllib.parse
import time
from utils import normalize_doi
import re

def find_doi_crossref(full_ref, expected_doi):
    '''Check if a reference matches corresponds to the cited_doi we are searching. Query crossref to return possible dois and compare them to the cited doi''' 

    if not isinstance(full_ref, str):
        return None
        
    query = urllib.parse.quote(full_ref)
    url = f"https://api.crossref.org/works?query.bibliographic={query}&select=DOI"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        items = data.get("message", {}).get("items", [])
        if not items:
            return False
        for item in items:
            doi = item.get("DOI")
            if doi and doi.lower() == expected_doi.lower():
                return True
        time.sleep(0.3)
        return False
    except requests.exceptions.RequestException as e:
        print(f"CrossRef request failed: {e}")
        return False

def clean_text(text):
    '''Clean text for normalisation'''
    text = unidecode(str(text).lower())
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace("’", "").replace("–", "").replace("—", "").replace("“", "").replace("”", "")
    text = text.replace("-", "").replace("/", "").replace("_", "").replace("  ", "")
    return text.strip()

def compare_articles(ref_title, title_B, ref_authors, first_author_B):
    '''Compare the reference to the article we are searching by comparing titles and authors'''
    ref_title_norm = clean_text(unidecode(str(ref_title).lower()).strip())
    title_B_norm = clean_text(unidecode(str(title_B).lower()).strip())
    if isinstance(ref_authors, list) and len(ref_authors) > 0:
        ref_first_author = unidecode(ref_authors[0].lower()).strip()
    else:
        ref_first_author = unidecode(str(ref_authors).lower()).strip()
    first_author_B_norm = unidecode(str(first_author_B).lower()).strip()
    
    title_match = ref_title_norm == title_B_norm
    author_match = ref_first_author == first_author_B_norm

    if title_match and author_match:
        return True
    return False

def build_tei_lookup(tei_root="../CORPUS_ENRICHED/citing_articles"):
    '''return a dic in which for each doi the file path for pds that were processed by grobid'''
    tei_files = glob.glob(f"{tei_root}/*/*.grobid.tei.xml")
    tei_lookup = {}
    for i, fn in enumerate(tei_files, 1):
        base = os.path.basename(fn)
        doi_part = base.replace(".grobid.tei.xml", "")
        tei_lookup[doi_part] = fn
    return tei_lookup

def preload_citation_data(tei_lookup, citations_root="../citing_articles_citations"):
    '''Load csv file for a tei path'''
    citation_data = {}
    loaded_sources = set()
    loaded_csvs = []

    for i, (safe_doi, tei_path) in enumerate(tei_lookup.items(), 1):
        source = os.path.basename(os.path.dirname(tei_path))
        if source in loaded_sources:
            continue

        csv_path = os.path.join(citations_root, f"references_citing_{source}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            citation_data[source] = df
            loaded_sources.add(source)
            loaded_csvs.append(csv_path)

    return citation_data

def load_graph(graph_path):
    '''Load the citation graph'''
    with open(graph_path, "r", encoding="utf-8") as f:
        return json.load(f)

def reconstruct_graph(graph, graph2, output_path):
    '''Keeps article and finding metadata from graph output by build_graph, takes the corresponding relations from graph2
    Was necessary because i had started extracting dois from crossref already that were saved in graph2'''
    edges = graph2.get("edges", [])
    new_edges = []

    
    for i, edge in enumerate(tqdm(edges, desc="Counting citation_passages")):
        citing_doi = normalize_doi(edge["citing_doi"])
        cited_doi  = normalize_doi(edge["cited_doi"])
        print(citing_doi)
        new_edges.append(edge)

        #If citation passages were found directly in graph
        if edge.get("citation_passages"):
            continue


    new_graph = {"articles": graph.get("articles", []), "edges": new_edges}
    with open(output_path, "w") as f:
        json.dump(new_graph, f, indent=2)

    print(f"Saved updated graph to {output_path}")

def safe_read_csv(path):
    try:
        if os.path.getsize(path) == 0:
            print(f"Empty CSV file: {path}")
            return None

        return pd.read_csv(path, engine="python", sep=",", quotechar='"', on_bad_lines="warn")

    except pd.errors.EmptyDataError:
        print(f"EmptyDataError while reading: {path}")
        return None

    except Exception as e:
        print(f"Unexpected error reading CSV {path}: {e}")
        return None

def search_citation_passage(total_citations_to_find, cited_doi, citing_doi, article_meta, safe_citing, doi_to_path, citation_data, edge, dois_csv, doc_path):
    '''Search the corresponding citation passage for a pair citing doi and cited doi. Searches in the corresponding csv citation file if there is a doi matching the cited doi we are looking for.
    Returns the citation passages and the section'''
    total_citations_found = 0

    #if the edge was already processed but the section is missing edge_processed will be true
    edge_processed = False
    language_choice = ["en", "unkown", "pt", "es"]

    sources = {"tei":["ISTEX", "Wiley", "Unpaywall", "Crossref"], "xml":["PMC", "PLOS", "Elsevier"]}
    cited_article = article_meta.get(cited_doi, {})
    citing_article = article_meta.get(citing_doi, {})
    lg = citing_article.get("lg")
    if lg is None:
        #check latter why language not available
        print(f"No 'lg' field for citing article {citing_doi} → skipping")
        return "", ""


    authors_list = [authors_list] if isinstance(authors_list := cited_article.get("author", []), str) else authors_list
    ref_authors_raw = ", ".join([unidecode(a) for a in authors_list])
    ref_title = cited_article.get("title", "")
    referenced_year_int = int(cited_article["year"]) if str(cited_article.get("year", "")).isdigit() else None

    source = os.path.basename(os.path.dirname(doc_path))

    if source in sources["tei"]:
        df_citing_all = citation_data.get(source)
    else:
        df_citing_all = safe_read_csv(doc_path)

    if df_citing_all is None:
        return "", ""

    if lg not in language_choice:
        #only accepts en, pt, es and unknown for now
        return "", ""

    doi_col = None
    for col in ["DOI_DOC", "DOC_DOI"]:
        if col in df_citing_all.columns:
            doi_col = col
            break

    df_citing = df_citing_all[df_citing_all[doi_col] == citing_doi]
    df_both = df_citing[df_citing["REF_DOI"] == cited_doi]
    passages = []
    sections = []
    context_cols = ["S-3", "S-2", "S-1", "S-0", "CITATION", "S+0", "S+1", "S+2", "S+3"]

    for _, row in df_both.iterrows():
        section = row["SECTION"]
        sections.append(section)
        if edge.get("citation_passages"):
            passage = edge["citation_passages"][0]["Decomposed_passage"]
        else:
            passage= {col: str(row[col]) if col in row and pd.notna(row[col]) else "" for col in context_cols}
            
        passages.append(passage)
        return passages, sections

    if df_both.empty:   
        clean_ref_authors = clean_text(ref_authors_raw).strip()
        first_letter_ref_author = clean_ref_authors[0].lower() if clean_ref_authors else None
        surname_only = len(clean_ref_authors.split()) < 2

        family_name = None
        if ";" in ref_authors_raw:
            first_author_str = ref_authors_raw.split(";")[0].strip()
        else:
            first_author_str = ref_authors_raw.strip()

        first_author_parts = first_author_str.split(" ")

        if len(first_author_parts) == 2:
            #pattern: Firstname Lastname
            family_name = clean_text(first_author_parts[1])
            first_letter_ref_author = family_name[0].lower()

        elif len(first_author_parts) == 3:
            #pattern: Firstname Initial. Lastname 
            if re.fullmatch(r"[A-Z]\.", first_author_parts[1]):
                if not re.fullmatch(r"[A-Z]\.", first_author_parts[2]):
                    family_name = clean_text(first_author_parts[2])
                
        crossref_found = False
        full_ref_found_crossref=""

        #If the cited doi is not in the references, we need to check all the references
        for _, row in df_citing.iterrows():
            
            passage_exists = False
            section = row["SECTION"]

            if edge.get("citation_passages"):
                #to add section : here case where i have already found the citation passage
                citation_passage_ref = {col: str(row[col]) if col in row and pd.notna(row[col]) else "" for col in context_cols}
                full_text = " ".join([
                    "(CITSEG)" if col == "CITATION" and citation_passage_ref[col] else citation_passage_ref[col]
                    for col in ["S-3","S-2","S-1","S-0","CITATION","S+0","S+1","S+2","S+3"]
                    if col in citation_passage_ref and citation_passage_ref[col]
                ]).strip()
                #comparison of the passage to get section
                for structured in edge["citation_passages"]:
                    if citation_passage_ref == structured["Decomposed_passage"]:
                        sections.append(section)
                        passages.append(structured["Decomposed_passage"])
                        passage_exists = True
                        edge_processed = True
                        break   
                if passage_exists:
                    continue



            if edge_processed is False:
                match = False
                year_ref = str(row.get("REF_YEAR", ""))[:4]
                
                first_author_col = None
                for col in ["FIRST_AUTHOR_SURNAME", "AUTHOR1_SURNAME"]:
                    if col in df_citing_all.columns:
                        first_author_col = col
                        break
                
                full_ref = row.get("FULL_REF", "")
                first_author_citing = clean_text(row.get(first_author_col, "") or "").lower()
                potential_title = row.get("TITLE_REF", "") or ""

                #if we have already found the correct reference with crossref we only need to take the passage where the full_ref is the same as the one found with crossref
                if crossref_found is True:
                    if full_ref != full_ref_found_crossref:
                        continue
                    else:
                        #we can directly say it's a match !
                        match = True

                #first letter of citing author
                first_letter_author_citing = first_author_citing[0].lower() if first_author_citing else None

                #if there is a family name but the first letter does not match the one of the family name of the first author (ref article) --> mismatch
                if surname_only and first_letter_author_citing and first_letter_ref_author:
                    if first_letter_author_citing != first_letter_ref_author:
                        if pd.notna(first_author_citing) and first_author_citing.lower() != "nan":
                            continue

                if str(full_ref).lower() == "nan" and str(potential_title).lower() == "nan":
                    continue

                #if there is a family name detected check if the first letter matches
                if family_name:
                    if not first_author_citing.startswith(family_name.lower()):
                        continue
                
                if not match:
                    if family_name:
                        match = compare_articles(ref_title, potential_title, ref_authors_raw, family_name)
                    else:
                        match = compare_articles(ref_title, potential_title, ref_authors_raw, first_author_str)
                    
                match_crossref = False
                

                if not match:
                    #if the full_ref is empty then we cannot do a crossref search
                    if str(full_ref).lower().replace(" ", "").strip() == "nan":
                        continue

                    match_crossref = find_doi_crossref(full_ref, cited_doi)

                if match or match_crossref:
                    total_citations_found += 1
                    passages.append({col: str(row[col]) if col in row and pd.notna(row[col]) else "" for col in context_cols})
                    sections.append(section)
                    full_ref_found_crossref = full_ref
                    crossref_found = True
                
            else:
                print("Edge already processed, passing directly to the next row")

        return passages, sections


def complete_graph3(graph3):
    """Complete the graph with citation passages including section."""

    csv_files = glob.glob("../../../AlvisNLP/anneso/outputAlvisNLP/citing_articles/*/*.csv")
    dois_csv = {f.split("/")[-1].replace("_references_citing.csv", ""): f for f in csv_files}
    doi_to_path = build_tei_lookup()
    citation_data = preload_citation_data(doi_to_path)
    doi_to_path.update({f.split("/")[-1].replace("_references_citing.csv", ""): f for f in csv_files if f.split("/")[-1].replace("_references_citing.csv", "") not in doi_to_path})
    article_meta = {normalize_doi(a["doi"]): a for a in graph3.get("articles", [])}

    total_citations_to_find = 0
    total_citations_found = 0

    edges = graph3.get("edges", [])
    n_edges = len(edges)
    save_every = max(1, n_edges // 100)   #save every 1%
    processed_count = 0

    for idx, edge in enumerate(tqdm(edges, desc="Completing edges")):
        if ("citation_passages" in edge and any("section" in p for p in edge["citation_passages"])):
            processed_count += 1
            continue
        elif "citation_passages" in edge and edge["citation_passages"] and edge["citation_passages"][0]["Full-text"] == "NOT_FOUND":
            processed_count += 1
            continue


        citing_doi = normalize_doi(edge["citing_doi"])
        cited_doi = normalize_doi(edge["cited_doi"])
        doc_path = doi_to_path.get(citing_doi.replace("/", "_"))

        if not doc_path:
            continue

        total_citations_to_find += 1
        passages, sections = search_citation_passage(
            total_citations_to_find=total_citations_found,
            cited_doi=cited_doi,
            citing_doi=citing_doi,
            article_meta=article_meta,
            safe_citing=citing_doi.replace("/", "_"),
            doi_to_path=doi_to_path,       
            citation_data=citation_data,
            edge=edge,
            dois_csv=dois_csv,
            doc_path=doc_path)

        if len(sections) != len(passages):
            print(citing_doi, cited_doi)

            
        structured_passages = [{"section": sections[i], "Full-text": " ".join("(CITSEG)" if col == "CITATION" and p.get(col) else p.get(col, "") for col in ["S-3","S-2","S-1","S-0","CITATION","S+0","S+1","S+2","S+3"] if p.get(col)).strip(), "Decomposed_passage": p } for i, p in enumerate(passages)]

        if not passages or not sections:
            # If search_citation_passage found nothing but edge was already processed
            #write not found to not reprocess it
            empty_decomposed = {col: "" for col in ["S-3","S-2","S-1","S-0","CITATION","S+0","S+1","S+2","S+3"]}
            
            passages = [empty_decomposed]
            sections = ["NOT_FOUND"]
            structured_passages = [{"section": "NOT_FOUND", "Full-text": "NOT_FOUND", "Decomposed_passage": empty_decomposed}]
            
        edge["citation_passages"] = structured_passages
        processed_count += 1

        if processed_count % save_every == 0:
            print(f"Checkpoint: saving graph at {processed_count}/{n_edges} edges")
            with open(output_path, "w") as f:
                json.dump(graph3, f, indent=2)


    new_graph = { "articles": graph3.get("articles", []), "edges": edges}

    #final save
    with open(output_path, "w") as f:
        json.dump(new_graph, f, indent=2)
    print(f"Saved updated graph to {output_path}")
    



if __name__ == "__main__":
    graph_path = "graph_query_ref_citing.json"
    existing_enriched_graph = "graph_query_ref_citing_with_citations.json"
    output_path = "graph_with_citations_V2.json"

    graph = load_graph(graph_path)

    if os.path.exists(existing_enriched_graph):
        graph2 = load_graph(existing_enriched_graph)
        print(f"Number of edges in graph2: {len(graph2.get('edges', []))}")
        #reconstruct_graph(graph, graph2, output_path)

    graph3 = load_graph(output_path)
    print(f"Number of edges in graph3: {len(graph3.get('edges', []))}")

    edges = graph3.get('edges', [])    
    complete_graph3(graph3)