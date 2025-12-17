
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

def filter_graph(graph):
    edges = graph.get("edges", [])
    articles = graph.get("articles", [])
    year_arts = graph.get("year_arts", [])
    years = graph.get("years", [])
    findings = graph.get("findings")

    new_edges = []
    new_years = set()
    new_findings = []
    language_choice = ["en"]

    new_articles_by_doi = {}
    findings_dois = set()  

    articles_by_doi = {normalize_doi(a.get("doi")): a for a in articles if "doi" in a}

    for edge in tqdm(edges, desc="Counting citation_passages"):
        citing_doi = normalize_doi(edge["citing_doi"])
        cited_doi  = normalize_doi(edge["cited_doi"])

        citing_article = articles_by_doi.get(citing_doi, {})
        cited_article = articles_by_doi.get(cited_doi, {})

        lg = citing_article.get("lg", "unknown")
        
        if lg in language_choice:
            new_edges.append(edge)

            #findings = cited DOI only
            findings_dois.add(cited_doi)
            new_findings.append(cited_article)

            #articles = citing DOI only (new compared to build graph where articles was both citing and cited articles)
            new_articles_by_doi[citing_doi] = citing_article

            if citing_article.get("year"):
                new_years.add(int(citing_article["year"]))
            if cited_article.get("year"):
                new_years.add((int(cited_article["year"])))

    new_articles = list(new_articles_by_doi.values())
    new_years_arts = {int(year): [] for year in sorted(new_years)}

    for article in new_articles + new_findings:
        year = article.get("year")
        try:
            y = int(float(year))
            new_years_arts[y].append(article["doi"])
        except:
            pass

    new_graph = {"years": sorted(new_years),
        "year_arts": new_years_arts,
        "articles": new_articles, 
        "findings": new_findings,
        "edges": new_edges}

    return new_graph

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

def normalize_and_filter_df(df_citing_all, citing_doi, source, sources):
    #check which is the correct column
    doi_col = next((c for c in ["DOI_DOC", "DOC_DOI"] if c in df_citing_all.columns), None)
    if doi_col is None:
        return None

    if source in sources["tei"]:
        df = df_citing_all[df_citing_all[doi_col] == citing_doi]
    else:
        df = df_citing_all

    df = df.copy()
    df["REF_DOI"] = df["REF_DOI"].apply(lambda x: normalize_doi(x) if isinstance(x, str) else x)
    return df

def extract_passages_exact_match(df_citing, cited_doi, context_cols):
    """Seach if the cited doi is mentioned by a reference"""
    df_match = df_citing[df_citing["REF_DOI"] == cited_doi]
    passages, sections = [], []

    for _, row in df_match.iterrows():
        section = row["SECTION"]
        sections.append(section)

        passage = {col: str(row[col]) if col in row and pd.notna(row[col]) else "" for col in context_cols}
        passages.append(passage)

    return passages, sections, df_match

def pmid_from_doi(doi: str):
    """Return the PubMed ID (PMID) corresponding to a DOI, or None if not found."""
    try:
        if not doi:
            return None

        time.sleep(0.3)

        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": doi + "[DOI]", "retmode": "xml", "email": config_APIs.INSTITUTION_EMAIL}

        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None

        root = ET.fromstring(r.text)
        id_list = root.findall(".//IdList/Id")
        if id_list:
            return id_list[0].text  #return the first PMID found

    except Exception as e:
        return None

def search_match(df, edge, context_cols, df_citing_all, first_author_str, ref_title, ref_authors_raw, first_letter_ref_author, surname_only, family_name, cited_doi, citing_doi, first_author_col):
    '''Search for each reference from a citing article if this reference corresponds to the article we are looking for (cited doi). Use crossref as query as last resort'''
    
    if "AUTHOR1_SURNAME" in df.columns:
        df["AUTHOR1_SURNAME_CLEAN"] = df["AUTHOR1_SURNAME"].apply(lambda x: clean_text(x).lower() if isinstance(x, str) else "")

    passages, sections = [], []
    crossref_found = False
    full_ref_found_crossref = ""
    edge_processed = False

    for _, row in df.iterrows():

        passage_exists = False
        section = row["SECTION"]

        if edge.get("citation_passages"):
            if edge["citation_passages"][0]["Full-text"] != "NOT_FOUND":
                citation_passage_ref = {col: str(row[col]) if col in row and pd.notna(row[col]) else "" for col in context_cols}
                full_text = " ".join(["(CITSEG)" if col == "CITATION" and citation_passage_ref[col] else citation_passage_ref[col] for col in ["S-3","S-2","S-1","S-0","CITATION","S+0","S+1","S+2","S+3"] if col in citation_passage_ref and citation_passage_ref[col]]).strip()

                #Compare to each passage in the edge to know if the edge was processed
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
            full_ref = row.get("FULL_REF", "")
            first_author_citing = clean_text(row.get(first_author_col, "") or "").lower()
            potential_title = row.get("TITLE_REF", "") or ""

            if crossref_found is True:
                #if the reference was already found with crossref then no point in searching a match again
                if full_ref != full_ref_found_crossref:
                    continue
                else:
                    print("Found Crossref same matched referenced")
                    match = True

            #First letter of citing author
            first_letter_author_citing = first_author_citing[0].lower() if first_author_citing else None

            #Basic initial mismatch check on the first letter
            if surname_only and first_letter_author_citing and first_letter_ref_author:
                if first_letter_author_citing != first_letter_ref_author:
                    if pd.notna(first_author_citing) and first_author_citing.lower() != "nan":
                        continue

            #check if the first authors name is matching the ref family name
            if family_name:
                if not first_author_citing.startswith(family_name.lower()):
                    continue

            if str(full_ref).lower() == "nan" and str(potential_title).lower() == "nan" and match==False:
                #no title --> skip
                continue

            if not match:
                if family_name:
                    match = compare_articles(ref_title, potential_title, ref_authors_raw, family_name)
                else:
                    match = compare_articles(ref_title, potential_title, ref_authors_raw, first_author_str)

            match_crossref = False

            if not match:
                if str(full_ref).lower().replace(" ", "").strip() == "nan":
                    if potential_title != "nan":
                        #if the full ref is not given, we search for first author+ titile + year
                        full_ref = f'{row[first_author_col]}, {potential_title}, {year_ref}'
                    else: 
                        continue
                #check if the doi returned by crossref matches the cited doi
                match_crossref = find_doi_crossref(full_ref, cited_doi)

            if match or match_crossref:
                passages.append({col: str(row[col]) if col in row and pd.notna(row[col]) else "" for col in context_cols})
                sections.append(section)
                full_ref_found_crossref = full_ref
                crossref_found = True

        else:
            print("Edge already processed, passing directly to the next row")

    return passages, sections


def search_citation_passage(total_citations_to_find, cited_doi, citing_doi, article_meta, safe_citing, doi_to_path, citation_data, edge, dois_csv, doc_path):
    '''Search the corresponding citation passage for a pair citing doi and cited doi. Searches in the corresponding csv citation file if there is a doi matching the cited doi we are looking for.
    Returns the citation passages and the section'''
    total_citations_found = 0

    #if the edge was already processed but the section is missing edge_processed will be true
    edge_processed = False
    language_choice = ["en"]

    sources = {"tei":["ISTEX", "Wiley", "Unpaywall", "Crossref"], "xml":["PMC", "PLOS", "Elsevier"]}
    cited_article = article_meta.get(cited_doi, {})
    citing_article = article_meta.get(citing_doi, {})
    lg = citing_article.get("lg")
    if lg is None:
        if "query_article" in doc_path:
            lg = "en"
        else:
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

    df_citing = normalize_and_filter_df(df_citing_all, citing_doi, source, sources)

    if df_citing is None:
        return "", ""
    
    passages = []
    sections = []
    context_cols = ["S-3", "S-2", "S-1", "S-0", "CITATION", "S+0", "S+1", "S+2", "S+3"]
    passages, sections, df_match = extract_passages_exact_match(df_citing, cited_doi, context_cols)

    if passages:
        return passages, sections
  
    clean_ref_authors = clean_text(ref_authors_raw).strip()
    first_letter_ref_author = clean_ref_authors[0].lower() if clean_ref_authors else None
    surname_only = len(clean_ref_authors.split()) < 2
    df_surname = pd.DataFrame()

    first_author_col = None
    #colomn name changes depending on source
    for col in ["FIRST_AUTHOR_SURNAME", "AUTHOR1_SURNAME"]:
        if col in df_citing_all.columns:
            first_author_col = col
            break

    if surname_only:
        #only the surname is given as author name
        df_citing[first_author_col] = df_citing[first_author_col].apply(lambda x: unidecode(x).lower() if isinstance(x, str) else x)
        df_surname = df_citing[df_citing[first_author_col] == unidecode(clean_ref_authors.lower())]


    #Search for family name type "Firstname Lastname"
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
        
    if family_name:
        df_citing[first_author_col] = df_citing[first_author_col].apply(lambda x: unidecode(x).lower() if isinstance(x, str) else x)
        df_surname = df_citing[df_citing[first_author_col] == unidecode(family_name.lower())]

    if not df_surname.empty:
        passages, sections = search_match(df_citing, edge, context_cols, df_citing_all, first_author_str, ref_title, ref_authors_raw, first_letter_ref_author, surname_only, family_name, cited_doi, citing_doi, first_author_col)
        return passages, sections
    else:
        df_pmids = pd.DataFrame()
        if "anneso" in doc_path:
            #if it was a document processed with AlvisNLP then there might be pmids available for the different references
            #get the pmid from the cited doi and try to find it in the references
            pmid = pmid_from_doi(cited_doi)
            
            if pmid is not None:
                pmid_float = float(pmid)
                df_pmids = df_citing[df_citing["REF_PMID"] == pmid_float]
        
        if not df_pmids.empty:
            passages, sections = search_match(
                df_pmids, edge, context_cols, df_citing_all,
                first_author_str, ref_title, ref_authors_raw,
                first_letter_ref_author, surname_only, family_name,
                cited_doi, citing_doi, first_author_col)
            return passages, sections

        #if none of the previous attempt work then search with crossref
        passages, sections = search_match(df_citing, edge, context_cols, df_citing_all, first_author_str, ref_title, ref_authors_raw, first_letter_ref_author, surname_only, family_name, cited_doi, citing_doi, first_author_col)


    return passages, sections


def complete_graph_enriched(graph_enriched):
    """Complete the graph with citation passages including section."""

    csv_files = glob.glob("../../../AlvisNLP/anneso/outputAlvisNLP/citing_articles/*/*.csv")
    dois_csv = {f.split("/")[-1].replace("_references_citing.csv", ""): f for f in csv_files}
    doi_to_path = build_tei_lookup()
    citation_data = preload_citation_data(doi_to_path)
    doi_to_path.update({f.split("/")[-1].replace("_references_citing.csv", ""): f for f in csv_files if f.split("/")[-1].replace("_references_citing.csv", "") not in doi_to_path})
    article_meta = {normalize_doi(a["doi"]): a for a in graph_enriched.get("articles", [])}

    total_citations_to_find = 0
    total_citations_found = 0

    edges = graph_enriched.get("edges", [])
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
                json.dump(graph_enriched, f, indent=2)

    year_art = graph_enriched.get("year_arts", {})

    new_graph = {"years": graph_enriched.get("years", []),
        "year_arts": year_art,
        "articles": graph_enriched.get("articles", []), 
        "findings": graph_enriched.get("findings", []),
        "edges": edges}

    #final save
    with open(output_path, "w") as f:
        json.dump(new_graph, f, indent=2)
    print(f"Saved updated graph to {output_path}")
    



if __name__ == "__main__":
    graph_path = "graph_query_ref_citing.json"
    output_path = "graph_with_citations_V2.json"

    graph = load_graph(graph_path)
    graph_filtered = filter_graph(graph)

    if os.path.exists(output_path):
        graph_enriched = load_graph(output_path)
        print(f"Number of edges in graph_enriched: {len(graph_enriched.get('edges', []))}")
        complete_graph_enriched(graph_enriched)
    else:
        complete_graph_enriched(graph_filtered)