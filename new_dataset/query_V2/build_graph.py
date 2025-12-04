
import re
import pandas as pd
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import pickle
from unidecode import unidecode
from collections import defaultdict
import numpy as np
import ast
import string
import glob
import os
import config_APIs as config
import requests
from utils import normalize_doi

def retrieve_references_crossref(doi):
    """Query Crossref API to retrieve language, authors and title information."""

    language = "unknown"
    title = "TO_BE_SEARCHED"
    authors = "TO_BE_SEARCHED"

    url = f"https://api.crossref.org/works/{doi}"

    headers = {"User-Agent": config.HEADER}
    proxies = {
        "http": f"http://{config.PROXY_USERNAME}:{config.PROXY_PASS}@{config.PROXY}",
        "https": f"http://{config.PROXY_USERNAME}:{config.PROXY_PASS}@{config.PROXY}",
    }

    try:
        response = requests.get(url, timeout=10, headers=headers, proxies=proxies)
        response.raise_for_status()

        msg = response.json().get("message", {})

        language = msg.get("language", language)
        title = msg.get("title", [title])[0]

        authors_raw = msg.get("author", [])
        if authors_raw:
            authors = "; ".join(f"{a.get('given', '')} {a.get('family', '')}".strip()for a in authors_raw)

    except requests.exceptions.ReadTimeout:
        pass
    except Exception as e:
        print(f"Crossref error DOI {doi}: {e}")

    return language, title, authors
    


def find_author(author_json):
    """
    detect correct surnames of all authors out of the author-bibtex-field
    :param author_json: author-field of an article from json export
    :return: list of all surnames
    """
    if "," not in author_json:
        pattern = re.compile(r"(?P<name>[A-Za-z\-]+)(?: +and +| +AND +|$)")
    else:
        pattern = re.compile(r"(?:^|and +|AND +)(?P<name>[A-Za-z\- ]+)")
    authors = pattern.findall(author_json)
    return authors

def extract_ref_metadata(ref):
    """
    Extrait le DOI, le titre (simplifié), les auteurs (simplifiés) et l'année d'une chaîne de référence brute.
    """
    #DOI (format standard)
    doi_match = re.search(r"10\.\d{4,9}/\S+", ref)
    ref_doi = doi_match.group(0) if doi_match else ""


    #Année (format: 4 chiffres, généralement entre parenthèses ou seuls)
    year_match = re.search(r"\b(19|20)\d{2}\b", ref)
    ref_year = year_match.group(0) if year_match else ""

    
    parts = ref.split(", ")

    ref_authors = parts[0].strip()
    if len(parts) >= 3:
        ref_title = parts[2].strip()
    else:
        ref_title = ''

    return {"doi": ref_doi, "year": ref_year, "authors": ref_authors, "title": ref_title }
    
def save_counts(counts_dic, path):
    with open(path, "wb") as f:
        pickle.dump(counts_dic, f)

def load_counts(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def clean_text(text):
    '''Supprime la ponctuation, guillemets, tirets, slashes, et retourne le texte nettoyé.'''
    text = unidecode(text.lower())
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace("’", "").replace("–", "").replace("—", "").replace("“", "").replace("”", "")
    text = text.replace("-", "").replace("/", "").replace("_", "").replace("  ", "")
    return text.strip()

def compare_articles(ref_title, title_B, ref_authors, first_author_B):
    '''Compares two articles by title and first authors, returns if there is an exact match'''
    #Normalize titles
    ref_title_norm = clean_text(unidecode(str(ref_title).lower()).strip())
    title_B_norm = clean_text(unidecode(str(title_B).lower()).strip().replace("-", "").replace("_", ""))

    #Normalize authors
    if isinstance(ref_authors, list) and len(ref_authors) > 0:
        ref_first_author = unidecode(ref_authors[0].lower()).strip().replace("-", "").replace("_", "")
    else:
        ref_first_author = unidecode(str(ref_authors).lower()).strip().replace("-", "").replace("_", "")

    first_author_B_norm = unidecode(str(first_author_B).lower()).strip().replace("-", "").replace("_", "")
    title_match = ref_title_norm == title_B_norm
    author_match = ref_first_author == first_author_B_norm
    if title_match and author_match:
        return True
    return False


def get_citing_article_dic(articles_citing_df, doi, referenced_year, ref_authors, ref_title, Articles_dois):
    """
    Récupère les articles citants pour un doi donné et construit :
      - une liste de dicts représentant ces articles
      - une liste de tuples représentant les edges (citing_doi → doi)
    
    Args:
        articles_citing_df: DataFrame avec colonnes ["ref_doi", "citing_doi", "title", "author", "year", "source"]
        doi: str (doi de l'article cité)
        Articles_dois: set (contient déjà les DOIs connus, pour éviter doublons)
    
    Returns:
        citing_articles: list[dict]
        edges: list[tuple]
        wrong dates: list[dict] when the citing article is more than one year older than the cited article
    """
    citing_articles = []
    edges = []
    wrong_dates = []

    citing_rows = articles_citing_df.loc[articles_citing_df["ref_doi"] == doi]

    if citing_rows.empty:
        return citing_articles, edges, wrong_dates

    # normalize citing DOIs
    citing_rows = citing_rows.assign(citing_doi=citing_rows["citing_doi"].map(normalize_doi))

    #build dictionary of citing articles with metadata
    for citing_doi, citing_row in citing_rows.drop_duplicates("citing_doi").set_index("citing_doi").iterrows():
        if citing_doi and citing_doi not in Articles_dois:
            citing_art_dic = {
                "year": citing_row.get("year", ""),
                "author": citing_row.get("author", ""),
                "doi": citing_doi,
                "title": citing_row.get("title", ""),
                "source": citing_row.get("source", ""),
                "self-citation": citing_row.get("self-citation", ""),
                "lg": citing_row.get("lg", "unknown")
            }

            if not citing_row.get("year", ""):
                continue

            if not citing_doi or citing_doi.lower() in {"none", "nan", "null"}:
                continue

            try:
                citing_year_int = int(citing_row.get("year", ""))
            except Exception:
                print(f"Failed to convert to int year : {citing_row.get("year", "")}")
                citing_year_int = None
            
            referenced_year_int = None

            if citing_year_int is not None and referenced_year not in ("", None):
                try:
                    referenced_year_int = int(referenced_year)
                except Exception:
                    referenced_year_int = None

                if referenced_year_int is not None and citing_year_int < referenced_year_int:
                    
                    if referenced_year_int -  citing_year_int!=1:
                        print(f"\nIssue : citing year {citing_row.get('year', '')} earlier than referenced year {referenced_year} "
                            f"for citing doi {citing_doi} referenced doi {doi}")
                    
                    wrong_dates.append({
                    "referenced_doi": doi,
                    "referenced_year": referenced_year_int,
                    "citing_doi": citing_doi,
                    "citing_year": citing_year_int,
                    "source": citing_row.get("source", "")})


            citing_articles.append(citing_art_dic)
            Articles_dois.add(citing_doi)

        if citing_doi and doi:
            edges.append((citing_doi, doi))

    return citing_articles, edges, wrong_dates




def build_graph(articles_query, articles_citing, output_path):
    '''Build a first graph with articles we want to study that are in english and for which dates are correct'''
    
    all_articles = articles_query
    all_wrong_dates = []
    #if we also want to add the articles from references
    #articles_ref_dict = {normalize_doi(art["doi"]): art for art in articles_reference if art.get("doi")}

    total_other_citing_sans_doublons = 0

    graph = {}

    years = set()

    #Extract years from query articles
    for article in all_articles:
        year = article.get("year")
        try:
            years.add(int(float(year)))
        except:
            pass

    print(articles_citing.head())
    citing_rows_unique = articles_citing.drop_duplicates("citing_doi").set_index("citing_doi")

    for citing_doi, citing_row in citing_rows_unique.iterrows():
        year = citing_row.get("year")
        try:
            years.add(int(float(year)))
        except:
            pass
       
    graph["years"] = sorted(years)

    #Build dictionary year → DOIs
    graph["year_arts"] = {year: [] for year in graph["years"]}

    #indexing query articles in the year doi dic
    for article in all_articles:
        year = article.get("year")
        try:
            y = int(float(year))
            graph["year_arts"][y].append(article["doi"])
        except:
            pass


    #indexing citing articles in the year doi dic
    for citing_doi, citing_row in citing_rows_unique.iterrows():
        year = citing_row.get("year")
        try:
            y = int(float(year))
            graph["year_arts"][y].append(citing_doi)
        except:
            pass
    
    Findings = [] #a finding is a paper that has been cited at least once and is not a review
    Articles = [] #articles that appear in the dataset and can cite others
    

    #relation between the article and the finding
    Edges_set = set()
    Articles_dois = set()
    Findings_dois = set()

    Findings_counts_dic = {}

    for article in tqdm(articles_query, desc="Processing articles for edges"):
        year, authors, doi, title  = article.get("year", ""), find_author(article.get("author")), normalize_doi(article.get("doi")), article.get("title")

        current_art_dic = {"year": year, "author": authors, "doi": doi, "title": title}

        #un article == un noeud
        if doi not in Articles_dois:
            Articles.append(current_art_dic)
            Articles_dois.add(doi)

        #index citing articles in the graph
        citing_articles, citing_edges, wrong_dates = get_citing_article_dic( articles_citing, doi, year, authors, title, Articles_dois)
        Articles.extend(citing_articles)
        all_wrong_dates.extend(wrong_dates)

        if len(citing_articles) > 1:
            Findings.append(current_art_dic)
            Findings_dois.add(doi)
        
        for edge in set(citing_edges):
            if edge not in Edges_set:
                Edges_set.add(edge)
                citing_doi, target_doi = edge
                Findings_counts_dic[target_doi] = Findings_counts_dic.get(target_doi, 0) + 1


        
    graph["articles"] = Articles
    graph["findings"] = Findings

    graph["edges"] = [{"citing_doi": f, "cited_doi": t} for f, t in Edges_set]
    
    

    with open(output_path, "w", encoding="utf-8") as jf:
        json.dump(graph, jf, indent=2, ensure_ascii=False)

    wrong_dates_file="query_citing_wrong_dates.json"

    with open(wrong_dates_file, "w", encoding="utf-8") as wf:
        json.dump(all_wrong_dates, wf, indent=2, ensure_ascii=False)


    return Findings_counts_dic

def plot_distribution(Findings_counts_dic):
    """
    Plot distribution with:
    - x-axis = articles
    - y-axis = number of citations
    Limited to 100 citations
    """
    
    moyenne_cited_by_par_article = 0

    set_citation_counts = set()
    distribution = {}
    list_of_cited_by_values = []

    for key, value in Findings_counts_dic.items():
        list_of_cited_by_values.append(value)
        if value >= 328:
            print(f"DOI {key} has {value} citations")
        #key = doi, value = number of cited by
        if value not in set_citation_counts:
            set_citation_counts.add(value)
            
    counts = np.array(list_of_cited_by_values)
    
    #stats on citations per finding
    mean_count = np.mean(counts)
    median_count = np.median(counts)
    std_count = np.std(counts)
    min_count = np.min(counts)
    max_count = np.max(counts)

    print("=== Statistics of citation counts ===")
    print(f"Mean: {mean_count:.2f}")
    print(f"Median: {median_count}")
    print(f"Std: {std_count:.2f}")
    print(f"Min: {min_count}")
    print(f"Max: {max_count}")
    print("===================================")

    for count in set_citation_counts:
        distribution[count] = 0
        for key, value in Findings_counts_dic.items():
            if value == count:
                distribution[count]+=1


    print(distribution)
    distribution = {k: v for k, v in distribution.items() if k <= 100}

    x = sorted(distribution.keys())
    y = [distribution[k] for k in x]

    plt.figure(figsize=(10,5))
    plt.bar(x, y, color="skyblue", edgecolor="black")
    plt.xlabel("Number of citations (per article)")
    plt.ylabel("Number of articles")
    plt.title("Distribution of citations across articles (≤100 citations)")
    plt.xticks(range(0, max(x)+1, 5))
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()
    
def histogramme_types_ref():
    '''Plots counts of doc types for the references from the articles got with the WOS query'''
    info_reference_type_path = "info_reference_openalex_dedup.tsv"
    df_info_ref = pd.read_csv(info_reference_type_path, sep="\t").fillna("")
    Reference_info_types = df_info_ref["Reference_Type"]

    count_types = Reference_info_types.value_counts()
    #print("Nombre de documents par type :")
    #print(count_types)

    plt.figure(figsize=(12,7))
    count_types.plot(kind="bar")
    plt.title("Nombre de documents par type de référence", fontsize=24)
    plt.xlabel("Type de référence", fontsize=18)
    plt.ylabel("Nombre de documents", fontsize=18)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

def calculate_number_references(df):
    info_reference_type_path = "info_reference_openalex_dedup.tsv"
    df_info_ref = pd.read_csv(info_reference_type_path, sep="\t").fillna("")
    
    df_info_ref["Reference_DOI"] = df_info_ref["Reference_DOI"].apply(normalize_doi)

    Reference_info_dois_set = set(df_info_ref["Reference_DOI"])    
    Reference_info_types = df_info_ref["Reference_Type"]
    Reference_info_titles = df_info_ref["Reference_Title"]

    non_processed_ref = set()

    print("Total references processed : ", len(Reference_info_dois_set))

    #keep articles only
    df_articles = df_info_ref[df_info_ref["Reference_Type"] == "article"]
    df_articles.to_csv("df_references_articles_only.csv", sep=",", index=False)

    article_dois = set(df_articles["Reference_DOI"])
    print("TOTAL ARTICLE REFERENCE", len(article_dois))

    present_in_openalex_df = set()

    #Mapping DOI → titre
    doi_to_title = dict(zip(df_articles["Reference_DOI"], df_articles["Reference_Title"]))

    total_ref_avec_doublons = []
    total_ref_avec_doi = set()
    total_ref_article_avec_doi = set()

    references = df["CR"].tolist()

    for row in references:
        ref_list = [ref.strip() for ref in row.split(";") if ref.strip()]
        
        for ref in ref_list:
            ref_meta = extract_ref_metadata(ref)
            ref_doi = normalize_doi(ref_meta.get("doi"))

            total_ref_avec_doublons.append(ref)

            if ref_doi:
                total_ref_avec_doi.add(ref_doi)
                
                if ref_doi in article_dois:
                    total_ref_article_avec_doi.add(ref_doi)

                if ref_doi and ref_doi not in Reference_info_dois_set:
                    non_processed_ref.add(ref_doi)

    count_doi_processed = 0
    count_doi_non_processed = 0

    correcly_found = set()
    for doi in total_ref_avec_doi:
        if doi in Reference_info_dois_set:
            count_doi_processed +=1
            correcly_found.add(doi)
        else:
            count_doi_non_processed += 1

    print("COUNT PROCESSED : ", count_doi_processed)
    print("COUNT NON PROCESSED : ", count_doi_non_processed)


    total_dois_to_process_openalex = count_doi_processed + count_doi_non_processed
    print("TOTAL : ", total_dois_to_process_openalex)

    total_ref_sans_doublons = set(total_ref_avec_doublons)

    print(f"TOTAL REFERENCE AVEC DOUBLONS : {len(total_ref_avec_doublons)}")
    print(f"TOTAL REFERENCE SANS DOUBLONS : {len(total_ref_sans_doublons)}")

    print(f"TOTAL REFERENCE AVEC DOIS SANS DOUBLONS : {len(total_ref_avec_doi)}")

    print(f"TOTAL REFERENCE ARTICLE AVEC DOIS SANS DOUBLONS : {len(total_ref_article_avec_doi)}")
    print(f"NON PROCESSED BY OPEN ALEX : {len(non_processed_ref)}")


    return total_ref_article_avec_doi



def get_WOS_ref(file_path):
    """
    Load WOS references file (TSV) and extract publication year, DOI, title, authors, and citation counts
    """
    df_WOS = pd.read_csv(file_path, sep="\t").fillna("")
    df = df_WOS[df_WOS["DT"].str.contains("Article", case=False, na=False)]
    articles = []

    for i, row in df.iterrows():
        article = {
            "title": row["TI"],
            "author": row["AU"],
            "doi": row["DI"],
            "year": row["PY"],
            "references": row["CR"],
            "wos_citing_count" : row["Z9"]}

        articles.append(article)

    return articles

def extract_doi_from_ids(citing_ids: str) -> str:
    match = re.search(r"doi:([^\s]+)", citing_ids)
    if match:
        return match.group(1).strip().lower()
    return ""


def check_date(ref_year, citing_year):
    '''Return if the citing year is a valid date or not'''
    valid_date = False

    if ref_year < citing_year or ref_year == citing_year:
        valid_date = True
    elif ref_year - citing_year == 1:
        valid_date = True
    else:
        pass
    return valid_date

def get_WOS_citing(articles_query, file_path_open_alex, file_path_coci, output_csv=None):
    """
    Load OpenAlex and COCI citing articles and extract publication year, DOI, title, authors.
    Returns:
        - cited_articles: dict {ref_doi: {citing_doi: {title, author, year, source}}}
        - df_all: DataFrame with flattened results
    """
    cited_articles = {}
    rows = []
    count_common_found = 0
    coci_new_relation = 0
    count_new_doi_processed = 0

    total_citing_openalex = set()
    total_citing_coci = set()
    total_relations = set()
    citing_non_found_with_openalex = set() 

    df_openalex = pd.read_csv(file_path_open_alex).fillna("")
    df_openalex["Doi_Reference_Processed"] = df_openalex["Doi_Reference_Processed"].map(normalize_doi)

    df_coci = pd.read_csv(file_path_coci).fillna("")
    
    df_coci["Citing_DOI_norm"] = df_coci["citing"].apply(lambda x: normalize_doi(extract_doi_from_ids(x)) if x else None)
    df_coci["Citing_DOI"] = df_coci["citing"].apply(lambda x: extract_doi_from_ids(x) if x else None)

    df_none_doi = {}
    dic_errors = {}
    possible_coci_errors = {}
    possible_openalex_errors = {}
    possible_both_errors = {}

    other_languages_dic = {}


    for article in articles_query:
        
        ref_doi = normalize_doi(article["doi"])

        try:
            ref_year = int(article["year"])
        except:
            continue
        

        wos_count = article.get("wos_citing_count", None)

        if not ref_doi:
            continue

        # Filter citing works for this DOI
        current_work_open_alex_df = df_openalex[df_openalex["Doi_Reference_Processed"] == ref_doi]
        total_count_citing_openalex = len(current_work_open_alex_df["Citing_DOI"])

        current_work_coci_df = df_coci[df_coci["Doi_Reference_Processed"] == ref_doi]
        total_count_citing_coci = len(current_work_coci_df["citing"])

        for _, row in current_work_open_alex_df.iterrows():
            citing_doi = normalize_doi(row["Citing_DOI"])
            
            if not citing_doi or citing_doi == "error":
                df_none_doi.setdefault(ref_doi, {}).setdefault("Openalex", 0)
                df_none_doi[ref_doi]["Openalex"] += 1
                continue

            if ref_doi not in cited_articles:
                cited_articles[ref_doi] = {}

            if citing_doi not in cited_articles[ref_doi]:
                total_citing_openalex.add(citing_doi)
                total_relations.add((ref_doi, citing_doi))

                # search if the citing DOI is in COCI to get self citation info
                coci_match = current_work_coci_df[current_work_coci_df["Citing_DOI_norm"] == citing_doi]
                
                if not coci_match.empty:
                    sc = coci_match["author_sc"].iloc[0]

                    creation_val = str(coci_match["creation"].iloc[0])
                    if creation_val and creation_val[:4].isdigit():
                        year_coci = int(creation_val[:4])
                    else:
                        year_coci = None
                else:
                    sc = "unknown"
                    year_coci = None

                #OpenAlex year if valid, fallback to COCI if needed
                try:
                    citing_openalex_year = int(row["Citing_Publication_Year"])
                except Exception:
                    citing_openalex_year = year_coci if year_coci is not None else "unknown"

                meta = {
                    "title": row["Citing_Title"],
                    "author": row["Citing_Authors"],
                    "year": citing_openalex_year,
                    "source": "OpenAlex",
                    "self-citation": sc,
                    "lg": row["Citing_Language"]
                }
                

                is_valid_date = check_date(ref_year, int(row["Citing_Publication_Year"]))
                
                if is_valid_date:
                    cited_articles[ref_doi][citing_doi] = meta
                    rows.append({"ref_doi": ref_doi, "citing_doi": citing_doi, **meta})

                    lg = row["Citing_Language"]

                    #TO BE UPDATED :add other languages ?
                    if lg != "en" and lg not in other_languages_dic:
                        other_languages_dic[lg]={}
                            
                    if lg != "en" and citing_doi not in other_languages_dic[lg]:
                        other_languages_dic[lg][citing_doi]={"title": row["Citing_Title"], "authors": row["Citing_Authors"],"cited_doi": ref_doi}

                else:
                    dic_errors.setdefault(ref_doi, [])

                    dic_errors[ref_doi].append({"ref_year": ref_year, "Source": "OpenAlex", "citing_doi": citing_doi, "citing_year": citing_openalex_year})
                    
                    print("\n")
                    print(f"Processing reference doi : {ref_doi}")
                    print(f"Error openalex citing doi {citing_doi}")


        #valeur de différence fixée empiriquement en regardant les différences des comptes de citations pour les cas d'erreur
        #les deux dois ici dans la liste sont des cas d'erreurs
        if total_count_citing_openalex - total_count_citing_coci > -10 and ref_doi not in ["10.3389/fpls.2019.00001", "10.1073/pnas.1105664108"]:
            for _, row in current_work_coci_df.iterrows():
                citing_coci = row["Citing_DOI_norm"]
                if not citing_coci:
                    df_none_doi.setdefault(ref_doi, {}).setdefault("COCI", 0)
                    df_none_doi[ref_doi]["COCI"] += 1
                    continue    

                if citing_coci not in cited_articles[ref_doi]:
                    dates = row["creation"]
                    try:
                        year_citing = int(dates[:4])
                    except:
                        continue

                    sc = row["author_sc"]
                    lg, title, author = retrieve_references_crossref(row["Citing_DOI"])
                    meta = {
                        "title": title,
                        "author": author,
                        "year": year_citing,
                        "source": "COCI",
                        "self-citation": sc,
                        "lg": lg
                        }

                    is_valid_date = check_date(ref_year, year_citing)
            
                    if is_valid_date:
                        cited_articles[ref_doi][citing_coci] = meta
                        rows.append({"ref_doi": ref_doi, "citing_doi": citing_coci, **meta})

                        #To be updated depending of language choice
                        if lg != "en" and lg not in other_languages_dic:
                            other_languages_dic[lg]={}

                        if lg != "en" and citing_doi not in other_languages_dic[lg]:
                            
                            other_languages_dic[lg][citing_doi]={"title": title, "authors": author,"cited_doi": ref_doi}

                    else:
                        #create a dictionary of errors
                        dic_errors.setdefault(ref_doi, [])
                        dic_errors[ref_doi].append({"ref_year": ref_year, "Source": "COCI", "citing_doi": citing_coci, "citing_year": year_citing})
    
        elif total_count_citing_openalex > wos_count:
            possible_openalex_errors.setdefault(ref_doi, [])
            possible_openalex_errors[ref_doi].append({"ref_year": ref_year, "Count_openalex": total_count_citing_openalex, "Count_WOS": wos_count, "Count_COCI": total_count_citing_coci, "difference":total_count_citing_openalex-wos_count})

        if total_count_citing_coci > wos_count:
            possible_coci_errors.setdefault(ref_doi, [])
            possible_coci_errors[ref_doi].append({"ref_year": ref_year, "Count_openalex": total_count_citing_openalex, "Count_WOS": wos_count, "Count_COCI": total_count_citing_coci, "difference":total_count_citing_coci-wos_count})

        if total_count_citing_coci > wos_count or total_count_citing_openalex > wos_count:
            possible_both_errors.setdefault(ref_doi, [])
            possible_both_errors[ref_doi].append({"ref_year": ref_year, "Count_openalex": total_count_citing_openalex, "Count_WOS": wos_count, "Count_COCI": total_count_citing_coci, "difference_Openalex_WOS":total_count_citing_openalex-wos_count, "difference_COCI_WOS":total_count_citing_coci-wos_count})


    #all citing articles metadata
    df_all = pd.DataFrame(rows)

    #cases where the doi does not exists
    rows_none_doi = []
    for ref_doi, sources in df_none_doi.items():
        for source, count in sources.items():
            rows_none_doi.append({
                "ref_doi": ref_doi,
                "source": source,
                "count_none_doi": count
            })

    df_none_doi_flat = pd.DataFrame(rows_none_doi)
    df_none_doi_flat.to_csv("errors/df_none_doi.csv", index=False)

    #errors (citation counts and dates)
    rows_errors = []
    for ref_doi, error_list in dic_errors.items():
        for err in error_list:
            rows_errors.append({
                "ref_doi": ref_doi,
                "ref_year": err.get("ref_year"),
                "source": err.get("Source"),
                "citing_doi": err.get("citing_doi"),
                "citing_year": err.get("citing_year")
            })

    df_errors_flat = pd.DataFrame(rows_errors)
    df_errors_flat.to_csv("errors/dic_errors.csv", index=False)

    #Possible errors from openalex (openalex count > to WOS or count COCI > Openalex)
    rows_openalex_errors = []
    for ref_doi, error_list in possible_openalex_errors.items():
        for err in error_list:
            rows_openalex_errors.append({
                "ref_doi": ref_doi,
                "ref_year": err.get("ref_year"),
                "Count_openalex": err.get("Count_openalex"),
                "Count_WOS": err.get("Count_WOS"),
                "Count_COCI": err.get("Count_COCI"),
                "difference_openalex_wos": err.get("difference")
            })

    '''
    #If we want to save the errors
    df_openalex_errors_flat = pd.DataFrame(rows_openalex_errors)
    df_openalex_errors_flat = df_openalex_errors_flat.sort_values(by="difference_openalex_wos", ascending=False)
    df_openalex_errors_flat.to_csv("errors/possible_openalex_errors.csv", index=False)

    #possible_coci_errors
    rows_coci_errors = []
    for ref_doi, error_list in possible_coci_errors.items():
        for err in error_list:
            rows_coci_errors.append({
                "ref_doi": ref_doi,
                "ref_year": err.get("ref_year"),
                "Count_openalex": err.get("Count_openalex"),
                "Count_WOS": err.get("Count_WOS"),
                "Count_COCI": err.get("Count_COCI"),
                "difference_coci_wos": err.get("difference")
            })

    df_coci_errors_flat = pd.DataFrame(rows_coci_errors)
    df_coci_errors_flat = df_coci_errors_flat.sort_values(by="difference_coci_wos", ascending=False)
    df_coci_errors_flat.to_csv("errors/possible_coci_errors.csv", index=False)

    rows_both_errors = []
    for ref_doi, error_list in possible_both_errors.items():
        for err in error_list:
            rows_both_errors.append({
                "ref_doi": ref_doi,
                "ref_year": err.get("ref_year"),
                "Count_openalex": err.get("Count_openalex"),
                "Count_WOS": err.get("Count_WOS"),
                "Count_COCI": err.get("Count_COCI"),
                "difference_Openalex_WOS": err.get("difference_Openalex_WOS"),
                "difference_COCI_WOS": err.get("difference_COCI_WOS")
            })

    df_both_errors_flat = pd.DataFrame(rows_both_errors)

    #sort by COCI difference (highest first)
    df_both_errors_flat = df_both_errors_flat.sort_values(by="difference_COCI_WOS", ascending=False)

    df_both_errors_flat.to_csv("errors/possible_both_errors.csv", index=False)'''

    #to analyse content of citing languages that are not in english
    with open('other_language_citing_articles.json', 'w', encoding='utf-8') as f:
        json.dump(other_languages_dic, f, indent=2, ensure_ascii=False)

    return cited_articles, df_all


def load_references():
    WOS_references_path = "references_WOS.tsv"
    
    OpenAlex_art_citing_query = "OPENALEX_citing_references_citations.csv"
    COCI_art_citing_query = "Crossref_COCI_references_citations.csv"
    articles_query = get_WOS_ref(WOS_references_path)
    print(f"Len articles rst of query : {len(articles_query)}")
    articles_citing, df_all_articles_citing = get_WOS_citing(articles_query, OpenAlex_art_citing_query, COCI_art_citing_query , "Citing_ariticles_per_doi_combined.csv")
    print(f"\nLen articles in citing : {len(articles_citing)}")
    
    output_path = "graph_query_ref_citing.json"
    articles_citing_the_references, df_articles_citing_the_references = [], pd.DataFrame()
    citing_doc_set = set(df_all_articles_citing["citing_doi"].dropna().unique())
    print(f"Len citing docs without duplicates: {len(citing_doc_set)}")

    Findings_counts_dic = build_graph(articles_query, df_all_articles_citing, output_path)
    output_counts = "findings_counts_without_citing_references.pkl"
    save_counts(Findings_counts_dic, output_counts )
    Findings_counts_dic = load_counts(output_counts)
    plot_distribution(Findings_counts_dic)


if __name__ == "__main__":
    load_references()