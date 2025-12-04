import pandas as pd
import matplotlib.pyplot as plt
from unidecode import unidecode
from utils import get_downloaded_dois, normalize_doi
import json
import config_APIs as config
import ast 
from extract_pmc_istex_unpaywall import try_sources, Elsevier_extract_text, Wiley_extract_text, retrieve_references_crossref, retreive_microbilogy_soc

def plot_publishers_by_count(publishers_list, top_n=50):
    """
    Plot the top N most frequent publishers by percentage of articles.

    Args:
        publishers_list (list): List of publisher names.
        top_n (int): Number of top publishers to show (default=50).
    """
    publisher_series = pd.Series(publishers_list)
    total = len(publisher_series)
    publisher_percent = (publisher_series.value_counts(normalize=True).head(top_n)) * 100

    plt.figure(figsize=(14, 6))
    publisher_percent.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.ylabel('Percentage of articles')
    plt.xlabel('Publisher')
    plt.title(f'Top {top_n} Publishers by percentage of articles')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def union_plot_publishers_by_count(publishers_list, DOIs_WOS, journals, dowloaded_dois, top_n=50):

    new_publisher_list = []
    set_other = set()
    dic_theorical_access = {"oxford univ press":0, "amer soc microbiology":0, "taylor & francis ltd":0, "amer phytopathological soc":0, "microbiology soc":0}

    with open("revues_sous_abonnement_par_plateforme.json", "r") as f:
        journals_dict = json.load(f)

    asm_journals = journals_dict["ASM_journals"]
    taylor_journals = journals_dict["taylor_and_francis_revue"]
    oxford_journals = journals_dict["oxford_subscribed_journals"]
    APS_journals = journals_dict["amer phytopathological soc"]
    microbiology_journals = journals_dict["microbiology"]

    repartition_journals_oxford = {}
    
    PUBLISHER_MAP = {
    "elsevier": "elsevier",
    "springer": "springer",
    "wiley": "wiley",
    "blackwell": "wiley",
    "oxford": "oxford univ press",
    "mdpi": "mdpi",
    "cambridge": "cambridge",
    "nature": "nature portfolio",
    "taylor & francis": "taylor & francis ltd",
    "bmc": "bmc",
    "peerj": "peerj",
    "c a b": "cab international",
    "royal soc chemistry": "royal soc chemistry",
    "royal soc": "royal soc chemistry",
    "frontiers": "frontiers media sa",
    "mary ann liebert": "mary ann liebert",
    "hindawi": "hindawi",
    "sage publications": "sage publications",
    "kluwer acad": "kluwer academic publ",
    "akademiai kiado": "akademiai kiado",
    "canadian science publishing": "canadian science publishing",
    "de gruyter": "walter de gruyter",
    "pensoft publ": "pensoft publ",
    "elife": "elife sciences publ",
    "academic press": "academic press inc",
    "plenum publ": "plenum publ corp",
    "entomological soc amer": "entomological soc amer",
    }

    ACCESS_MAP = {
        "microbiology soc": microbiology_journals,
        "amer phytopathological soc": APS_journals,
        "amer soc microbiology": asm_journals,
        "oxford univ press": oxford_journals,
        "taylor & francis ltd": taylor_journals,
    }

    for journal, publisher_raw, doi in zip(journals, publishers_list, DOIs_WOS):

        #theoretical institutional access based on journal lists
        for access_label, journal_set in ACCESS_MAP.items():
            if journal in journal_set:
                dic_theorical_access[access_label] += 1
                break

        #normalize publisher
        publisher = publisher_raw
        for key, mapped in PUBLISHER_MAP.items():
            if key in publisher_raw:
                publisher = mapped
                break
        if publisher == "oxford univ press":
            repartition_journals_oxford[journal] = (repartition_journals_oxford.get(journal, 0) + 1 )

        if publisher_raw not in PUBLISHER_MAP and journal in microbiology_journals:
            publisher = "microbiology soc"

        new_publisher_list.append(publisher)
        if publisher_raw not in PUBLISHER_MAP:
            set_other.add(publisher)

    publisher_series = pd.Series(new_publisher_list)
    publisher_counts_df = publisher_series.value_counts().reset_index()
    publisher_counts_df.columns = ['Publisher', 'Article_Count']
    publisher_counts_df.to_csv('publisher_counts.csv', index=False)

    #map doi per prefix
    doi_mapping = {}
    for i in range(len(set(new_publisher_list))):
        publisher = new_publisher_list[i]
        doi = normalize_doi(DOIs_WOS[i])
        first_part_doi = doi.split("/")[0]
        if not first_part_doi:
            pass
        else:
            if publisher not in doi_mapping:
                doi_mapping[publisher]=set()
            doi_mapping[publisher].add(first_part_doi)

    #the mapping was then used to create the file doi_mapping_publisher.txt
    print(doi_mapping)

    return new_publisher_list, dic_theorical_access

def plot_download_proportion_publisher(new_publisher_list, downloaded_dois, DOIs_WOS, dic_theorical_access, top_n=50):
    '''Plot top N publisher by article counts based on the WOS query, and plot the number of dowloaded articles'''
    
    df = pd.DataFrame({"publisher": new_publisher_list, "doi": [normalize_doi(d) for d in DOIs_WOS]})
    #1 if the doi is dowloaded 0 otherwise
    df["downloaded"] = df["doi"].isin(downloaded_dois).astype(int)
    #group by publisher: count total and downloaded
    publisher_stats = df.groupby("publisher").agg(total_articles=("doi", "count"), downloaded_articles=("downloaded", "sum"))
    #keep top N publishers by total number of articles
    top_publishers = publisher_stats.sort_values("total_articles", ascending=False).head(top_n)
    springer_stats = publisher_stats.loc["springer"]
    print(f"Springer: {springer_stats['downloaded_articles']} downloaded out of {springer_stats['total_articles']} total articles")

    plt.figure(figsize=(14, 10))
    x = range(len(top_publishers))
    plt.bar(x, top_publishers["total_articles"], color="skyblue", edgecolor="black", label="Total articles")
    plt.bar(x, top_publishers["downloaded_articles"], color="red", edgecolor="black", label="Downloaded")
    plt.ylabel("Number of articles", fontsize=12)
    plt.xlabel("Publisher", fontsize=12)
    plt.title(f"Number of articles and downloaded articles for top {top_n} publishers", fontsize=14)
    plt.xticks(x, top_publishers.index, rotation=90, fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    file_path = "references_WOS.tsv"
    df = pd.read_csv(file_path, sep="\t").fillna("")
    DOIs_WOS = df["DI"].tolist()
    dates = df["PY"].tolist()
    publishers = [unidecode(j.lower()).strip().replace(" ltd", "") for j in df["PU"].astype(str)]
    journals = [unidecode(j.lower()).strip() for j in df["SO"].astype(str)]

    dowloaded_dois = get_downloaded_dois(file_path)
    new_publisher_list, dic_theorical_access = union_plot_publishers_by_count(publishers, DOIs_WOS, journals, dowloaded_dois)
    #dic_theorical_access --> dictionary of papers for which we could have institutional access
  
    plot_download_proportion_publisher(new_publisher_list, dowloaded_dois, DOIs_WOS, dic_theorical_access)  