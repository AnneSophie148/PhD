from enriche_graphV2 import load_graph
import json
from tqdm import tqdm
from utils import normalize_doi, get_downloaded_dois

def recreatre_structure_graph(findings, citing_articles, old_edges):
    '''Function to restructure the graph for modeling Citation Event from the Enriched graph'''

    nodes = {"articles": [], "findings": [],
        "citation_events": [], "in_context_citations": [], "self-citations": [], "rhetorical_classes": [], "sections": []}

    edges = {"article_claims_finding": [], "article_produces_citation_event": [], "citation_event_refers_to_finding": [],
        "citation_event_has_rhetorical_class": [], "citation_event_is_self_citation": [], "citation_event_is_expressed_as_in_context_citation": [],
        "citation_event_is_located_in_section": []}

    ids = {"ce": 0, "ic": 0, "sc": 0, "rc" :0, "st":0}

    all_articles = findings + citing_articles

    citing_articles_by_doi = {}
    existing_dois = set()

    for article in tqdm(all_articles, desc="Adding citing articles in articles"):
        dic = {"doi": article.get("doi"),
            "year": article.get("year"),
            "author": article.get("author"),
            "title": article.get("title")}

        doi = dic["doi"]
        if doi not in existing_dois:
            nodes["articles"].append(dic)
            existing_dois.add(doi)

        citing_articles_by_doi[doi] = article

    seen_findings_doi = set()
    counter_citation_passage = 0
    number_multiple_CE = 0

    #From CITATION EDGES to CITATION EVENTS
    for edge in tqdm(old_edges, desc="processing edges"):
        cited_doi = edge.get("cited_doi")
        citing_doi = edge.get("citing_doi")
        citation_passages = edge.get("citation_passages", [])

        finding_id = f"finding_{cited_doi}"
        finding = {"id": finding_id, "cited_doi": cited_doi}

        if normalize_doi(cited_doi) not in seen_findings_doi:
            seen_findings_doi.add(normalize_doi(cited_doi))
            nodes["findings"].append(finding)

            #in the current graph version there is only one finding per article, if there was more this line should be moved
            edges["article_claims_finding"].append({"from": cited_doi, "to": finding_id})

            

        if not citation_passages or edge["citation_passages"][0]["Full-text"] in ("NOT_FOUND", "NOT_FOUND_Alvis"):
            ids["ce"]+=1
            citation_event_id = f"ce_{ids['ce']}"
            citation_event = {"id": citation_event_id}
            nodes["citation_events"].append(citation_event)            
            # article → citation_event
            edges["article_produces_citation_event"].append({"from": citing_doi, "to": citation_event_id})
            # citation_event → finding
            edges["citation_event_refers_to_finding"].append({"from": citation_event_id, "to": finding_id})
            continue

        if len(citation_passages)>1:
            number_multiple_CE+=1

        #for each citation passage for a pair citing - cited doi
        for p in citation_passages:
            counter_citation_passage +=1
            #one citation event per passage
            ids["ce"]+=1
            citation_event_id = f"ce_{ids['ce']}"
            citation_event = {"id": citation_event_id}

            nodes["citation_events"].append(citation_event)            
            #article → citation_event
            edges["article_produces_citation_event"].append({"from": citing_doi, "to": citation_event_id})
            #citation_event → finding
            edges["citation_event_refers_to_finding"].append({"from": citation_event_id, "to": finding_id})
            #article → finding (claims)
            

            section = p.get("section")
            text = p.get("Full-text")
            decomposed_passage = p.get("Decomposed_passage")
            rhetorical_class = p.get("predicted_rhetorical_class")

            #in-context citation
            ids["ic"]+=1
            ic_id = f"ic_{ids['ic']}"
            in_context_citation = {"id": ic_id, "Full-text": text, "Decomposed_passage": decomposed_passage}
            nodes["in_context_citations"].append(in_context_citation)
            edges["citation_event_is_expressed_as_in_context_citation"].append({"from": citation_event_id, "to": ic_id})

            ids["st"]+=1
            st_id = f"st_{ids['st']}"
            #section
            if not section:
                section = "unknown"
            nodes["sections"].append({"id": st_id, "section": section})
            edges["citation_event_is_located_in_section"].append({"from": citation_event_id, "to": st_id})

            ids["rc"]+=1
            rc_id = f"rc_{ids['rc']}"
            #rhetoric class
            if not rhetorical_class:
                rhetorical_class = "unknown"
            nodes["rhetorical_classes"].append({"id": rc_id, "rhetorical_class" : rhetorical_class})
            edges["citation_event_has_rhetorical_class"].append({"from": citation_event_id, "to": rc_id})

            #self-citation --> should be moved up if we won't to add this metadata for each citation event even if it's not characterized with the rhetorical class
            self_citation = citing_articles_by_doi.get(citing_doi, {}).get("self-citation", "unknown")
            ids["sc"]+=1
            sc_id = f"sc_{ids['sc']}"
            nodes["self-citations"].append({"id": sc_id, "self_citation":self_citation})
            edges["citation_event_is_self_citation"].append({"from": citation_event_id, "to": sc_id})

    print(f"Number of citation passage : {counter_citation_passage}")
    print("Number of multiple CE : ", number_multiple_CE)

    

    return {"nodes": nodes, "edges": edges}


def print_graph_summary_v1(graph):
    '''Plot a summary of the previous graph version'''
    print("\n===== V1 GRAPH SUMMARY =====\n")

    print(f"  - articles: {len(graph['articles'])}")
    print(f"  - findings: {len(graph['findings'])}")
    print(f"  - edges: {len(graph['edges'])}")
            

def plot_rhetoric_class_counts(rhetoric_class_proportion):
    '''Plot rhetorical class counts'''
    import matplotlib.pyplot as plt

    RHETORIC_COLORS = {
        "support": "#2ca02c", "basis": "#1f77b4", "usage": "#9467bd",
        "motivation": "#ff7f0e", "future": "#17becf", "weakness": "#d62728",
        "neutral": "#7f7f7f", "similar": "#8c564b", "cocores": "#e377c2",
        "cocogm": "#bcbd22", "cocoxy": "#aec7e8", "unknown": "#000000"
    }

    sorted_items = sorted(rhetoric_class_proportion.items(), key=lambda x: x[1], reverse=True)
    classes, counts = zip(*sorted_items)
    colors = [RHETORIC_COLORS.get(c, "#cccccc") for c in classes]

    plt.figure(figsize=(11, 7))
    bars = plt.bar(classes, counts, color=colors)
    plt.ylabel("Number of citation events", fontsize=22, labelpad=10)
    plt.xlabel("Rhetorical class", fontsize=22, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks(fontsize=16)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{int(height)}", ha="center", va="bottom", fontsize=14)

    plt.tight_layout()
    plt.show()



def get_rhetoric_class_distribution(graph):
    '''Get rhetorical class distribution from the RCE graph'''
    from collections import Counter
    rc_nodes = graph["nodes"].get("rhetorical_classes", [])
    labels = [n["rhetorical_class"] for n in rc_nodes]
    
    return dict(Counter(labels))


if __name__ == "__main__":
    graph_output = "graph_with_classified_citations_V1_BIOBERT.json"
    graph_path = "Rhetorical_Citation_Event_Graph.json"
    graph_data = load_graph(graph_path)
    findings = graph_data.get("findings", [])
    citing_articles = graph_data.get("articles", [])
    edges = graph_data.get("edges", [])
    
    print("V1 GRAPH :")
    print_graph_summary_v1(graph_data)

    new_graph = recreatre_structure_graph(findings, citing_articles, edges)
    rhetoric_class_proportion = get_rhetoric_class_distribution(new_graph)
    plot_rhetoric_class_counts(rhetoric_class_proportion)

    with open(graph_output, "w", encoding="utf-8") as f:
        json.dump(new_graph, f, indent=2, ensure_ascii=False)