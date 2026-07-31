import json
from tqdm import tqdm
from utils import normalize_doi, get_downloaded_dois
from collections import Counter

def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def get_relation_by_doi(graph_abstract_info):
    dic_relation_by_doi = {}
    for item in graph_abstract_info:
        doi = normalize_doi(item.get("DOI"))
        relations = item.get("contribution_relations", [])
        if not isinstance(relations, list):
            relations = []
        dic_relation_by_doi[doi] = relations

    return dic_relation_by_doi

def get_contribution_relation(cited_doi, dic_relation_by_doi):
    cited_doi = normalize_doi(cited_doi)
    relations_for_doi = dic_relation_by_doi.get(cited_doi, [])
    finding_number = 0
    print("----------------")
    print(cited_doi)
    #print("Relation for this doi : ")
    filtered_relations_for_doi = []

    for relation in relations_for_doi:
        if not isinstance(relation, dict):
            continue

        label_contribution = relation.get("contribution")
        if label_contribution == "contribution":
            finding_number += 1
            #print("--> relation is a contribution !")
            finding_id = f"finding_{cited_doi}_{finding_number}"
            #print("Finding id : ", finding_id)
            finding_relation_not_normalized = {
                "source":relation.get("source"),
                "type":relation.get("type"),
                "target":relation.get("target"),
            }
            finding_relation_normalized = {
                "code_source":relation.get("code_source"),
                "type":relation.get("type"),
                "code_target":relation.get("code_target"),

            }
            #dic_finding = {"id": finding_id, "cited_doi": cited_doi, "relation_before_norm": finding_relation_not_normalized, "relation_norm":finding_relation_normalized}
            dic_finding = {"id": finding_id, "cited_doi": cited_doi, "relation_before_norm": finding_relation_not_normalized, "relation_norm": finding_relation_normalized, "finding_origin": "contribution"}
            filtered_relations_for_doi.append(dic_finding)

    return filtered_relations_for_doi


def recreatre_structure_graph(findings, citing_articles, old_edges, dic_relation_by_doi):
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
    findings_by_doi_and_codes = {}
    generic_findings_by_doi = {}
    counter_citation_passage = 0
    number_multiple_CE = 0

    #From CITATION EDGES to CITATION EVENTS
    for edge in tqdm(old_edges, desc="processing edges"):
        
        cited_doi = edge.get("cited_doi")
        citing_doi = edge.get("citing_doi")
        citation_passages = edge.get("citation_passages", [])
        normalized_cited_doi = normalize_doi(cited_doi)

        if normalized_cited_doi not in seen_findings_doi:
            print("\n--------------------------------------------")
            print("Cited doi:", cited_doi)
            #an article has at least one finding
            filtered_relations_for_doi = get_contribution_relation(cited_doi, dic_relation_by_doi)
            findings_by_doi_and_codes[normalized_cited_doi] = {}

            #print("\nRelations after filtering :")
            #print(filtered_relations_for_doi)
        
            if len(filtered_relations_for_doi) == 0:
                #no contribution found in the abstract
                finding_id = f"finding_{normalized_cited_doi}_0"
                generic_finding_id = finding_id
                generic_findings_by_doi[normalized_cited_doi] = generic_finding_id
                #finding = {"id": finding_id, "cited_doi": cited_doi, "relation_before_norm": "NOT_FOUND", "relation_norm":"NOT_FOUND"}
                finding = {"id": finding_id, "cited_doi": cited_doi, "relation_before_norm": "NOT_FOUND", "relation_norm": "NOT_FOUND", "finding_origin": "generic_no_contribution"}
                
                seen_findings_doi.add(normalized_cited_doi)
                nodes["findings"].append(finding)
                edges["article_claims_finding"].append({"from": cited_doi, "to": finding_id})
            
            else:
                #add as many findings as there are contribution relations
                for finding_item in filtered_relations_for_doi:
                    finding_id = finding_item.get("id")
                    nodes["findings"].append(finding_item)
                    relation_norm = finding_item.get("relation_norm", {})

                    code_source = relation_norm.get("code_source")
                    code_target = relation_norm.get("code_target")

                    if code_source and code_target:
                        code_pair = (str(code_source).strip(), str(code_target).strip())
                        findings_by_doi_and_codes[normalized_cited_doi][code_pair] = finding_id
                        
                    article_finding_edge = {"from": cited_doi, "to": finding_id}
                    edges["article_claims_finding"].append(article_finding_edge)
                    #print(article_finding_edge)
                seen_findings_doi.add(normalized_cited_doi)

        if not citation_passages or citation_passages[0].get("Full-text") in ("NOT_FOUND", "NOT_FOUND_Alvis"):
            #The citation context could not be extracted
            ids["ce"]+=1
            citation_event_id = f"ce_{ids['ce']}"

            #change here to add relation value
            citation_event = {"id": citation_event_id}
            nodes["citation_events"].append(citation_event)            
            # article → citation_event
            edges["article_produces_citation_event"].append({"from": citing_doi, "to": citation_event_id})
            # citation_event → finding
            generic_finding_id = generic_findings_by_doi.get(normalized_cited_doi)
            if generic_finding_id is None:
                generic_finding_id = f"finding_{normalized_cited_doi}_0"
                generic_findings_by_doi[normalized_cited_doi] = generic_finding_id
                #finding = {"id": generic_finding_id, "cited_doi": normalized_cited_doi, "relation_before_norm": "NOT_FOUND", "relation_norm": "NOT_FOUND"}
                finding = {"id": generic_finding_id, "cited_doi": normalized_cited_doi, "relation_before_norm": "NOT_FOUND", "relation_norm": "NOT_FOUND", "finding_origin": "generic_missing_context"}
                nodes["findings"].append(finding)
                edges["article_claims_finding"].append({"from": cited_doi, "to": generic_finding_id})

            edges["citation_event_refers_to_finding"].append({"from": citation_event_id, "to": generic_finding_id})
            continue

        if len(citation_passages)>1:
            number_multiple_CE+=1

        #for each citation passage for a pair citing - cited doi
        for p in citation_passages:

            #addition with the relations from the findings
            passage_relations = p.get("EPPO_entities", [])

            passage_code_pairs = set()

            #ADD WHEN I HAVE THE FINAL OUTPUT WITH CITATION RELATION EXTRACTED
            '''for passage_relation in passage_relations:
                if not isinstance(passage_relation, dict):
                    continue

                passage_code_source = passage_relation.get("code_source")
                passage_code_target = passage_relation.get("code_target")

                if passage_code_source and passage_code_target:
                    code_pair = (str(passage_code_source).strip(), str(passage_code_target).strip())
                    passage_code_pairs.add(code_pair)'''
            
            list_code = []
            for entity_object in passage_relations:
                eppo_code_entity = entity_object[-1]

                if eppo_code_entity:
                    if eppo_code_entity not in list_code:
                        list_code.append(eppo_code_entity)

            if len(list_code) > 1:
                for source in list_code:
                    for target in list_code:
                        if source == target:
                            continue
                        potential_relation = (source, target)
                        passage_code_pairs.add(potential_relation)

            relation_alignment = {}
            matching_finding_ids = set()
            findings_for_doi = findings_by_doi_and_codes.get(normalized_cited_doi, {})

            for relation_number, code_pair in enumerate(sorted(passage_code_pairs), start=1):
                relation_marker = f"r{relation_number}"
                matching_finding_id = findings_for_doi.get(code_pair)

                if matching_finding_id is not None:
                    alignment = "yes"
                    matching_finding_ids.add(matching_finding_id)
                else:
                    alignment = "no"

                relation_alignment[relation_marker] = {
                    "code_source": code_pair[0],
                    "code_target": code_pair[1],
                    "alignment": alignment,
                    "finding_id": matching_finding_id
                }

            counter_citation_passage += 1
            ids["ce"] += 1
            citation_event_id = f"ce_{ids['ce']}"

            citation_event = {
                "id": citation_event_id,
                "one_finding_aligned": "yes" if matching_finding_ids else "no",
                "potential_relations": [
                    {
                        "code_source": code_source,
                        "code_target": code_target
                    }
                    for code_source, code_target in sorted(passage_code_pairs)
                ],
                "relation_alignment": relation_alignment
            }

            if citation_event.get("one_finding_aligned") =="yes":
                print("\nCE : ")
                print(citation_event)

            nodes["citation_events"].append(citation_event)            
            #article → citation_event
            edges["article_produces_citation_event"].append({"from": citing_doi, "to": citation_event_id})
            
            #Updated
            #citation_event → finding
            #edges["citation_event_refers_to_finding"].append({"from": citation_event_id, "to": finding_id})
            if matching_finding_ids:
                for matching_finding_id in matching_finding_ids:
                    edges["citation_event_refers_to_finding"].append({"from": citation_event_id, "to": matching_finding_id})
            else:
                generic_finding_id = generic_findings_by_doi.get(normalized_cited_doi)

                if generic_finding_id is None:
                    generic_finding_id = f"finding_{normalized_cited_doi}_0"
                    generic_findings_by_doi[normalized_cited_doi] = generic_finding_id
                    #finding = {"id": generic_finding_id, "cited_doi": normalized_cited_doi, "relation_before_norm": "NOT_FOUND", "relation_norm": "NOT_FOUND"}
                    finding = {"id": generic_finding_id, "cited_doi": normalized_cited_doi, "relation_before_norm": "NOT_FOUND", "relation_norm": "NOT_FOUND", "finding_origin": "generic_no_aligned_relation"}
                    nodes["findings"].append(finding)
                    edges["article_claims_finding"].append({"from": cited_doi, "to": generic_finding_id})

                edges["citation_event_refers_to_finding"].append({"from": citation_event_id, "to": generic_finding_id})
            

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
    graph_output = "graph_restructured_with_finding.json"

    graph_with_citations_file = "graph_with_EPPO_entities.json"
    info_abstracts_file = "eppo_codes_relation_prediction_Qwen_Qwen3-32B.json"

    #graph_path = "Rhetorical_Citation_Event_Graph.json"
    graph_path = "graph_citations/graph_with_EPPO_entities.json"
    graph_data = load_json(graph_path)
    graph_abstract_info = load_json(info_abstracts_file)

    findings = graph_data.get("findings", [])
    citing_articles = graph_data.get("articles", [])
    edges = graph_data.get("edges", [])
    dic_relation_by_doi = get_relation_by_doi(graph_abstract_info)
    
    print("V1 GRAPH :")
    print_graph_summary_v1(graph_data)

    new_graph = recreatre_structure_graph(findings, citing_articles, edges, dic_relation_by_doi)
    rhetoric_class_proportion = get_rhetoric_class_distribution(new_graph)
    plot_rhetoric_class_counts(rhetoric_class_proportion)

    with open(graph_output, "w", encoding="utf-8") as f:
        json.dump(new_graph, f, indent=2, ensure_ascii=False)

    total_findings_newgraph = len(new_graph.get("nodes", {}).get("findings", []))
    print("\nFinal number of findings:", total_findings_newgraph)

    finding_origin_counts = Counter(finding.get("finding_origin", "unknown") for finding in new_graph.get("nodes", {}).get("findings", []))
    print("\nContribution findings:", finding_origin_counts["contribution"])
    print("Generic findings because no contribution was extracted:", finding_origin_counts["generic_no_contribution"])
    print("Generic findings created because the citation context was unavailable:", finding_origin_counts["generic_missing_context"])
    print("Generic findings created because no relation aligned:", finding_origin_counts["generic_no_aligned_relation"])
    print("Unknown finding origin:", finding_origin_counts["unknown"])
    print("Sum of finding categories:", sum(finding_origin_counts.values()))