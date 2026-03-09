from collections import defaultdict, Counter
from tqdm import tqdm
import json
import pickle
import os
import zipfile

import sys 
sys.path.append('../query_V2')
from enriche_graphV2 import load_graph
from utils import normalize_doi

def save_indexes(path, articles_by_doi, ce_by_finding, rc_by_ce, self_ce_ids, article_by_ce, self_citation_by_id, citation_passage_by_ce, ic_by_id):
    """Saves data in dic format accessible by id"""

    data = {
        "articles_by_doi": articles_by_doi,
        "ce_by_finding": dict(ce_by_finding),
        "rc_by_ce": rc_by_ce,
        "self_ce_ids": self_ce_ids,
        "article_by_ce": article_by_ce,
        "self_citation_by_id": self_citation_by_id,
        "citation_passage_by_ce": citation_passage_by_ce,
        "ic_by_id": ic_by_id
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, sort_keys=True)

def load_indexes(path):
    """Load dic by id data"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["ce_by_finding"] = defaultdict(list, data["ce_by_finding"])
    return (data["articles_by_doi"], data["ce_by_finding"], data["rc_by_ce"], data["self_ce_ids"], data["article_by_ce"], data["self_citation_by_id"], data["citation_passage_by_ce"], data["ic_by_id"])



def get_majority(classes):
    """Return majority class if one exists, otherwise (None, top_classes) if tie."""
    if not classes:
        return None, None
    
    counts = Counter(classes)
    most_common = counts.most_common()
    top_count = most_common[0][1]
    top_classes = [cls for cls, cnt in most_common if cnt == top_count]
    
    if len(top_classes) == 1:
        return top_classes[0], None
    else:
        return None, top_classes

def uniform_rheotoric_class(class_list, passages):
    '''Chose one rhetoric class given a list of rc classes from a same citing article refering to the same finding
    Choses according to rules:
    - opinion first (weakness, support) --> take the maj, if equal returns "CONTRADICTION"
    - coco second : priority to cocores
    - neutral and cocoxy last : try to remove them to see if then there is only one other rc class left
    - last case: return "OTHER"'''

    final_class = ""
    opinion_class = {"weakness", "support"}
    coco_opinion_class = {"cocores", "cocogm"}
    ignore_classes = {"neutral", "cocoxy"}
    final_passages = []

    set_class = set(class_list)

    # Case 1: only one class
    if len(set_class) == 1:
        final_class = class_list[0]
        final_passages = [p for p, rc in zip(passages, class_list) if rc == final_class]
        return class_list[0], final_passages

    
    # Remove neutral/cocoxy
    clean_set = set_class - ignore_classes
    if len(clean_set) == 1:
        final_class = clean_set.pop()
        final_passages = [p for p, rc in zip(passages, class_list) if rc == final_class]
        return final_class, final_passages

    # Filtered lists
    opinions = [c for c in class_list if c in opinion_class]
    coco = [c for c in class_list if c in coco_opinion_class]
    remaining = [c for c in class_list if c in clean_set]

    # Case 2: at least one opinion
    if opinions:
        maj_opinion, tie_classes = get_majority(opinions)
        if maj_opinion:
            final_class = maj_opinion
            final_passages = [p for p, rc in zip(passages, class_list) if rc == final_class]
            return maj_opinion, final_passages
        elif tie_classes:
            #print(opinions)
            final_passages = [p for p, rc in zip(passages, class_list) if rc in {"weakness", "support"}]
            return "CONTRADICTION", final_passages


    # Case 3: coco opinion
    if coco:
        maj_coco, tie_classes = get_majority(coco)
        if maj_coco:
            maj_opinion = maj_coco
            final_passages = [p for p, rc in zip(passages, class_list) if rc == maj_opinion]
            return maj_coco, final_passages
        elif tie_classes:
            #cocogm over cocores
            final_class = "cocores"
            final_passages = [p for p, rc in zip(passages, class_list) if rc == final_class]
            return final_class, final_passages

    # Case 4: remaining classes (other)
    maj_remaining = get_majority(remaining)

    if maj_remaining:
        if len(set_class) == 2 and "neutral" in set_class and "cocoxy" in set_class:
            final_class = "neutral"
            final_passages = [p for p, rc in zip(passages, class_list) if rc == final_class]
            return final_class, final_passages
        final_passages = [p for p, rc in zip(passages, class_list) if rc == final_class]
        return "OTHER", final_passages


def get_rheotoric_class_by_finding(graph):
    """
    Returns:
    {
        finding_doi: {
            "finding_info": {...},
            "citing_articles": [...],
            "citing_years": [...],
            "citing_years_count": {year: count},
            "citing_years_count_self": {year: count},
            "rhetoric_by_year": {year: {class: count}},
            "rhetoric_total_per_year": {year: total}
        }
    }
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    citation_events = nodes.get("citation_events")
    findings = nodes.get("findings")
    articles = nodes.get("articles")
    self_citations = nodes.get("self-citations")
    rhetorical_citations = nodes.get("rhetorical_classes")
    in_context_citations = nodes.get("in_context_citations")

    article_claims_finding = edges.get("article_claims_finding")
    article_produces_citation_event = edges.get("article_produces_citation_event")
    citation_event_refers_to_finding = edges.get("citation_event_refers_to_finding")
    citation_event_has_rhetorical_class = edges.get("citation_event_has_rhetorical_class")
    citation_event_is_self_citation = edges.get("citation_event_is_self_citation")
    citation_event_is_expressed_as_in_context_citation = edges.get("citation_event_is_expressed_as_in_context_citation")


    index_path = "graph_indexes.json"

    if os.path.exists(index_path):
        print("Loading RCE indexes...")
        articles_by_doi, ce_by_finding, rc_by_ce, self_ce_ids, article_by_ce, self_citation_by_id, citation_passage_by_ce, ic_by_id = load_indexes(index_path)

    else:
        print("Building RCE indexes")

        # DOI → article
        articles_by_doi = {a["doi"]: a for a in articles}
        self_citation_by_id = {sc["id"]: sc.get('self_citation') for sc in self_citations}
        ic_by_id = {ic["id"]: ic.get('Full-text') for ic in in_context_citations}

        # finding_id → citation events
        ce_by_finding = defaultdict(list)
        for e in citation_event_refers_to_finding:
            ce_by_finding[e["to"]].append(e["from"])

        # ce_id → rhetorical class
        rc_by_ce = {}
        
        for e in tqdm(citation_event_has_rhetorical_class, desc="Processing ce_id → rhetorical class"):
            rc_id = e["to"]
            for rc in rhetorical_citations:
                if rc["id"] == rc_id:
                    rc_by_ce[e["from"]] = rc["rhetorical_class"]
                    break

        # ce_id → self citation
        self_ce_ids = {e["from"]: e["to"] for e in citation_event_is_self_citation}

        # ce_id → citing article
        article_by_ce = {e["to"]: e["from"] for e in article_produces_citation_event}
        citation_passage_by_ce = {e["from"]: e["to"] for e in citation_event_is_expressed_as_in_context_citation}

        save_indexes(index_path, articles_by_doi, ce_by_finding, rc_by_ce, self_ce_ids, article_by_ce, self_citation_by_id, citation_passage_by_ce, ic_by_id)

    findings_citations = {}

    for edge in tqdm(article_claims_finding):
        fdoi = edge["from"]
        finding_id = edge["to"]
        rhetoric_by_year = defaultdict(lambda: defaultdict(list))
        citing_articles = []
        total_citing_dois = set()
        citing_years = []
        self_citing_articles = []
        self_citing_years = []

        # group CE by citing article
        article_ce_map = defaultdict(list)
        
        for ce_id in ce_by_finding.get(finding_id, []):
            citing_article = article_by_ce.get(ce_id)
            total_citing_dois.add(citing_article)
            article_ce_map[citing_article].append(ce_id)

        # process each citing article
        for citing_article, ce_list in article_ce_map.items():
            article_meta = articles_by_doi.get(citing_article, {})
            citing_year = article_meta.get("year")
            rc_list = []
            passage_list = []

            for ce_id in ce_list:

                rc_class = rc_by_ce.get(ce_id, "unknown")
                #when the citing article could not be dowloaded we attribut rc to the value unknown
                passage = ic_by_id.get(citation_passage_by_ce.get(ce_id), "NOT_FOUND")

                if "NOT_FOUND" in passage:
                    rc = "unknown"

                rc_list.append(rc_class)
                passage_list.append(passage)

                if ce_id in self_ce_ids:
                    sc_id = self_ce_ids.get(ce_id)
                    if self_citation_by_id.get(sc_id) == "yes":
                        self_citing_articles.append(citing_article)
                        self_citing_years.append(citing_year)
                        #dont append the self citation event in the rhetoric counts
                        continue

            if len(rc_list) > 1:
                #more than one passage for a same citing paper --> chose one rc
                uniformed_class, corresponding_passage = uniform_rheotoric_class(rc_list, passage_list)
            else:
                uniformed_class = rc_list[0]
                corresponding_passage = passage_list[0]
            
            if citing_article not in citing_articles:
                citing_articles.append(citing_article)
            citing_years.append(citing_year)

            rhetoric_by_year[citing_year][uniformed_class].append({"doi": citing_article, "passage": corresponding_passage})
            
        # build final structure
        rhetoric_by_year_structured = {}
        for year in sorted(rhetoric_by_year):
            rhetoric_by_year_structured[year] = {}
            for cls in sorted(rhetoric_by_year[year]):
                articles_list = rhetoric_by_year[year][cls]
                rhetoric_by_year_structured[year][cls] = {
                    "count": len(articles_list),
                    "passages": [a["passage"] for a in articles_list],
                    "citing_dois": [a["doi"] for a in articles_list]
                }
                print(rhetoric_by_year_structured[year][cls])


        findings_citations[fdoi] = {
            "finding_info": articles_by_doi.get(fdoi),
            "citing_articles": citing_articles,
            "citing_years": sorted(set(citing_years)),
            "citing_years_count": dict(Counter(citing_years)),
            "citing_years_count_self": dict(Counter(self_citing_years)),
            "rhetoric_by_year": rhetoric_by_year_structured,
            "rhetoric_total_per_year": {y: sum(cls_data["count"] for cls_data in year_dict.values()) for y, year_dict in rhetoric_by_year_structured.items()}
        }
                
            
    
    for fdoi, info in list(findings_citations.items())[:15]:
        print("\nFinding:", fdoi)
        print("Finding year:", info["finding_info"]["year"] )
        print("Cited by years:", info["citing_years"])
        print("Number of citing articles:", len(info["citing_articles"]))
        print("Sample citing article:", info["citing_articles"] if info["citing_articles"] else None)
        print("rhetoric_total_per_year :", info["rhetoric_total_per_year"])

    return findings_citations


graph_path = "../query_V2/Rhetorical_Citation_Graph.zip"

# Load JSON from zip
with zipfile.ZipFile(graph_path) as z:
    with z.open("restructured_graph.json") as f:
        graph_data = json.load(f)

#pkl file one rc per citation event only without self citation for now
save_path_rhetoric = "RC_ONE_CE_per_pair_findings_dic_without_citing_references.pkl"

findings_citations = get_rheotoric_class_by_finding(graph_data)

with open(save_path_rhetoric, "wb") as f:
    pickle.dump(findings_citations, f)

print(f"Saved findings_citations to {save_path_rhetoric}")