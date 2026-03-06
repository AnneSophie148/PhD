from collections import defaultdict, Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.ticker as mticker
import zipfile

import sys 
sys.path.append('../query_V2')
from enriche_graphV2 import load_graph
from utils import normalize_doi

def save_indexes(path, articles_by_doi, ce_by_finding, rc_by_ce, self_ce_ids, article_by_ce, self_citation_by_id):
    """Save RCE indexes into a json"""
    data = {
        "articles_by_doi": articles_by_doi,
        "ce_by_finding": dict(ce_by_finding),
        "rc_by_ce": rc_by_ce,
        "self_ce_ids": self_ce_ids,
        "article_by_ce": article_by_ce,
        "self_citation_by_id": self_citation_by_id
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, sort_keys=True)

    print(f"Indexes saved to {path}")

def load_indexes(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["ce_by_finding"] = defaultdict(list, data["ce_by_finding"])
    data["self_ce_ids"] = data["self_ce_ids"]

    return (data["articles_by_doi"], data["ce_by_finding"], data["rc_by_ce"], data["self_ce_ids"], data["article_by_ce"], data["self_citation_by_id"])


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

def uniform_rheotoric_class(class_list):
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

    #Case 1: only one class
    if len(set_class) == 1:
        final_class = class_list[0]
        return class_list[0]

    
    #Remove neutral/cocoxy
    clean_set = set_class - ignore_classes
    if len(clean_set) == 1:
        final_class = clean_set.pop()
        return final_class

    #Filtered lists
    opinions = [c for c in class_list if c in opinion_class]
    coco = [c for c in class_list if c in coco_opinion_class]
    remaining = [c for c in class_list if c in clean_set]

    #Case 2: at least one opinion
    if opinions:
        maj_opinion, tie_classes = get_majority(opinions)
        if maj_opinion:
            final_class = maj_opinion
            return maj_opinion
        elif tie_classes:
            #print(opinions)
            return "CONTRADICTION"


    #Case 3: coco opinion
    if coco:
        maj_coco, tie_classes = get_majority(coco)
        if maj_coco:
            maj_opinion = maj_coco
            return maj_coco
        elif tie_classes:
            #cocogm over cocores
            final_class = "cocores"
            return final_class

    #Case 4: remaining classes (other)
    maj_remaining = get_majority(remaining)

    if maj_remaining:
        if len(set_class) == 2 and "neutral" in set_class and "cocoxy" in set_class:
            final_class = "neutral"
            return final_class

        return "OTHER"


def default_to_regular(obj):
    """
    Recursively convert all defaultdict objects in a nested structure into regular dicts.
    """
    if isinstance(obj, defaultdict):
        obj = dict(obj)
    if isinstance(obj, dict):
        return {k: default_to_regular(v) for k, v in obj.items()}
    return obj



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

    article_claims_finding = edges.get("article_claims_finding")
    article_produces_citation_event = edges.get("article_produces_citation_event")
    citation_event_refers_to_finding = edges.get("citation_event_refers_to_finding")
    citation_event_has_rhetorical_class = edges.get("citation_event_has_rhetorical_class")
    citation_event_is_self_citation = edges.get("citation_event_is_self_citation")


    index_path = "graph_indexes.json"

    if os.path.exists(index_path):
        print("Loading RCE indexes...")
        articles_by_doi, ce_by_finding, rc_by_ce, self_ce_ids, article_by_ce, self_citation_by_id = load_indexes(index_path)

    else:
        print("Building RCE indexes")

        # DOI → article
        articles_by_doi = {a["doi"]: a for a in articles}
        self_citation_by_id = {sc["id"]: sc.get('self_citation') for sc in self_citations}

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

        save_indexes(index_path, articles_by_doi, ce_by_finding, rc_by_ce, self_ce_ids, article_by_ce, self_citation_by_id)

    findings_citations = {}

    for edge in tqdm(article_claims_finding, desc="Processing articles"):

        doi = edge["from"]
        finding_id = edge["to"]
        dic_rc_to_finding_from_citing_article = {}
        findings_citations[doi] = {"finding_info": articles_by_doi.get(doi), "citing_articles": [], "citing_years": [], "rhetoric_total_per_year": defaultdict(lambda: defaultdict(int))}

        for ce_id in ce_by_finding.get(finding_id, []):
            if ce_id in self_ce_ids:
                sc_id = self_ce_ids.get(ce_id)
                if self_citation_by_id.get(sc_id)=="yes":
                    #skip self citations
                    continue

            rc_class = rc_by_ce.get(ce_id)
            if not rc_class:
                continue

            citing_article = article_by_ce.get(ce_id)

            if citing_article not in dic_rc_to_finding_from_citing_article:
                dic_rc_to_finding_from_citing_article[citing_article]=[]

            dic_rc_to_finding_from_citing_article[citing_article].append(rc_class)


        for citing_article, rc_list in dic_rc_to_finding_from_citing_article.items():
            rc= ""
            if len(rc_list)>1:
                print(f"\nSeveral CE for a same citing article\nRhetorical class list : {rc_list}")
                rc_class = uniform_rheotoric_class(rc_list)
                print(f"Selecting a unique class : {rc_class}")
            else:
                rc_class = rc_list[0]

         
            findings_citations[doi]["citing_articles"].append(citing_article)
            citing_year = articles_by_doi.get(citing_article, {}).get("year")
            findings_citations[doi]["citing_years"].append(citing_year)
            findings_citations[doi]["rhetoric_total_per_year"][citing_year][rc_class] += 1
                
            
    
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
findings_citations_clean = default_to_regular(findings_citations)

with open(save_path_rhetoric, "wb") as f:
    pickle.dump(findings_citations_clean, f)

print(f"Saved findings_citations to {save_path_rhetoric}")