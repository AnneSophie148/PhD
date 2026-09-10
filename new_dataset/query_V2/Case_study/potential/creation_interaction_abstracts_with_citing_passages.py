import json
import os
from collections import defaultdict
from tqdm import tqdm


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_doi(doi):
    if doi is None:
        return ""
    return str(doi).strip().lower()


def make_interaction_key(interaction):
    """
    Creates a stable readable key for one biological interaction.
    """
    source_id = interaction.get("source_id", "UNKNOWN_SOURCE")
    target_id = interaction.get("target_id", "UNKNOWN_TARGET")
    rel_type = interaction.get("rel_type", "UNKNOWN_RELTYPE")
    rel_subtype = interaction.get("rel_subtype", "UNKNOWN_RELSUBTYPE")

    return f"{source_id}__{rel_type}__{rel_subtype}__{target_id}"


def build_graph_indexes(graph):
    """
    Builds indexes needed to go from:

    cited DOI / finding article DOI
        -> finding_id
        -> citation_event_id
        -> citing DOI
        -> citing passage

    Assumes graph structure similar to your get_rheotoric_class_by_finding() function.
    """

    nodes = graph.get("nodes", {})
    edges = graph.get("edges", {})

    articles = nodes.get("articles", [])
    in_context_citations = nodes.get("in_context_citations", [])
    sections = nodes.get("sections", [])
    rhetorical_citations = nodes.get("rhetorical_classes", [])

    article_claims_finding = edges.get("article_claims_finding", [])
    citation_event_refers_to_finding = edges.get("citation_event_refers_to_finding", [])
    article_produces_citation_event = edges.get("article_produces_citation_event", [])
    citation_event_is_expressed_as_in_context_citation = edges.get("citation_event_is_expressed_as_in_context_citation", [])
    citation_event_is_located_in_section = edges.get("citation_event_is_located_in_section", [])
    citation_event_has_rhetorical_class = edges.get("citation_event_has_rhetorical_class", [])

    print("NODES:")
    for node_type, node_list in nodes.items():
        print(f"  - {node_type}: {len(node_list)}")

    print("\nEDGES:")
    for edge_type, edge_list in edges.items():
        print(f"  - {edge_type}: {len(edge_list)}")

    print("\nBuilding graph indexes...")

    # DOI -> article metadata
    articles_by_doi = {normalize_doi(article.get("doi")): article for article in articles if article.get("doi")}

    # in-context citation id -> passage text
    full_text_by_ic_id = {ic.get("id"): ic.get("Full-text", "") for ic in in_context_citations}

    decomposed_passage_by_ic_id = {ic.get("id"): ic.get("Decomposed_passage", "") for ic in in_context_citations}

    # section id -> section name
    section_by_id = {section.get("id"): section.get("section", "") for section in sections}

    # rhetorical class id -> rhetorical class
    rhetorical_class_by_id = {rc.get("id"): rc.get("rhetorical_class", "") for rc in rhetorical_citations}

    # cited DOI -> finding ids
    doi_to_finding_ids = defaultdict(list)

    for edge in tqdm(article_claims_finding, desc="Indexing DOI -> findings"):
        cited_doi = normalize_doi(edge.get("from"))
        finding_id = edge.get("to")

        if cited_doi and finding_id:
            doi_to_finding_ids[cited_doi].append(finding_id)

    # finding id -> citation event ids
    ce_by_finding = defaultdict(list)

    for edge in tqdm(citation_event_refers_to_finding, desc="Indexing finding -> citation events"):
        ce_id = edge.get("from")
        finding_id = edge.get("to")

        if ce_id and finding_id:
            ce_by_finding[finding_id].append(ce_id)

    # citation event id -> citing DOI
    citing_doi_by_ce = {}

    for edge in tqdm(article_produces_citation_event, desc="Indexing citation event -> citing DOI"):
        citing_doi = normalize_doi(edge.get("from"))
        ce_id = edge.get("to")

        if ce_id and citing_doi:
            citing_doi_by_ce[ce_id] = citing_doi

    # citation event id -> in-context citation id
    ic_id_by_ce = {
        edge.get("from"): edge.get("to")
        for edge in citation_event_is_expressed_as_in_context_citation
        if edge.get("from") and edge.get("to")
    }

    # citation event id -> section id
    section_id_by_ce = {
        edge.get("from"): edge.get("to")
        for edge in citation_event_is_located_in_section
        if edge.get("from") and edge.get("to")
    }

    # citation event id -> rhetorical class
    rhetorical_class_by_ce = {}

    for edge in tqdm(citation_event_has_rhetorical_class, desc="Indexing citation event -> rhetorical class"):
        ce_id = edge.get("from")
        rc_id = edge.get("to")

        if ce_id and rc_id:
            rhetorical_class_by_ce[ce_id] = rhetorical_class_by_id.get(rc_id, "")

    indexes = {
        "articles_by_doi": articles_by_doi,
        "doi_to_finding_ids": doi_to_finding_ids,
        "ce_by_finding": ce_by_finding,
        "citing_doi_by_ce": citing_doi_by_ce,
        "ic_id_by_ce": ic_id_by_ce,
        "full_text_by_ic_id": full_text_by_ic_id,
        "decomposed_passage_by_ic_id": decomposed_passage_by_ic_id,
        "section_id_by_ce": section_id_by_ce,
        "section_by_id": section_by_id,
        "rhetorical_class_by_ce": rhetorical_class_by_ce,
    }

    print("Graph indexes built.")
    print(f"Indexed cited DOIs with findings: {len(doi_to_finding_ids)}")
    print(f"Indexed findings with citation events: {len(ce_by_finding)}")

    return indexes


def get_citing_data_for_doi(cited_doi, indexes):
    """
    For one DOI that has an interesting abstract, retrieve all citing DOIs
    and all citing passages from the graph.
    """

    cited_doi_norm = normalize_doi(cited_doi)

    doi_to_finding_ids = indexes["doi_to_finding_ids"]
    ce_by_finding = indexes["ce_by_finding"]
    citing_doi_by_ce = indexes["citing_doi_by_ce"]
    articles_by_doi = indexes["articles_by_doi"]
    ic_id_by_ce = indexes["ic_id_by_ce"]
    full_text_by_ic_id = indexes["full_text_by_ic_id"]
    decomposed_passage_by_ic_id = indexes["decomposed_passage_by_ic_id"]
    section_id_by_ce = indexes["section_id_by_ce"]
    section_by_id = indexes["section_by_id"]
    rhetorical_class_by_ce = indexes["rhetorical_class_by_ce"]

    finding_ids = doi_to_finding_ids.get(cited_doi_norm, [])

    citing_data = defaultdict(lambda: {
        "citing_doi": "",
        "citing_article_metadata": {},
        "citation_events": []
    })

    for finding_id in finding_ids:

        ce_ids = ce_by_finding.get(finding_id, [])

        for ce_id in ce_ids:

            citing_doi = citing_doi_by_ce.get(ce_id)

            if not citing_doi:
                continue

            ic_id = ic_id_by_ce.get(ce_id)
            section_id = section_id_by_ce.get(ce_id)

            passage = full_text_by_ic_id.get(ic_id, "")
            decomposed_passage = decomposed_passage_by_ic_id.get(ic_id, "")
            section = section_by_id.get(section_id, "")
            rhetorical_class = rhetorical_class_by_ce.get(ce_id, "")

            citing_data[citing_doi]["citing_doi"] = citing_doi
            citing_data[citing_doi]["citing_article_metadata"] = articles_by_doi.get(
                citing_doi, {}
            )

            citing_data[citing_doi]["citation_events"].append({
                "citation_event_id": ce_id,
                "finding_id": finding_id,
                "in_context_citation_id": ic_id,
                "citing_passage": passage,
                "decomposed_citing_passage": decomposed_passage,
                "section": section,
                "rhetorical_class": rhetorical_class
            })

    return {
        "cited_doi": cited_doi_norm,
        "finding_ids": finding_ids,
        "n_findings": len(finding_ids),
        "n_citing_dois": len(citing_data),
        "citing_dois": list(citing_data.values())
    }



def build_interaction_citation_file(interesting_abstracts_path, graph_path, output_path, save_every=10):
    """
    Loads Files_for_analysis.json and graph, then creates:

    {
        interaction_key: {
            "matched_interaction": {...},
            "abstracts_with_this_interaction": [...],
            "cited_dois": [...]
        }
    }
    """

    print(f"Loading interesting abstracts from: {interesting_abstracts_path}")
    interesting_abstracts = load_json(interesting_abstracts_path)

    print(f"Loading graph from: {graph_path}")
    graph = load_json(graph_path)

    indexes = build_graph_indexes(graph)

    output = {}

    processed_matches = 0

    for cited_doi, abstract_records in tqdm(interesting_abstracts.items(), desc="Processing interesting abstract DOIs"):
        cited_doi_norm = normalize_doi(cited_doi)
        citing_data_for_doi = get_citing_data_for_doi(cited_doi_norm, indexes)

        for record in abstract_records:
            interaction = record.get("matched_interaction", {})
            interaction_key = make_interaction_key(interaction)

            if interaction_key not in output:
                output[interaction_key] = {
                    "interaction_key": interaction_key,
                    "matched_interaction": interaction,
                    "abstracts_with_this_interaction": [],
                    "cited_dois": []
                }

            output[interaction_key]["abstracts_with_this_interaction"].append({
                "doi": cited_doi_norm,
                "title": record.get("title", ""),
                "abstract": record.get("abstract", ""),
                "source_id": record.get("source_id", ""),
                "source_preferred_name": record.get("source_preferred_name", ""),
                "source_variants": record.get("source_variants", []),
                "target_id": record.get("target_id", ""),
                "target_preferred_name": record.get("target_preferred_name", ""),
                "target_variants": record.get("target_variants", []),
                "rel_type": record.get("rel_type", ""),
                "rel_subtype": record.get("rel_subtype", "")
            })

            output[interaction_key]["cited_dois"].append(citing_data_for_doi)
            processed_matches += 1

            if processed_matches % save_every == 0:
                save_json(output, output_path)
                tqdm.write(f"Saved after {processed_matches} interaction/abstract matches: {output_path}")

    save_json(output, output_path)

    print("\nDone.")
    print(f"Total interactions in output: {len(output)}")
    print(f"Total interaction/abstract matches processed: {processed_matches}")
    print(f"Saved to: {output_path}")

    return output


interesting_abstracts_path = "intersting_abstract/Files_for_analysis.json"
graph_path = "restructured_graph.json"
output_path = "interaction_abstracts_with_citing_passages.json"

interaction_citations = build_interaction_citation_file(interesting_abstracts_path=interesting_abstracts_path, graph_path=graph_path, output_path=output_path, save_every=10)