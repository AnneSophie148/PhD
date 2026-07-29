def plot_eppo_graph_venn(unique_ref, set_ref_in_query_art, set_ref_in_citing_art, total_graph_refs=87974, title="Common references EPPO - Vector Graph"):

    # EPPO set
    eppo_set = {normalize_doi(ref) for ref in unique_ref if ref}

    # References from graph
    graph_ref_set = (set_ref_in_query_art | set_ref_in_citing_art)

    # Counts
    common_refs = len(graph_ref_set)
    eppo_only = len(eppo_set) - common_refs
    graph_only = total_graph_refs - common_refs

    print("EPPO references:", len(eppo_set))
    print("References found in graph:", common_refs)
    print("Missing EPPO references:", eppo_only)
    print("Graph-only references:", graph_only)

    plt.figure(figsize=(8, 8))

    v = venn2(subsets=(eppo_only, graph_only, common_refs), set_labels=("EPPO references", "Vector graph articles"))

    if v.set_labels:
        for label in v.set_labels:
            label.set_fontsize(16)

    if v.subset_labels:
        for label in v.subset_labels:
            if label:
                label.set_fontsize(14)

    plt.title(title, fontsize=18)
    plt.show()
    return graph_ref_set

def analyse_repartition_data_graph(citing_article_set, ref_cache, unique_ref, dois_wos):
    set_ref_in_query_art = set()
    set_ref_in_citing_art = set()

    for key_set, items in ref_cache["Found_in_VectorDataset_Graph"].items():
        if key_set == "Query_WOS":
            for norm_ref, item in items.items():
                norm_doi = normalize_doi(item.get("doi"))
                set_ref_in_query_art.add(norm_doi)
        else:
            for norm_ref, item in items.items():
                norm_doi = normalize_doi(item.get("doi"))
                set_ref_in_citing_art.add(norm_doi)

    #because i was stopping if the reference was in the query articles, i need to check if it's also a citing article
    for norm_doi in set_ref_in_query_art:
        if norm_doi in citing_article_set:
            set_ref_in_citing_art.add(norm_doi)

    print(f"Number of EPPO references that are in the Query article set : {len(set_ref_in_query_art)}")
    print(f"Number of EPPO references that are in the Citing article set : {len(set_ref_in_citing_art)}")

    total_number_references_eppo = len(unique_ref)
    number_missing_references_graph = len(unique_ref)-(len(set_ref_in_citing_art)+len(set_ref_in_query_art))
    print(f"Total number of references in EPPO after deduplication : {total_number_references_eppo}")
    print(f"Missing references from graph : ",number_missing_references_graph )

    graph_ref_set = plot_eppo_graph_venn(unique_ref, set_ref_in_query_art, set_ref_in_citing_art)
    return graph_ref_set, set_ref_in_query_art, set_ref_in_citing_art