import numpy as np
import pandas as pd
import re
import json
import argparse
import string
import glob
import random
from collections import Counter
import os

def print_sequence_samples_by_class(dic_classes, n=20, random_sample=False, seed=42):
    if random_sample:
        random.seed(seed)

    for classname, data in dic_classes.items():
        sequences = data["sequence"]
        sections = data["section"]
        indices = list(range(len(sequences)))

        if random_sample:
            sampled_indices = random.sample(indices, min(n, len(indices)))
        else:
            sampled_indices = indices[:n]

        for sample_number, idx in enumerate(sampled_indices, start=1):
            '''print(f"\n--- Sample {sample_number} ---")
            print(f"Section: {sections[idx]}")
            print(sequences[idx])'''
            pass

def normalise_section(section):
    if section is None:
        return "Missing"

    section = str(section).lower().strip()
    section = section.translate(str.maketrans("", "", string.punctuation))
    section = re.sub(r"\s+", " ", section).strip()

    if section == "":
        return "Missing"
        
    if "abstract" in section:
        return "Abstract"

    if "introduction" in section or "motivation" in section:
        return "Introduction"

    if "related work" in section or "other work" in section or "background" in section or "literature" in section or "previous work" in section or "prior work" in section or "past work" in section or "history" in section:
        return "Related work"

    if "conclusion" in section or "concluding" in section:
        return "Conclusion"

    if "future work" in section or "future direction" in section or "further work" in section or "perspective" in section in section or "further" in section:
        return "Future work"

    if "discussion" in section:
        return "Discussion"

    if "method" in section or "methodology" in section or "materials and methods" in section or "approach" in section or "model" in section or "algorithm" in section or "implementation" in section or "framework" in section or "data" in section or "corpus" in section:
        return "Method"

    if "experiment" in section or "experimental" in section or "evaluation" in section or "benchmark" in section or "setup" in section:
        return "Experiment"

    if "result" in section or "analysis" in section or "finding" in section:
        return "Results"

    if str(section).strip().isdigit():
        if str(section).strip() == "0":
            return "Introduction"
        else:
            #print("\nSection not mapped:")
            #print(section)
            pass

    return "Missing"
   
def load_scicite_coco_augmentation(scicite_csv_path, mapping_section, use_position_embedding=False, max_sections=7):
    """
    Load Cohan_compare_contrast.csv and convert SciCite
    """
    if not os.path.exists(scicite_csv_path):
        raise FileNotFoundError(f"SciCite augmentation CSV not found: {scicite_csv_path}")

    df_scicite = pd.read_csv(scicite_csv_path)

    required_columns = ["sequence_masked", "section"]
    missing_columns = [col for col in required_columns if col not in df_scicite.columns]
    if missing_columns:
        raise ValueError(f"SciCite CSV is missing required columns: {missing_columns}. Available columns: {list(df_scicite.columns)}")

    df_scicite = df_scicite.dropna(subset=["sequence_masked"]).copy()
    df_scicite["sequence_masked"] = df_scicite["sequence_masked"].astype(str)
    df_scicite = df_scicite[df_scicite["sequence_masked"].str.strip() != ""]

    scicite_x = df_scicite["sequence_masked"].tolist()
    scicite_y = ["compareorcontrast"] * len(scicite_x)

    scicite_mapped_sections = [normalise_section(section) for section in df_scicite["section"].tolist()]

    if use_position_embedding:
        scicite_section_features = build_scicite_position_embeddings(scicite_mapped_sections, max_sections=max_sections)
    else:
        section_to_idx = {section_name: idx for idx, section_name in enumerate(mapping_section)}
        missing_idx = section_to_idx["Missing"]
        scicite_section_features = [section_to_idx.get(section, missing_idx) for section in scicite_mapped_sections]

    print("\n" + "=" * 100)
    print("SCICITE AUGMENTATION")
    print("=" * 100)
    print("CSV:", scicite_csv_path)
    print("Number of added SciCite examples:", len(scicite_x))
    print("Assigned label: compareorcontrast")
    print("Normalised section distribution:")
    print(pd.Series(scicite_mapped_sections).value_counts())
    print("=" * 100 + "\n")

    return scicite_x, scicite_y, scicite_mapped_sections, scicite_section_features

def protect_abbreviations(text):
    text = text.replace("et al. ", "et al<DOT> ")
    return text

def restore_abbreviations(text):
    return text.replace("et al<DOT>", "et al.")

def norm_space(x):
    return re.sub(r"\s+", " ", str(x)).strip()

def find_citance_jurgens(citing_string, ctx_split):
    citseg_norm = norm_space(citing_string)
    citseg_norm2 = citing_string.replace(" ,", ",").replace("( ", "(").replace(" )", ")").replace("  ", " ").replace("[ ", "[").replace(" ]", "]")

    citance = None
    citance_with_citseg_masked = None
    candidate_citances = []

    # Extract first author from CITSEG
    match_author = re.search(r"\b[A-Z][A-Za-z]+(?:[-'][A-Z]?[A-Za-z]+)?\b", citing_string)
    author_name = match_author.group(0) if match_author else ""

    # Extract year from CITSEG
    match_year = re.search(r"\b[0-9]{4}[a-z]?\b", citing_string, flags=re.IGNORECASE)
    citseg_year = match_year.group(0) if match_year else ""

    for sentence in ctx_split:
        sentence_norm = norm_space(sentence)

        # 1. Exact CITSEG match
        if citseg_norm in sentence_norm:
            citance_with_citseg_masked = sentence_norm.replace(citseg_norm, "CITSEG", 1)
            return sentence_norm, citance_with_citseg_masked

        elif citseg_norm2 in sentence_norm:
            citance_with_citseg_masked = sentence_norm.replace(citseg_norm2, "CITSEG", 1)
            return sentence_norm, citance_with_citseg_masked

        # 2. Candidate sentence contains author name
        elif author_name and author_name in sentence_norm:
            candidate_citances.append(sentence_norm)

    # 3. Use author + year only if there is one unambiguous candidate sentence
    if len(candidate_citances) == 1 and author_name and citseg_year:
        citance = candidate_citances[0]

        pattern = rf"\b{re.escape(author_name)}\b.*?\b{re.escape(citseg_year)}\b"
        match_citseg_span = re.search(pattern, citance)

        if match_citseg_span:
            start, end = match_citseg_span.span()

            citance_with_citseg_masked = (citance[:start] + "CITSEG" + citance[end:])

            return citance, citance_with_citseg_masked

    return None, None

def mask_citseg_in_known_citance(citing_string, citance, debug=False):
    """
    For Teufel data:
    - citance is already known
    - replace the citation marker with CITSEG
    - handles cases where Teufel removed the year, e.g.
      CITSEG = "Moortgat ( 0000 )"
      citance = "Moortgat 's M-System ..."
    """

    citance_norm = norm_space(citance)
    citseg_norm = norm_space(citing_string)

    # Basic CITSEG variants
    citseg_variants = [citseg_norm, citing_string.replace(" ,", ",").replace("( ", "(").replace(" )", ")").replace("  ", " ").replace("[ ", "[").replace(" ]", "]"),
        citing_string.replace(",", " ,"),
        citing_string.replace("(", "( ").replace(")", " )"),
    ]

    #Try direct replacement first
    for variant in citseg_variants:
        variant = norm_space(variant)

        if variant and variant in citance_norm:
            masked = citance_norm.replace(variant, "CITSEG", 1)
            return norm_space(masked)

    #2. Extract first author
    match_author = re.search(r"\b[A-Z][A-Za-z]+(?:[-'][A-Z]?[A-Za-z]+)?\b", citing_string)
    author_name = match_author.group(0) if match_author else ""

    #3. Extract year, including 0000
    match_year = re.search(r"\b[0-9]{4}[a-z]?\b", citing_string, flags=re.IGNORECASE)
    citseg_year = match_year.group(0) if match_year else ""

    #4. Try author + year if both are available and present in the citance
    if author_name and citseg_year:
        pattern_author_year = (
            rf"[\(\[]?\s*"
            rf"\b{re.escape(author_name)}\b"
            rf".{{0,80}}?"
            rf"\b{re.escape(citseg_year)}\b"
            rf"\s*[\)\]]?"
        )

        match_citseg_span = re.search(pattern_author_year, citance_norm, flags=re.IGNORECASE)

        if match_citseg_span:
            start, end = match_citseg_span.span()
            masked = citance_norm[:start] + "CITSEG" + citance_norm[end:]
            return norm_space(masked)

    # 5. Teufel fallback: author possessive, e.g. "Moortgat 's" or "Moortgat's"
    if author_name:
        pattern_possessive = rf"\b{re.escape(author_name)}\b\s*'\s*s\b"

        if re.search(pattern_possessive, citance_norm, flags=re.IGNORECASE):
            masked = re.sub(
                pattern_possessive,
                "CITSEG 's",
                citance_norm,
                count=1,
                flags=re.IGNORECASE
            )
            return norm_space(masked)

    # 6. Last fallback: author only
    if author_name:
        pattern_author_only = rf"\b{re.escape(author_name)}\b"

        if re.search(pattern_author_only, citance_norm, flags=re.IGNORECASE):
            masked = re.sub(
                pattern_author_only,
                "CITSEG",
                citance_norm,
                count=1,
                flags=re.IGNORECASE
            )
            return norm_space(masked)

    if debug:
        print("\n----------")
        print("Could not mask known Teufel citance")
        print("Author name detected:", author_name)
        print("Year detected:", citseg_year)
        print("Norm CITSEG:", citseg_norm)
        print("Original citance:", citance)
        print("Normalised citance:", citance_norm)

    return None

def split_context_sentences(citation_ctx):
    citation_ctx_protected = protect_abbreviations(citation_ctx)

    # Split after sentence-final punctuation when followed by optional space + capital letter
    ctx_split = re.split(r"(?<=[.!?])\s*(?=[A-Z])", citation_ctx_protected)

    ctx_split = [norm_space(restore_abbreviations(s)) for s in ctx_split if norm_space(s)]

    return ctx_split

def load_jurgens_data(length_left, length_right, jurgens_data_path, include_Teufel_data):

    citation_sequence_x_jurgens = []
    citation_sequence_y_jurgens = []
    citation_sections = []
    citation_sections_left = []
    citation_section_position = []
    dic_classes = {}

    df = pd.read_csv(jurgens_data_path).fillna("")
    counter = 0
    if include_Teufel_data:
        print("Including Teufel data")
    else:
        print("Skipping Teufel data")

    for _, row in df.iterrows():
        counter +=1    


        classname = str(row["Citation_function"]).strip().lower()
        citation_ctx = str(row["Citation_context"]).strip()
        citseg = str(row["CITSEG"]).strip()
        section = str(row["Section_title"]).strip()
        section_number = str(row["Section_number"]).strip()
        sections_left = str(row["Sections_left"]).strip()
        dataset_source = str(row["source_dataset"]).strip()
        left_sentence_1 = str(row["left_1"]).strip()
        left_sentence_2 = str(row["left_2"]).strip()
        left_sentence_3 = str(row["left_3"]).strip()
        citance_teufel = str(row["citance"]).strip()

        right_sentence_1 = str(row["right_1"]).strip()
        right_sentence_2 = str(row["right_2"]).strip()
        right_sentence_3 = str(row["right_3"]).strip()

        if section == "" or section.lower() == "false":
            section = section_number

        if classname == "" or citation_ctx == "" or citseg == "":
            continue

        if classname not in dic_classes:
            dic_classes[classname] = {
                "citance": [],
                "left_ctx": [],
                "right_ctx": [],
                "sequence": [],
                "section": [],
                "sections_left": []
            }

        ctx_split = split_context_sentences(citation_ctx)
        citseg_norm = norm_space(citseg)
        citance, citance_with_citseg_masked = find_citance_jurgens(citseg, ctx_split)

        if not include_Teufel_data:
            if dataset_source == "teufel":
                continue

        if citance:
            for i, sentence in enumerate(ctx_split):
                sentence_norm = norm_space(sentence)

                if citance in sentence:
                    if dataset_source == "jurgens":
                        left_sentences = ctx_split[max(0, i - length_left):i]
                        right_sentences = ctx_split[i + 1:i + 1 + length_right]

                        left_context = " ".join(norm_space(s) for s in left_sentences)
                        right_context = " ".join(norm_space(s) for s in right_sentences)

                        sequence = (left_context + " " + citance_with_citseg_masked + " " + right_context).strip()
                        sequence = norm_space(sequence)

                    else:
                        left_available = [left_sentence_3, left_sentence_2, left_sentence_1]
                        right_available = [right_sentence_1, right_sentence_2, right_sentence_3]

                        left_available = [norm_space(s) for s in left_available if norm_space(s)]
                        right_available = [norm_space(s) for s in right_available if norm_space(s)]

                        left_sentences = left_available[-length_left:] if length_left > 0 else []
                        right_sentences = right_available[:length_right] if length_right > 0 else []

                        left_context = " ".join(left_sentences)
                        right_context = " ".join(right_sentences)

                        citance = norm_space(citance_teufel)

                        citance_with_citseg_masked = mask_citseg_in_known_citance(citing_string=citseg, citance=citance)

                        if citance_with_citseg_masked is None:
                            print("\n" + "=" * 100)
                            print("Could not mask CITSEG in known Teufel citance")
                            print("Row counter:", counter)
                            print("Dataset source:", dataset_source)
                            print("Class:", classname)
                            print("CITSEG:", citseg)
                            print("Known Teufel citance:", citance_teufel)
                            print("Context:", citation_ctx)
                            print("=" * 100)

                            continue

                        sequence = (left_context + " " + citance_with_citseg_masked + " " + right_context).strip()

                        sequence = norm_space(sequence)

                        if "CITSEG" not in sequence:
                            print("\n" + "=" * 100)
                            print("Final Teufel sequence still has no CITSEG")
                            print("Row counter:", counter)
                            print("Dataset source:", dataset_source)
                            print("Class:", classname)
                            print("CITSEG:", citseg)
                            print("Known Teufel citance:", citance_teufel)
                            print("Masked citance:", citance_with_citseg_masked)
                            print("Final sequence:", sequence)
                            print("=" * 100)

                            continue

                                                
                    if sequence in citation_sequence_x_jurgens:
                        #skip duplicates
                        continue
                    

                    dic_classes[classname]["citance"].append(citance)
                    dic_classes[classname]["left_ctx"].append(left_context)
                    dic_classes[classname]["right_ctx"].append(right_context)
                    dic_classes[classname]["sequence"].append(sequence)
                    dic_classes[classname]["section"].append(section)
                    dic_classes[classname]["sections_left"].append(sections_left)

                    citation_sequence_x_jurgens.append(sequence)
                    citation_sequence_y_jurgens.append(classname)
                    citation_sections.append(section)
                    citation_section_position.append(section_number)
                    citation_sections_left.append(sections_left)

                    break

        else:
            citseg_norm2 = citseg.replace(" ,", ",").replace("( ", "(").replace(" )", ")").replace("  ", " ").replace("[ ", "[").replace(" ]", "]")


    print_sequence_samples_by_class(dic_classes, n=20, random_sample=False)
    return citation_sequence_x_jurgens, citation_sequence_y_jurgens, citation_sections, citation_sections_left, citation_section_position





def load_PD_data(length_left, length_right):
    def clean_sentence(sentence):
        cleaned_sentence = re.sub(r'<ref[^>]*>', '', sentence)
        cleaned_sentence = re.sub(r'type="[^"]*"\s*target="[^"]*">|type="bibr">', '', cleaned_sentence)
        cleaned_sentence = re.sub(r'</ref>|<ref', '', cleaned_sentence)
        return cleaned_sentence

    def map_labels_to_jurgens(y):
        new_y = []
        jurgens_labels = ['background', 'motivation', 'uses', 'extends', 'compareorcontrast', 'future']
        mapping = {"neutral":"Background", "motivation": "Motivation", "similar":"CompareOrContrast", "usage":"Uses", "cocores":"CompareOrContrast", "basis":"Extends", "weakness":"CompareOrContrast", "future":"Future", "support":"CompareOrContrast", "cocogm":"CompareOrContrast", "cocoxy": "Background"}
        old_labels= ['similar', 'neutral', 'usage', 'cocores', 'motivation', 'basis', 'weakness', 'future', 'support', 'cocogm', 'cocoxy']
        for label in y:
            if label.lower() not in jurgens_labels:
                mapped_label = mapping.get(label)
                new_y.append(mapped_label)
            else:
                new_y.append(label.lower())

        return new_y
    
    def define_y_100citation(labels):
        y = []
        for i in range(len(labels)):
            label = labels[i].split('|')[0].replace(' ', '').lower()
            y.append(label)
        y = map_labels_to_jurgens(y)
        return y
    
    def load_context_and_citances(df, begining_citation_sentences, end_citation_sentences, length_left, length_right):
        citances = []
        left_context_sentences = []
        right_context_sentences = []
        citation_sections_left = []
        context_dic_lists = {}
            

        for i in range(len(begining_citation_sentences)):
            citance = begining_citation_sentences[i]+ ' (CITSEG) '+end_citation_sentences[i]
            citance = clean_sentence(citance)
            citances.append(citance)

            if length_left is not None and length_right is not None:

                left_context = ''
                right_context = ''

                for n in range(1, length_left +1):
                    context_dic_lists[f'l{n}'] = df[f"l{n}"].astype(str).tolist()

                for n in range(1, length_right + 1):
                    context_dic_lists[f'r{n}'] = df[f"r{n}"].astype(str).tolist()

                for n in reversed(range(1, length_left+ 1)):
                    left_context += context_dic_lists[f'l{n}'][i] if context_dic_lists[f'l{n}'][i] != 'nan' else ''
                left_context = clean_sentence(left_context)
                left_context_sentences.append(left_context)


                for n in range(1, length_right +1):
                    right_context += context_dic_lists[f'r{n}'][i]+' ' if context_dic_lists[f'r{n}'][i] != 'nan' else ''
                    right_context = clean_sentence(right_context)
                right_context_sentences.append(right_context)
        
        return left_context_sentences, right_context_sentences, citances


    def clean_section_tag(section):
        from html import unescape
        if section is None:
            return ""

        section = str(section)

        #Decode HTML/XML entities
        section = unescape(section)
        #Remove XML/HTML tags
        section = re.sub(r"<[^>]+>", " ", section)
        #Normalise spaces
        section = re.sub(r"\s+", " ", section).strip()

        return section

    dataset ='100_citation_sample - annotation_jurgens.csv'
    df = pd.read_csv(dataset)
    begining_citation_sentences = df["citation_sentence"].astype(str).tolist()
    end_citation_sentences = df["end_citation_sentence"].astype(str).tolist()
    section_pd100cit = df["section"].astype(str).tolist()
    labels = df["annotation_rhetorical_function"].astype(str).tolist()

    section_pd100cit_clean = [clean_section_tag(section) for section in section_pd100cit]

    #The section does not seem to improve the scores
    #section_position = df["section_position"].astype(str).tolist()
    citation_sections_left_pd100cit = ["" for i in range(len(section_pd100cit_clean))]
    section_position = ["" for i in range(len(section_pd100cit_clean))]

    citation_sequence_y_100citations = define_y_100citation(labels)
    left_context_sentences, right_context_sentences, citances = load_context_and_citances(df, begining_citation_sentences, end_citation_sentences, length_left, length_right)
    citation_sequence_x_100citations = [left_context_sentences[i]+citances[i] + right_context_sentences[i] for i in range(len(citances))]

    return citation_sequence_x_100citations, citation_sequence_y_100citations, section_pd100cit_clean, citation_sections_left_pd100cit, section_position


def print_section_counts(section_list, dataset_name, mapping_section):
    mapped_sections = [normalise_section(section) for section in section_list]
    section_counts = Counter(mapped_sections)

    print(f"\nCount per mapped section - {dataset_name}:")
    for section in mapping_section:
        print(f"{section}: {section_counts.get(section, 0)}")

    return mapped_sections, section_counts

def to_categorical(labels, all_labels):
    return [all_labels.index(lbl.lower()) for lbl in labels]

def get_data_list(window_context, jurgens_data_path, include_Teufel_data):
  if window_context is not None:
    length_left, length_right = int(window_context.split('-')[0]), int(window_context.split('-')[1])
  else:
     length_left, length_right = 0, 0


  citation_sequence_x_100citations, citation_sequence_y_100citations, section_pd100cit, citation_sections_left_pd100cit, section_position_pd100cit = load_PD_data(length_left, length_right)
  citation_sequence_x_jurgens, citation_sequence_y_jurgens, citation_sections_jurgens, citation_sections_left_jurgens, citation_section_position_jurgens = load_jurgens_data(length_left, length_right, jurgens_data_path, include_Teufel_data)
  mapping_section = ["Abstract", "Introduction", "Related work", "Method", "Experiment", "Results", "Discussion", "Future work", "Conclusion", "Missing"]
  set_sections = set()
  
  mapped_sections_jurgens, section_counts_jurgens = print_section_counts(citation_sections_jurgens, "Jurgens", mapping_section)
  mapped_sections_pd100cit, section_counts_pd100cit = print_section_counts(section_pd100cit, "PD100cit", mapping_section)

  return citation_sequence_x_100citations, citation_sequence_y_100citations, mapped_sections_pd100cit, section_pd100cit, citation_sections_left_pd100cit, section_position_pd100cit, citation_sequence_x_jurgens, citation_sequence_y_jurgens, mapped_sections_jurgens, citation_sections_jurgens, citation_sections_left_jurgens, citation_section_position_jurgens
