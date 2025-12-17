import torch
from transformers import AutoTokenizer, AutoModel
import re
from pandas import *
import pandas as pd
import json
from tqdm import tqdm
import os

def clean_sentence(sentence):
    cleaned_sentence = re.sub(r'<ref[^>]*>', '', sentence)
    cleaned_sentence = re.sub(r'type="[^"]*"\s*target="[^"]*">|type="bibr">', '', cleaned_sentence)
    cleaned_sentence = re.sub(r'</ref>|<ref', '', cleaned_sentence)
    #on supprime toutes les balises xml en général
    cleaned_sentence = re.sub(r'<[^>]+>', '', sentence)
    return cleaned_sentence


def load_graph(graph_path):
    with open(graph_path, "r", encoding="utf-8") as f:
        return json.load(f)

def classify(loaded_model, graph, output_graph_path):
    '''Classify the citation passages from the citation graph and enrich the graph with the predicted rhetorical class indexed in the citation_passage'''

    all_labels = ['similar', 'neutral', 'usage', 'cocores', 'motivation', 'basis', 'weakness', 'future', 'support', 'cocogm', 'cocoxy']
    edges = graph.get('edges', [])

    models = {'PubMedBERT':'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext','BioLinkBERT': 'michiyasunaga/BioLinkBERT-base','BioBERT': 'dmis-lab/biobert-v1.1', 'SciBERT': 'allenai/scibert_scivocab_uncased', 'RoBERTa-large': 'all-roberta-large-v1', 'RoBERTa' : 'roberta-base'}
    model_name = models['BioBERT']
    model = AutoModel.from_pretrained(model_name)

    print(f'Model {model} loaded to device')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(['CITSEG'], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    for idx, edge in enumerate(tqdm(edges, desc="Classifying citation passages")):
        #Check that citation_passages exists and contains a section
        if "citation_passages" in edge and any("section" in p for p in edge["citation_passages"]):
            #Iterate over each passage
            for passage in edge["citation_passages"]:
                citation_before_cleaning = passage.get("Full-text", "")
                citation = clean_sentence(citation_before_cleaning)
                inputs = tokenizer(
                    citation,
                    return_tensors='pt',
                    padding='max_length',
                    max_length=512,
                    truncation=True,
                    add_special_tokens=True,
                    return_attention_mask=True
                )

                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                    
                preds = loaded_model(tokens=input_ids, attention_mask=attention_mask)

                predicted_classes = torch.argmax(preds, dim=1).cpu().item()
                predicted_rhetorical_class = all_labels[predicted_classes]
                passage["predicted_rhetorical_class"] = predicted_rhetorical_class

    new_graph = {"years": graph.get("years", []),
        "year_arts": graph.get("year_arts", {}),
        "articles": graph.get("articles", []), 
        "findings": graph.get("findings", []),
        "edges": edges}

    with open(output_graph_path, "w") as f:
        json.dump(new_graph, f, indent=2)

    print(f"Saved updated graph to {output_graph_path}")
    

if __name__ == "__main__":
    if torch.cuda.is_available():
      device = torch.device("cuda") 
      print(f"PyTorch is using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")

    print("Loading dataset")

    entry_graph_path = "../new_corpus/graph_with_citations_V2.json"
    output_graph_path = "graph_with_classified_citations_V1_BIOBERT.json"

    graph = load_graph(entry_graph_path)
    #classify on context window 3-3 sentences
    loaded_model = torch.load('BioBERT_FINALMODEL_2e-05_accseteps1_ctx_3-3_Jiang_train-PD_test_42.pt')
    loaded_model.eval()
    classify(loaded_model, graph, output_graph_path)









