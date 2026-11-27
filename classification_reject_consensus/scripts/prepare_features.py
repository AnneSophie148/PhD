import json
from collections import defaultdict
from collections import Counter, defaultdict
import math
import sys
sys.path.append("../..")
from transformers import AutoTokenizer, AutoModel
import seaborn as sns
import matplotlib.pyplot as plt
from citation_classifier import CitationClassifier
from torch.utils.data import DataLoader
import random
import numpy as np
from ArticleRNN import *
import torch
from torch.nn.utils.rnn import pad_sequence
from utils import CitationDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score



def load_model_from_checkpoint(checkpoint_path, device):
    models = {
        'PubMedBERT': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
        'BioLinkBERT': 'michiyasunaga/BioLinkBERT-base',
        'BioBERT': '/mnt/beegfs/projects/ppvir/models/biobert-v1.1',
        'SciBERT': 'allenai/scibert_scivocab_uncased',
        'RoBERTa-large': 'all-roberta-large-v1',
        'RoBERTa': 'roberta-base'
    }

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_short_name = checkpoint["model_short_name"]
    model_name = models[model_short_name]

    print("Loading checkpoint from:")
    print(checkpoint_path)

    print("Model short name:", model_short_name)
    print("Model HF name:", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)

    tokenizer.add_tokens(['CITSEG'], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    citseg_id = tokenizer.convert_tokens_to_ids("CITSEG")
    print("CITSEG token id:", citseg_id)

    loaded_model = CitationClassifier(
        linear_size=checkpoint["hidden_layers"],
        model=model,
        tokenizer=tokenizer,
        in_features=checkpoint["in_features"],
        num_class=len(checkpoint["all_labels"]),
        use_section=checkpoint["use_section"],
        num_sections=checkpoint["num_sections"],
        section_emb_dim=32,
        dropout=0.5
    )

    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model = loaded_model.to(device)
    loaded_model.eval()

    all_labels = checkpoint["all_labels"]

    print("Checkpoint loaded correctly")
    print("Labels:", all_labels)
    print("Use section:", checkpoint["use_section"])

    return loaded_model, tokenizer, all_labels


def tokenize_keeping_citseg(text, tokenizer, max_length=512, device=None):
    citseg_token = "CITSEG"
    citseg_id = tokenizer.convert_tokens_to_ids(citseg_token)

    if citseg_token not in text:
        return None, None, "Missing CITSEG in cleaned text"

    if citseg_id is None or citseg_id == tokenizer.unk_token_id:
        return None, None, "CITSEG token is unknown to tokenizer"

    before, after = text.split(citseg_token, 1)

    before_ids = tokenizer.encode(before, add_special_tokens=False)
    after_ids = tokenizer.encode(after, add_special_tokens=False)

    #Reserve space for CITSEG
    available_len = max_length - 3

    before_len = available_len // 2
    after_len = available_len - before_len

    #Keep the context closest to CITSEG
    before_ids = before_ids[-before_len:]
    after_ids = after_ids[:after_len]

    input_ids = ([tokenizer.cls_token_id] + before_ids + [citseg_id] + after_ids + [tokenizer.sep_token_id])
    attention_mask = [1] * len(input_ids)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.sep_token_id

    pad_len = max_length - len(input_ids)

    input_ids = input_ids + [pad_token_id] * pad_len
    attention_mask = attention_mask + [0] * pad_len

    input_ids = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long)

    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

    if citseg_id not in input_ids[0].tolist():
        return None, None, "CITSEG missing after tokenization"

    return input_ids, attention_mask, None

def embed_citations(loaded_model, tokenizer, citation_text, device):
    citation = citation_text
    input_ids, attention_mask, issue = tokenize_keeping_citseg(text=citation, tokenizer=tokenizer, max_length=512, device=device,)

    if issue is not None:
        return None

    encoder = loaded_model.model

    with torch.no_grad():
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask,)

    citseg_id = tokenizer.convert_tokens_to_ids("CITSEG")
    citseg_position = (input_ids[0] == citseg_id).nonzero(as_tuple=True)[0].item()

    return outputs.last_hidden_state[0, citseg_position].cpu().tolist()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def encode_identification_methods(identification_methods):
    encoded = [0.0] * len(identification_methods_list)
    def collect_methods(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "normalized_method":
                    methods = item if isinstance(item, list) else [item]
                    for method in methods:
                        if not isinstance(method, str) or not method.strip():
                            continue
                        method = method.strip().lower()
                        if method not in method_to_index:
                            print("Unknown normalized identification method:", method)
                            method = "other"
                        encoded[method_to_index[method]] = 1.0
                else:
                    collect_methods(item)
        elif isinstance(value, list):
            for item in value:
                collect_methods(item)
    collect_methods(identification_methods)
    return encoded


time_series_per_article = load_json("time_series_per_article.json")
rhetorical_classes = ['CoCoGM', 'background', 'compare_results', 'contrast - contradictory results', 'contrast_results', 'extends', 'future', 'motivation', 'unknown', 'uses']
categories = ["points_noirs", "points_blancs"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "../../model_save_sci_cite/DIC_FINAL_BioBERT_5e-05_accsteps1_ctx_3-3_section_False_32_scicite_True.pt"
loaded_model, tokenizer, all_labels = load_model_from_checkpoint(checkpoint_path=checkpoint_path, device=device)
citation_features_by_time = {}
method_features_by_article = {}

identification_methods_list = ["OTHER", "UNSPECIFIED", "Biological Identification / Differentiation Assay", "Morphological / Taxonomic Identification", "Light / Fluorescence Microscopy", "Electron Microscopy", "Isolation / Culture", "Biochemical / Physiological Identification", "Metabolic / Chemotaxonomic Profiling", "ELISA", "Other Immunological Assay", "PCR", "RT-PCR", "qPCR", "RT-qPCR", "Digital PCR", "LAMP / Isothermal Amplification", "Restriction-Based Identification", "DNA Fingerprinting / Genotyping", "SSR / Microsatellite Genotyping", "SNP Genotyping", "Sanger Sequencing", "DNA Barcoding", "Multilocus Sequence Analysis / Typing", "HTS / NGS", "Whole-Genome Sequencing", "Sequence Comparison / Phylogenetic Analysis", "Nucleic-Acid Hybridization", "Protein / Isozyme / Mass-Spectrometry Profiling"]
identification_methods_list = [e.lower() for e in identification_methods_list]
method_to_index = {method: index for index, method in enumerate(identification_methods_list)}

for category, relation_types in time_series_per_article.items():
    citation_features_by_time[category] = {}
    method_features_by_article[category] = {}

    for relation_type, articles in relation_types.items():
        for doi, article_info in articles.items():
            cited_info = article_info.get("cited_info") or {}
            publication_year = int(cited_info.get("year"))
            identification_methods = cited_info.get("identification_methods")
            encoded_identification_methods = encode_identification_methods(identification_methods)

            method_features_by_article[category][doi] = encoded_identification_methods
            features_by_time_step = citation_features_by_time[category].setdefault(doi, defaultdict(list))

            for citation in article_info.get("citation_passages", []):
                citing_year = citation.get("year")
                time_step = citing_year - publication_year
                rc = citation.get("rhetorical_class", "unknown")

                encoded_rc = rhetorical_classes.index(rc)
                embedding = embed_citations(loaded_model, tokenizer, citation.get("text"), device)
                if embedding is None:
                    continue

                citation_feature = {
                    "rhetorical_class": encoded_rc,
                    "citation_embedding": embedding,
                    "citing_doi": citation.get("citing_doi"),
                    "citation_event": citation.get("citation_event"),
                    "section": citation.get("section", "unknown"),
                    "self_citation": citation.get("self_citation", "unknown")
                }

                features_by_time_step[time_step].append(citation_feature)
            
x_noirs = []
x_blancs = []
for doi, features_by_time_step in citation_features_by_time["points_noirs"].items():
    if doi in citation_features_by_time["points_noirs"] and doi in citation_features_by_time["points_blancs"]:
        continue
    if any(features_by_time_step.values()):
        x_noirs.append({"doi": doi, "features_by_time_step": features_by_time_step, "method_features": method_features_by_article["points_noirs"][doi]})
for doi, features_by_time_step in citation_features_by_time["points_blancs"].items():
    if doi in citation_features_by_time["points_noirs"] and doi in citation_features_by_time["points_blancs"]:
        continue
    if any(features_by_time_step.values()):
        x_blancs.append({"doi": doi, "features_by_time_step": features_by_time_step, "method_features": method_features_by_article["points_blancs"][doi]})

SEED = 42
random.seed(SEED)
rng = random.Random(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

rng.shuffle(x_noirs)
rng.shuffle(x_blancs)

print("Points noirs:", len(x_noirs))
print("Points blancs:", len(x_blancs))

def split_articles(articles):
    n_val = round(len(articles) * 0.15)
    n_test = round(len(articles) * 0.20)
    train_end = len(articles) - n_val - n_test
    val_end = train_end + n_val
    return articles[:train_end], articles[train_end:val_end], articles[val_end:]


noirs_train, noirs_val, noirs_test = split_articles(x_noirs)
blancs_train, blancs_val, blancs_test = split_articles(x_blancs)

x_train = noirs_train + blancs_train
x_val = noirs_val + blancs_val
x_test = noirs_test + blancs_test

labels_idx_train = [0] * len(noirs_train) + [1] * len(blancs_train)
labels_idx_val = [0] * len(noirs_val) + [1] * len(blancs_val)
labels_idx_test = [0] * len(noirs_test) + [1] * len(blancs_test)

print("Composition of y test")
print(labels_idx_test)

train_dataset_full = CitationDataset(x_train, labels_idx_train, num_rhetorical_classes=len(rhetorical_classes), pooling=False)
val_dataset_full = CitationDataset(x_list=x_val, labels_ind=labels_idx_val, num_rhetorical_classes=len(rhetorical_classes), pooling=False)
test_dataset_full = CitationDataset(x_list=x_test, labels_ind=labels_idx_test, num_rhetorical_classes=len(rhetorical_classes), pooling=False)


batch_size = 1

g = torch.Generator()
g.manual_seed(SEED)
train_loader = DataLoader(train_dataset_full, batch_size=batch_size, shuffle=True, generator=g)
val_loader = DataLoader(val_dataset_full, batch_size=batch_size, shuffle=True, generator=g)
test_loader = DataLoader(test_dataset_full, batch_size=batch_size, shuffle=True, generator=g)


input_size = train_dataset_full[0]["features"].shape[1]
#model = ArticleRNN(input_size=input_size, hidden_size=128).to(device)
torch.manual_seed(SEED)
model = ArticleRNN(input_size=input_size, num_methods=len(identification_methods_list), hidden_size=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
num_epochs = 10


for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_labels = []
    train_predictions = []
    for batch in train_loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        #logits = model(features)
        method_features = batch["method_features"].to(device)
        logits = model(features, method_features)   
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * labels.size(0)
        train_labels.extend(batch["labels"].tolist())
        train_predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())

    model.eval()
    val_labels = []
    val_predictions = []
    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            #logits = model(features)
            method_features = batch["method_features"].to(device)
            logits = model(features, method_features)   
            predictions = logits.argmax(dim=1)
            val_labels.extend(batch["labels"].tolist())
            val_predictions.extend(logits.argmax(dim=1).cpu().tolist())

    train_acc = accuracy_score(train_labels, train_predictions)
    train_f1 = f1_score(train_labels, train_predictions, average="macro", zero_division=0)
    val_acc = accuracy_score(val_labels, val_predictions)
    val_f1 = f1_score(val_labels, val_predictions, average="macro", zero_division=0)
    print(f"Epoch {epoch + 1} | loss={train_loss / len(train_labels):.4f} | train accuracy={train_acc:.3f} F1={train_f1:.3f} | val accuracy={val_acc:.3f} F1={val_f1:.3f}")

model.eval()
test_labels = []
test_predictions = []
with torch.no_grad():
    for batch in test_loader:
        features = batch["features"].to(device)
        #logits = model(features)
        method_features = batch["method_features"].to(device)
        logits = model(features, method_features)   
        predictions = logits.argmax(dim=1).cpu().tolist()
        labels = batch["labels"].tolist()
        for predicted, true in zip(predictions, labels):
            print(f"Predicted: {predicted} | True: {true}")
        test_predictions.extend(predictions)
        test_labels.extend(labels)

test_f1 = f1_score(test_labels, test_predictions, average="macro", zero_division=0)
print(f"Test macro F1: {test_f1}")

class_ids = [0, 1]
class_names = ["Points noirs", "Points blancs"]
report = classification_report(test_labels, test_predictions, labels=class_ids, target_names=class_names, digits=3, zero_division=0)
print("\nTest classification report")
print(report)
