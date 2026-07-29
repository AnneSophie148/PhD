import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from get_citation_sequence_updated import get_data_list, load_scicite_coco_augmentation
from citation_classifier import CitationClassifier, training_step, validation_step
import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm
import numpy as np
import random
from utils import CitationDataset, list_idx, plot_metric_evolution
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import json
from collections import Counter


def print_class_distribution(y_labels, dataset_name, all_labels=None):
    """
    Print class distribution from y labels.
    """

    counts = Counter(y_labels)
    total = len(y_labels)

    print("\n" + "=" * 70)
    print(f"CLASS DISTRIBUTION: {dataset_name}")
    print("=" * 70)
    print(f"Total examples: {total}")

    if all_labels is not None:
        for label in all_labels:
            count = counts.get(label, 0)
            percent = 100 * count / total if total > 0 else 0
            print(f"{label:20s} {count:6d} ({percent:6.2f}%)")

        unexpected = set(counts.keys()) - set(all_labels)
        if unexpected:
            print("\nUnexpected labels:")
            for label in sorted(unexpected):
                count = counts[label]
                percent = 100 * count / total if total > 0 else 0
                print(f"{str(label):20s} {count:6d} ({percent:6.2f}%)")

    else:
        for label, count in counts.most_common():
            percent = 100 * count / total if total > 0 else 0
            print(f"{str(label):20s} {count:6d} ({percent:6.2f}%)")

def build_position_embeddings(section_positions, sections_left, max_sections=7):
    embeddings = []
    for i in range(len(section_positions)):
        section_from_beginning = int(section_positions[i])
        sections_left_i = int(sections_left[i])
        '''
        #use to impose a maximum of 7 sections
        if section_from_beginning > max_sections:
            section_from_beginning = max_sections
        if sections_left_i > max_sections:
            sections_left_i = max_sections
        if section_from_beginning < 0:
            section_from_beginning = 0
        if sections_left_i < 0:
            sections_left_i = 0'''
        embeddings.append([section_from_beginning, sections_left_i])
    return embeddings
    
def main(model_short_name, window, data_repartition, training_data_path, SEED, use_section, lr, num_of_epochs, ACC_STEP, batch_size, dropout, include_Teufel_data, section_emb_dim, augment_coco_with_scicite=False, scicite_csv_path="Cohan_compare_contrast.csv"):
  os.makedirs("metrics", exist_ok=True)
  os.makedirs("model_save", exist_ok=True)
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed_all(SEED)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

  print("La version de torch est : ",torch.__version__)

  if torch.cuda.is_available():
      device = torch.device("cuda") 
      print(f"PyTorch is using GPU: {torch.cuda.get_device_name(0)}")
  else:
      device = torch.device("cpu")
 

  models = {'PubMedBERT':'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext','BioLinkBERT': 'michiyasunaga/BioLinkBERT-base','BioBERT': 'dmis-lab/biobert-v1.1', 'SciBERT': 'allenai/scibert_scivocab_uncased', 'RoBERTa-large': 'all-roberta-large-v1', 'RoBERTa' : 'roberta-base'}
  
  model_name = models[model_short_name]
  model = AutoModel.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  

  model_name = models[model_short_name]
  tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
  model = AutoModel.from_pretrained(model_name, local_files_only=True)
  print(f'Model {model} loaded to device')

  tokenizer.add_tokens(['CITSEG'], special_tokens=True)
  model.resize_token_embeddings(len(tokenizer))
  
  in_features = model.config.hidden_size
  hidden_layers = in_features
  print("Number hidden layers : ", hidden_layers)
  
  citation_sequence_x_100citations, citation_sequence_y_100citations, mapped_sections_pd100cit, section_pd100cit, citation_sections_left_pd100cit, citation_section_position_pd100cit, citation_sequence_x_jurgens, citation_sequence_y_jurgens, mapped_sections_jurgens, citation_sections_jurgens, citation_sections_left_jurgens, citation_section_position_jurgens = get_data_list(window, training_data_path, include_Teufel_data)
  citation_sequence_y_100citations = [y.lower() for y in citation_sequence_y_100citations]
  all_labels = ['background', 'motivation', 'uses', 'extends', 'compareorcontrast', 'future']
  label_mapping = {label: idx for idx, label in enumerate(all_labels)}
  
  print_class_distribution(citation_sequence_y_jurgens, dataset_name="Jurgens", all_labels=all_labels)
  print_class_distribution(citation_sequence_y_100citations, dataset_name="PD100Cit", all_labels=all_labels)
  
  mapping_section = ["Abstract", "Introduction", "Related work", "Method", "Experiment", "Results", "Discussion", "Future work", "Conclusion", "Missing"]
  section_to_idx = {section_name: idx for idx, section_name in enumerate(mapping_section)}
  
  section_idx_jurgens = [section_to_idx[section] for section in mapped_sections_jurgens]
  section_idx_pd100cit = [section_to_idx[section] for section in mapped_sections_pd100cit]
  use_position_embedding = use_section and section_emb_dim == "Position_embedding"
  if use_position_embedding:
    embeddings_jurgens = build_position_embeddings(citation_section_position_jurgens, citation_sections_left_jurgens, max_sections=7)
    embeddings_pd100cit = build_position_embeddings(citation_section_position_pd100cit, citation_sections_left_pd100cit, max_sections=7)
    section_features_jurgens = embeddings_jurgens
    section_features_pd100cit = embeddings_pd100cit
  else:
    section_features_jurgens = section_idx_jurgens
    section_features_pd100cit = section_idx_pd100cit

  print("LEN citation_sequence_x_jurgens", len(citation_sequence_x_jurgens))
  print("LEN section_features_jurgens:", len(section_features_jurgens))
  print("Example section_features_jurgens:", section_features_jurgens[:10])
  data_to_shuffle = list(zip(citation_sequence_x_jurgens, citation_sequence_y_jurgens, section_features_jurgens))
  random.shuffle(data_to_shuffle)
  citation_sequence_x_jurgens_shuffled, citation_sequence_y_jurgens_shuffled, section_features_jurgens_shuffled = zip(*data_to_shuffle)
  citation_sequence_x_jurgens_shuffled = list(citation_sequence_x_jurgens_shuffled)
  citation_sequence_y_jurgens_shuffled = list(citation_sequence_y_jurgens_shuffled)
  section_features_jurgens_shuffled = list(section_features_jurgens_shuffled)
      
  if data_repartition == 'Jurgens_train-PD_test':
    train_ratio = 0.80
    val_ratio = 0.20
    total_samples = len(citation_sequence_x_jurgens_shuffled)
    train_size = int(train_ratio * total_samples)
    val_size = total_samples - train_size
    x_train = citation_sequence_x_jurgens_shuffled[:train_size]
    y_train = citation_sequence_y_jurgens_shuffled[:train_size]
    sec_train = section_features_jurgens_shuffled[:train_size]
    x_val = citation_sequence_x_jurgens_shuffled[train_size:]
    y_val = citation_sequence_y_jurgens_shuffled[train_size:]
    sec_val = section_features_jurgens_shuffled[train_size:]
    x_test = citation_sequence_x_100citations
    y_test = citation_sequence_y_100citations
    sec_test = section_features_pd100cit
    print("LEN X TEST : ", len(x_test))
  else:
    train_ratio = 0.65
    val_ratio = 0.15
    test_ratio = 0.20
    total_samples = len(citation_sequence_x_jurgens_shuffled)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    x_train = citation_sequence_x_jurgens_shuffled[:train_size]
    y_train = citation_sequence_y_jurgens_shuffled[:train_size]
    sec_train = section_features_jurgens_shuffled[:train_size]
    x_val = citation_sequence_x_jurgens_shuffled[train_size:train_size + val_size]
    y_val = citation_sequence_y_jurgens_shuffled[train_size:train_size + val_size]
    sec_val = section_features_jurgens_shuffled[train_size:train_size + val_size]
    x_test = citation_sequence_x_jurgens_shuffled[train_size + val_size:]
    y_test = citation_sequence_y_jurgens_shuffled[train_size + val_size:]
    sec_test = section_features_jurgens_shuffled[train_size + val_size:]


  if augment_coco_with_scicite:
    scicite_x, scicite_y, scicite_mapped_sections, scicite_section_features = load_scicite_coco_augmentation(scicite_csv_path=scicite_csv_path, mapping_section=mapping_section, use_position_embedding=use_position_embedding, max_sections=7)
    
    for i in range(len(scicite_x)):
        print("\nClass : ", )
        print("Scicite X : ", scicite_y[i])
        print(scicite_x[i])
        print(scicite_mapped_sections[i])
        print(scicite_section_features[i])
    
    x_train = list(x_train) + scicite_x
    y_train = list(y_train) + scicite_y
    sec_train = list(sec_train) + scicite_section_features
    train_data_to_shuffle = list(zip(x_train, y_train, sec_train))
    random.shuffle(train_data_to_shuffle)
    x_train, y_train, sec_train = zip(*train_data_to_shuffle)
    x_train = list(x_train)
    y_train = list(y_train)
    sec_train = list(sec_train)
    print("Added SciCite/Cohan examples only to train:", len(scicite_x))



  print("DATA REPARTITION : ", data_repartition)
  labels_idx_train, labels_idx_val, labels_idx_test = list_idx(label_mapping, y_train), list_idx(label_mapping, y_val), list_idx(label_mapping, y_test)

  citation_model = CitationClassifier(linear_size=hidden_layers, model=model, tokenizer=tokenizer, in_features=in_features, num_class=len(all_labels), use_section=use_section, num_sections=len(mapping_section), section_emb_dim=section_emb_dim, dropout=dropout)
  citation_model.to(device)
  
  train_dataset = CitationDataset(text_citations=x_train, labels_ind=labels_idx_train, tokenizer=tokenizer, sections_ind=sec_train, use_position_embedding=use_position_embedding)
  val_dataset = CitationDataset(text_citations=x_val, labels_ind=labels_idx_val, tokenizer=tokenizer, sections_ind=sec_val, use_position_embedding=use_position_embedding)
  test_dataset = CitationDataset(text_citations=x_test, labels_ind=labels_idx_test, tokenizer=tokenizer, sections_ind=sec_test, use_position_embedding=use_position_embedding)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
  
  optimizer = AdamW(citation_model.parameters(), lr=lr, weight_decay= 0.002)
  loss_fn = nn.CrossEntropyLoss()
  num_training_steps = math.ceil(len(train_loader) / ACC_STEP) * num_of_epochs
  warmup=0.2
  warmup_steps = math.floor(num_training_steps * warmup)

  scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
  
  best_f1, best_macrof1, best_epoch  = 0, 0, 0
  scicite_aug_key = f"scicite_{augment_coco_with_scicite}"
  os.makedirs(f"model_save_sci_cite/seed_{SEED}", exist_ok=True)
  best_macro_model_path = (
    f"model_save_sci_cite/seed_{SEED}/"
    f"BEST_MACROF1_{model_short_name}_{lr}_accsteps{ACC_STEP}_"
    f"ctx_{window}_section_{use_section}_{section_emb_dim}_{scicite_aug_key}.pt")
  train_losses, val_losses = [], []
  train_accuracies, val_accuracies = [], []
  val_F_measures, train_F = [], []
  val_macros, train_macros = [], []

  dic_values_per_class = {classe: {'F1': {'train': [], 'val': []}, 'P': {'train': [], 'val': []}, 'R': {'train': [], 'val': []}} for classe in all_labels}
  
  for epoch in tqdm(range(num_of_epochs)):      
    train_loss = training_step(train_loader, citation_model,optimizer, loss_fn, device, ACC_STEP, scheduler)
    train_acc, train_f1, train_loss2, train_F1macro, train_precision_per_class, train_recall_per_class, train_f1_per_class = validation_step(train_loader, citation_model, loss_fn, device, all_labels)
    val_acc, val_f1, val_loss, val_F1macro, val_precision_per_class, val_recall_per_class, val_f1_per_class = validation_step(val_loader, citation_model, loss_fn, device, all_labels)


    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    val_F_measures.append(val_f1)
    train_F.append(train_f1)
    val_losses.append(val_loss)
    val_macros.append(val_F1macro)
    train_macros.append(train_F1macro)

    for class_idx, classe in enumerate(all_labels):
        dic_values_per_class[classe]['F1']['val'].append(val_f1_per_class[class_idx])
        dic_values_per_class[classe]['P']['val'].append(val_precision_per_class[class_idx])
        dic_values_per_class[classe]['R']['val'].append(val_recall_per_class[class_idx])
      
    for class_idx, classe in enumerate(all_labels):
        dic_values_per_class[classe]['F1']['train'].append(train_f1_per_class[class_idx])
        dic_values_per_class[classe]['P']['train'].append(train_precision_per_class[class_idx])
        dic_values_per_class[classe]['R']['train'].append(train_recall_per_class[class_idx])


    if val_f1 > best_f1:
        best_f1 = val_f1

    if val_F1macro > best_macrof1:
        best_macrof1 = val_F1macro
        best_epoch=epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": citation_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_macrof1": best_macrof1,
            "best_f1": best_f1,
            "val_f1": val_f1,
            "val_F1macro": val_F1macro,
            "model_short_name": model_short_name,
            "lr": lr,
            "ACC_STEP": ACC_STEP,
            "window": window,
            "use_section": use_section,
            "mapping_section": mapping_section,
            "all_labels": all_labels,
            "label_mapping": label_mapping,
            "in_features": in_features,
            "hidden_layers": hidden_layers,
            "num_sections": len(mapping_section)
        }, best_macro_model_path)

        print(f"\nNew best macro-F1: {best_macrof1:.4f} at epoch {epoch}")
        print(f"Saved best macro-F1 model to: {best_macro_model_path}")

    
    mean_f1_per_class = {classe: np.nanmean(dic_values_per_class[classe]['F1']['val']) for classe in all_labels}
    print("Mean F1 par classe :", mean_f1_per_class )


    rows = []
    for class_name, metrics in dic_values_per_class.items():
        for metric, datasets in metrics.items():
            row = {'Class': class_name,'Metric': metric,'Train': str(datasets['train']),'Validation': str(datasets['val'])}
            rows.append(row)

    df = pd.DataFrame(rows)

    rows_mean = []
    for class_name, mean_f1 in mean_f1_per_class.items():
        rows_mean.append({'Class': class_name, 'Metric': 'Mean_F1_over_training', 'Train': '-', 'Validation': mean_f1})
    rows_mean.append({'Class': '-', 'Metric': 'Best_F1macro', 'Train': '-', 'Validation': best_macrof1})
    rows_mean.append({'Class': '-', 'Metric': 'Best_Val_F1', 'Train': '-', 'Validation': best_f1})


    df_means = pd.DataFrame(rows_mean)
    df = pd.concat([df, df_means], ignore_index=True)
    


    df.to_csv(f'metrics/metrics_per_class_{model_short_name}_{lr}_{ACC_STEP}_{data_repartition}_{window}_{section_emb_dim}_{scicite_aug_key}.csv', index=False)


  print(f"{model_short_name} Best val F : {best_f1} for epoch {best_epoch} for LR {lr} for ACC_STEPS {ACC_STEP} dropout {dropout}")
  print(f"{model_short_name} Best val Fmacro : {best_macrof1} for epoch {best_epoch} for LR {lr} for ACC_STEPS {ACC_STEP} dropout {dropout}")

  plot_metric_evolution("Loss", train_losses, val_losses, num_of_epochs, model_short_name, lr, ACC_STEP, window)
  plot_metric_evolution("Accuracy", train_accuracies, val_accuracies, num_of_epochs, model_short_name, lr, ACC_STEP, window)
  plot_metric_evolution("F-score", train_F, val_F_measures, num_of_epochs, model_short_name, lr, ACC_STEP, window)
  plot_metric_evolution("Fmacro-score", train_macros, val_macros, num_of_epochs, model_short_name, lr, ACC_STEP, window)

  for cls, metrics in dic_values_per_class.items():
    for metric, repartitions in metrics.items():
      if metric == 'F1':
        train_values = repartitions['train']
        val_values = repartitions['val']
        plot_metric_evolution(f"{cls}_{metric}", train_values, val_values, num_of_epochs, model_short_name, lr, ACC_STEP, window)

    
  final_model_name_not_dic = (f"model_save_sci_cite/seed_{SEED}/"
    f"{model_short_name}_FINALMODEL_{lr}_accseteps{ACC_STEP}_"
    f"ctx_{window}_{use_section}_{section_emb_dim}_{scicite_aug_key}.pt")
   
  final_model_name = (
    f"model_save_sci_cite/seed_{SEED}/"
    f"DIC_FINAL_{model_short_name}_{lr}_accsteps{ACC_STEP}_"
    f"ctx_{window}_section_{use_section}_{section_emb_dim}_{scicite_aug_key}.pt")
  torch.save(citation_model, final_model_name_not_dic)
  loaded_model = torch.load(final_model_name_not_dic)
  loaded_model.eval()


  torch.save({
    "model_state_dict": citation_model.state_dict(),
    "model_short_name": model_short_name,
    "lr": lr,
    "ACC_STEP": ACC_STEP,
    "window": window,
    "use_section": use_section,
    "mapping_section": mapping_section,
    "all_labels": all_labels,
    "label_mapping": label_mapping,
    "in_features": in_features,
    "hidden_layers": hidden_layers,
    "num_sections": len(mapping_section)}, final_model_name)

  
  checkpoint = torch.load(best_macro_model_path, map_location=device)
  citation_model.load_state_dict(checkpoint["model_state_dict"])
  citation_model.eval()

  csv_output = []
  predictions = []
  true_labels = []
  test_texts = []
  second_classes = []
  third_classes = []

  with torch.no_grad():
      for batch in test_loader:
          input_ids = batch["input_ids"].to(device)
          attention_mask = batch["attention_mask"].to(device)
          labels = batch["labels"].to(device)
          citation_text = batch["citation_text"]

          section_ids = batch.get("section_ids")
          if section_ids is not None:
              section_ids = section_ids.to(device)

          logits = citation_model(tokens=input_ids, attention_mask=attention_mask, section_ids=section_ids)

          topk_values, topk_indices = torch.topk(logits, k=3, dim=-1)

          predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()
          second_class = topk_indices[:, 1].cpu().numpy()
          third_class = topk_indices[:, 2].cpu().numpy()

          predictions.extend(predicted_classes.tolist())
          second_classes.extend(second_class.tolist())
          third_classes.extend(third_class.tolist())
          true_labels.extend(labels.cpu().numpy().tolist())
          test_texts.extend(citation_text)

  for i in range(len(true_labels)):
      csv_output.append({
          "Citation indice": i,
          "Citation text": test_texts[i],
          "Top1 Classe": all_labels[predictions[i]],
          "Top2 Classe": all_labels[second_classes[i]],
          "Top3 Classe": all_labels[third_classes[i]],
          "True Label": all_labels[true_labels[i]],
      })

  num_classes = len(all_labels)
  label_ids = list(range(num_classes))

  #Eveluation metrics

  test_accuracy = accuracy_score(true_labels, predictions)
 
  test_precision_micro = precision_score(true_labels, predictions, average="micro", labels=label_ids, zero_division=0)
  test_recall_micro = recall_score(true_labels, predictions, average="micro", labels=label_ids, zero_division=0)
  test_f1_micro = f1_score(true_labels, predictions, average="micro", labels=label_ids, zero_division=0)
  
  test_precision_macro = precision_score(true_labels, predictions, average="macro", labels=label_ids, zero_division=0)
  test_recall_macro = recall_score(true_labels, predictions, average="macro", labels=label_ids, zero_division=0)
  test_f1_macro = f1_score(true_labels, predictions, average="macro", labels=label_ids, zero_division=0)

  test_precision_weighted = precision_score( true_labels, predictions, average="weighted", labels=label_ids, zero_division=0)
  test_recall_weighted = recall_score( true_labels, predictions, average="weighted", labels=label_ids, zero_division=0)
  test_f1_weighted = f1_score(true_labels, predictions, average="weighted", labels=label_ids, zero_division=0)

  test_report = classification_report(true_labels, predictions, labels=label_ids, target_names=all_labels, zero_division=0, output_dict=True)
  test_confusion_matrix = confusion_matrix(true_labels, predictions, labels=label_ids).tolist()

  test_scores = {
      "seed": SEED,
      "model": model_short_name,
      "lr": lr,
      "dropout": dropout,
      "ACC_STEP": ACC_STEP,
      "batch_size": batch_size,
      "num_epochs": num_of_epochs,
      "window": window,
      "data_repartition": data_repartition,
      "use_section": use_section,

      "best_val_f1_micro": float(best_f1),
      "best_val_f1_macro": float(best_macrof1),
      "best_epoch": int(best_epoch),

      "test_accuracy": float(test_accuracy),

      "test_precision_micro": float(test_precision_micro),
      "test_recall_micro": float(test_recall_micro),
      "test_f1_micro": float(test_f1_micro),

      "test_precision_macro": float(test_precision_macro),
      "test_recall_macro": float(test_recall_macro),
      "test_f1_macro": float(test_f1_macro),

      "test_precision_weighted": float(test_precision_weighted),
      "test_recall_weighted": float(test_recall_weighted),
      "test_f1_weighted": float(test_f1_weighted),

      "classification_report": test_report,
      "confusion_matrix": test_confusion_matrix
  }

  os.makedirs(f"predictions_sci_cite/{SEED}", exist_ok=True)

  df = pd.DataFrame(csv_output)
  output_csv_path = (
      f"predictions_sci_cite/{SEED}/"
      f"{model_short_name}_predictions_"
      f"lr_{lr}_dropout_{dropout}_acc_steps_{ACC_STEP}_"
      f"repartition_{data_repartition}_ctx_{window}_section_{use_section}_{section_emb_dim}_data-Teufel-{include_Teufel_data}_{scicite_aug_key}.csv"
  )
  df.to_csv(output_csv_path, index=False)

  print(f"TEST SCORES for model {model_short_name} lr {lr} dropout {dropout} with section {use_section} include data Teufel {include_Teufel_data}")
  print(test_scores)

  return test_scores

  
  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run citation classification experiments")
    available_models = ["SciBERT", "BioBERT", "RoBERTa", "BioLinkBERT", "PubMedBERT"]

    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all"] + available_models,
        help="Model to use. Use 'all' to run all models."
    )

    parser.add_argument(
        "--window_context",
        type=str,
        default="3-3",
        help="The window for the right and left context. Example: '2-3' or None"
    )

    parser.add_argument(
        "--data_repartition",
        type=str,
        default="Jurgens_train-PD_test",
        help="Data repartition: 'Jurgens_train-PD_test', 'Jurgens_all', etc."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=5171,
        help="Seed. Only used if you want to run a single seed manually."
    )

    args = parser.parse_args()

    training_data_path = "Jurgens_features_augmented.csv"

    if args.model == "all":
        selected_models = available_models
    else:
        selected_models = [args.model]

    seeds = [42, 1965, 5171, 789]

    section_config = [False, True]
    #section_config = [True]
    #section_emb_dims = [32, 256]
    section_emb_dims = [32]
    #section_emb_dims = ["Position_embedding"]

    #include_Teufel_config = [False, True]
    include_Teufel_config = [True]
    augment_coco_with_scicite = True
    scicite_csv_path = "Cohan_compare_contrast.csv"

    num_of_epochs = 20
    lr = 5e-5
    ACC_STEP = 1
    batch_size = 32
    dropout = 0.5

    window_key = str(args.window_context).replace("/", "-")

    lr_key = f"lr_{lr}"
    dropout_key = f"dropout_{dropout}"

    all_scores = {}
    all_seed_rows = []
    summary_rows = []

    print("\nModels selected:", selected_models)
    print("Window size:", args.window_context)
    print("Data repartition:", args.data_repartition)
    print("Seeds:", seeds)
    print("Use section configs:", section_config)
    print("Section embedding dimensions:", section_emb_dims)
    print("Include Teufel configs:", include_Teufel_config)

    for model_name in selected_models:

        print("\n" + "#" * 120)
        print(f"RUNNING MODEL: {model_name}")
        print("#" * 120)

        results_dir = f"results_sci_cite/{model_name}"
        os.makedirs(results_dir, exist_ok=True)

        all_scores[model_name] = {}

        model_seed_rows = []
        model_summary_rows = []

        for include_Teufel_data in include_Teufel_config:

            for use_section in section_config:

                if use_section:
                    section_emb_dims_to_try = section_emb_dims
                else:
                    section_emb_dims_to_try = [None]

                # Loop over section embedding dim, meaningful when use_section=True
                for section_emb_dim in section_emb_dims_to_try:

                    section_emb_key = "NA" if section_emb_dim is None else str(section_emb_dim)

                    config_key = (
                        f"window_{window_key}_"
                        f"section_{use_section}_"
                        f"section_emb_dim_{section_emb_key}_"
                        f"teufel_{include_Teufel_data}_"
                        f"scicite_{augment_coco_with_scicite}"
                    )

                    all_scores[model_name][config_key] = {}

                    # Rows for this specific config, across seeds
                    rows_this_setting = []

                    print("\n" + "#" * 100)
                    print(
                        f"CONFIG: model={model_name} | "
                        f"window={args.window_context} | "
                        f"use_section={use_section} | "
                        f"section_emb_dim={section_emb_dim} | "
                        f"include_Teufel_data={include_Teufel_data} | "
                        f"augment_coco_with_scicite={augment_coco_with_scicite}"
                    )
                    print("#" * 100)

                    for seed in seeds:

                        seed_key = f"seed_{seed}"

                        if seed_key not in all_scores[model_name][config_key]:
                            all_scores[model_name][config_key][seed_key] = {}

                        if lr_key not in all_scores[model_name][config_key][seed_key]:
                            all_scores[model_name][config_key][seed_key][lr_key] = {}

                        print("\n" + "=" * 100)
                        print(
                            f"MODEL: {model_name} | "
                            f"SEED: {seed} | "
                            f"WINDOW: {args.window_context} | "
                            f"LR: {lr} | "
                            f"DROPOUT: {dropout} | "
                            f"SECTION: {use_section} | "
                            f"SECTION_EMB_DIM: {section_emb_dim} | "
                            f"TEUFEL: {include_Teufel_data} | "
                            f"SCICITE_AUG: {augment_coco_with_scicite}"
                        )
                        print("=" * 100)

                        # If use_section=False, section_emb_dim is not used.
                        section_emb_dim_for_main = section_emb_dim if use_section else 32

                        scores = main(
                            model_name,
                            args.window_context,
                            args.data_repartition,
                            training_data_path,
                            seed,
                            use_section,
                            lr,
                            num_of_epochs,
                            ACC_STEP,
                            batch_size,
                            dropout,
                            include_Teufel_data,
                            section_emb_dim_for_main,
                            augment_coco_with_scicite=augment_coco_with_scicite,
                            scicite_csv_path=scicite_csv_path
                        )

                        all_scores[model_name][config_key][seed_key][lr_key][dropout_key] = scores

                        row = {
                            "model": model_name,
                            "config": config_key,
                            "window_context": args.window_context,
                            "data_repartition": args.data_repartition,
                            "use_section": use_section,
                            "section_emb_dim": section_emb_dim,
                            "include_Teufel_data": include_Teufel_data,
                            "seed": seed,
                            "lr": lr,
                            "dropout": dropout,
                            "num_of_epochs": num_of_epochs,
                            "batch_size": batch_size,
                            "ACC_STEP": ACC_STEP,

                            "best_val_f1_micro": scores["best_val_f1_micro"],
                            "best_val_f1_macro": scores["best_val_f1_macro"],

                            "test_accuracy": scores["test_accuracy"],

                            "test_f1_micro": scores["test_f1_micro"],
                            "test_f1_macro": scores["test_f1_macro"],
                            "test_f1_weighted": scores["test_f1_weighted"],

                            "test_precision_micro": scores["test_precision_micro"],
                            "test_recall_micro": scores["test_recall_micro"],

                            "test_precision_macro": scores["test_precision_macro"],
                            "test_recall_macro": scores["test_recall_macro"],
                        }

                        rows_this_setting.append(row)

                        #Model-level and global-level per-seed rows
                        model_seed_rows.append(row)
                        all_seed_rows.append(row)

                    #Save per-seed CSV for this model/config
                    df_setting = pd.DataFrame(rows_this_setting)

                    per_seed_csv_path = (
                        f"{results_dir}/score_per_seed_"
                        f"model-{model_name}_"
                        f"window-{window_key}_"
                        f"section-{use_section}_"
                        f"section_emb_dim-{section_emb_key}_"
                        f"teufel-{include_Teufel_data}_"
                        f"scicite-{augment_coco_with_scicite}.csv"
                    )

                    os.makedirs(os.path.dirname(per_seed_csv_path), exist_ok=True)
                    df_setting.to_csv(per_seed_csv_path, index=False)

                    print(f"\nSaved per-seed CSV to: {per_seed_csv_path}")

                    #Average across seeds for this model/config
                    summary_row = {
                        "model": model_name,
                        "config": config_key,
                        "window_context": args.window_context,
                        "data_repartition": args.data_repartition,
                        "use_section": use_section,
                        "section_emb_dim": section_emb_dim,
                        "include_Teufel_data": include_Teufel_data,
                        "lr": lr,
                        "dropout": dropout,
                        "num_of_epochs": num_of_epochs,
                        "batch_size": batch_size,
                        "ACC_STEP": ACC_STEP,
                        "n_seeds": len(seeds),

                        "mean_best_val_f1_micro": df_setting["best_val_f1_micro"].mean(),
                        "std_best_val_f1_micro": df_setting["best_val_f1_micro"].std(),

                        "mean_best_val_f1_macro": df_setting["best_val_f1_macro"].mean(),
                        "std_best_val_f1_macro": df_setting["best_val_f1_macro"].std(),

                        "mean_test_accuracy": df_setting["test_accuracy"].mean(),
                        "std_test_accuracy": df_setting["test_accuracy"].std(),

                        "mean_test_f1_micro": df_setting["test_f1_micro"].mean(),
                        "std_test_f1_micro": df_setting["test_f1_micro"].std(),

                        "mean_test_f1_macro": df_setting["test_f1_macro"].mean(),
                        "std_test_f1_macro": df_setting["test_f1_macro"].std(),

                        "mean_test_f1_weighted": df_setting["test_f1_weighted"].mean(),
                        "std_test_f1_weighted": df_setting["test_f1_weighted"].std(),

                        "mean_test_precision_micro": df_setting["test_precision_micro"].mean(),
                        "std_test_precision_micro": df_setting["test_precision_micro"].std(),

                        "mean_test_recall_micro": df_setting["test_recall_micro"].mean(),
                        "std_test_recall_micro": df_setting["test_recall_micro"].std(),

                        "mean_test_precision_macro": df_setting["test_precision_macro"].mean(),
                        "std_test_precision_macro": df_setting["test_precision_macro"].std(),

                        "mean_test_recall_macro": df_setting["test_recall_macro"].mean(),
                        "std_test_recall_macro": df_setting["test_recall_macro"].std(),
                    }

                    #Model-specific summary
                    model_summary_rows.append(summary_row)

                    #Global summary usefull if args.model == "all"
                    summary_rows.append(summary_row)

        #Save all per-seed rows for this model
        df_model_seeds = pd.DataFrame(model_seed_rows)
        data_repartition_key = args.data_repartition

        model_seed_csv_path = (
            f"{results_dir}/score_per_seed_all_configs_"
            f"model-{model_name}_"
            f"repartition-{data_repartition_key}_"
            f"window-{window_key}_"
            f"scicite-{augment_coco_with_scicite}.csv"
        )
        os.makedirs(os.path.dirname(model_seed_csv_path), exist_ok=True)
        df_model_seeds.to_csv(model_seed_csv_path, index=False)

        print(f"\nSaved all per-seed rows for {model_name} to: {model_seed_csv_path}")

        # Save averaged results for this model
        # One row per config: use_section / section_emb_dim / include_Teufel_data
        df_model_summary = pd.DataFrame(model_summary_rows)

        model_summary_csv_path = (
            f"{results_dir}/score_average_"
            f"model-{model_name}_"
            f"repartition-{data_repartition_key}_"
            f"window-{window_key}_"
            f"scicite-{augment_coco_with_scicite}.csv"
        )

        model_summary_json_path = (
            f"{results_dir}/score_average_"
            f"model-{model_name}_"
            f"repartition-{data_repartition_key}_"
            f"window-{window_key}_"
            f"scicite-{augment_coco_with_scicite}.json"
        )

        os.makedirs(os.path.dirname(model_summary_csv_path), exist_ok=True)
        df_model_summary.to_csv(model_summary_csv_path, index=False)
        df_model_summary.to_json(model_summary_json_path, orient="records", indent=4)

        print(f"\nSaved averaged scores for {model_name} to: {model_summary_csv_path}")
        print(f"Saved averaged scores for {model_name} to: {model_summary_json_path}")

        score_per_seed_json_path = (
            f"{results_dir}/score_per_seed_all_configs_"
            f"model_{model_name}_"
            f"repartition_{data_repartition_key}_"
            f"window_{window_key}_"
            f"scicite_{augment_coco_with_scicite}.json"
        )

        os.makedirs(os.path.dirname(score_per_seed_json_path), exist_ok=True)

        with open(score_per_seed_json_path, "w", encoding="utf-8") as f:
            json.dump(all_scores[model_name], f, indent=4)

        print(f"\nSaved full per-seed JSON for {model_name} to: {score_per_seed_json_path}")

  
    #Save final global CSVs only when all models are run together
    if args.model == "all":

        global_results_dir = "results_sci_cite/all_models"
        os.makedirs(global_results_dir, exist_ok=True)

        #Full per-seed results for all models
        df_all_seeds = pd.DataFrame(all_seed_rows)

        all_seed_csv_path = (
            f"{global_results_dir}/score_per_seed_all_models_"
            f"window_{window_key}_"
            f"scicite-{augment_coco_with_scicite}.csv"
        )

        os.makedirs(os.path.dirname(all_seed_csv_path), exist_ok=True)
        df_all_seeds.to_csv(all_seed_csv_path, index=False)

        print(f"\nSaved global per-seed CSV for all models to: {all_seed_csv_path}")

        #Averaged results for all models
        df_summary = pd.DataFrame(summary_rows)

        summary_csv_path = (
            f"{global_results_dir}/score_average_all_models_"
            f"window_{window_key}_"
            f"scicite-{augment_coco_with_scicite}.csv"
        )

        summary_json_path = (
            f"{global_results_dir}/score_average_all_models_"
            f"window_{window_key}_"
            f"scicite-{augment_coco_with_scicite}.json"
        )

        os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
        df_summary.to_csv(summary_csv_path, index=False)
        df_summary.to_json(summary_json_path, orient="records", indent=4)

        print(f"\nSaved averaged scores for all models to: {summary_csv_path}")
        print(f"Saved averaged scores for all models to: {summary_json_path}")

    else:
        print("\nSingle-model run detected. Skipped results_sci_cite/all_models export to avoid overwriting global files.")