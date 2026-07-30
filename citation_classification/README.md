# Classification of rhetorical citation functions by discipline (12 classes)

This repository contains several files related to the classification of rhetorical citation functions across different disciplines.

## Repository Contents

### `PD100cit`

This folder contains:  
- **`100_citation_sample.csv`**: A random sample of 100 citations from the *Pear Decline* (PD100cit) corpus, including their contexts and annotations.  
- **`annotation_guidelines.pdf`**: Annotation guidelines used as a reference for PD100cit annotations, with examples from ecology.

### `script`

This folder contains several scripts for citation classification:

- **Scripts:**  
  - `finetune_for_citation_classification.py`: Main script to fine-tune a language model (**BioBERT, SciBERT, RoBERTa, BioLinkBERT**), specified via arguments, including the citation context window size.  
  - `citation_classifier.py`: Classifier and training/validation functions.  
  - `get_citation_sequence.py`: Extracts citation sequences from both corpora, output as lists.  
  - `utils.py`: Helper functions used by `finetune_for_citation_classification.py`.

- **Prompt Instructions:**  
  - `Prompt_instructions_citation_classification.pdf`: Instructions used for designing GPT-4-based prompts for citation classification.

### `essai_modele_Jiang`

This subfolder reproduces the model from Jiang & Chen (2023):

- `fine_tune_several_vectors.py`: Main script for training the Jiang & Chen model.  
- `citation_classifier.py`: Classifier and training/validation functions, called by `fine_tune_several_vectors.py`.  
- `prepare_data.py`: Extracts citation sequences and their contexts, delimited by `@`, output as lists. Used by `fine_tune_several_vectors.py`.

---------------------------------------------------------------------------------------
#Version française
    
# Classification des fonctions rhétoriques des citations par discipline

Ce dépôt contient plusieurs fichiers relatifs à la classification des fonctions rhétoriques des citations à travers différentes disciplines.

## Contenu du dépôt

### `PD100cit`
Ce dossier contient :  
- **`100_citation_sample.csv`** : 100 citations tirées aléatoirement du corpus *Pear Decline* (PD100cit), avec leurs contextes et annotations.  
- **`annotation_guidelines.pdf`** : Guide d'annotation utilisé comme référence pour les annotations de PD100cit, avec des exemples en écologie.  

### `script`
Ce dossier contient plusieurs scripts pour la classification des citations :  

- **Scripts :**  
  - `finetune_for_citation_classification.py` : Script principal pour fine-tuner un modèle de langue (**BioBERT, SciBERT, RoBERTa, BioLinkBERT**), à définir en argument, avec la taille de la fenêtre de contexte.  
  - `citation_classifier.py` : Classifieur et fonctions d'entraînement et de validation.  
  - `get_citation_sequence.py` : Extraction des séquences de citations des deux corpus, sorties sous forme de listes.  
  - `utils.py` : Fonctions auxiliaires utilisées par `finetune_for_citation_classification.py`.  

- **Instruction pour le prompt de GPT-4** : Fichier Prompt_instructions_citation_classification.pdf

- **Sous-dossier `essai_modele_Jiang`**  
  - `fine_tune_several_vectors.py` : Script principal pour reproduire le modèle de **Jiang & Chen (2023)**.  
  - `citation_classifier.py` : Classifieur et fonctions d'entraînement et de validation, appelé par `fine_tune_several_vectors.py`.  
  - `prepare_data.py` : Extraction des séquences de citations et de leurs contextes, avec délimitation par `@`. Sorties sous forme de listes. Script appelé par `fine_tune_several_vectors.py`.  
