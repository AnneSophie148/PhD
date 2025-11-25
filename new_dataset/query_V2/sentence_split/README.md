# Split sentences


## Directory content

- `split-sentences.plan`: AlvisNLP plan for splitting sentences.
- `ner-taxa.plan`: AlvisNLP plan for searching taxon names.
- `xml2alvisnlp.xslt`: stylesheet for reading XML files in AlvisNLP.
- `sample-docs/`: directory containing sample documents.
- `execute_alvisNLP.sh` : Bash script to exectue AlvisNLP plan on documents from a folder


## Requirements

- Install AlvisNLP.
- Files: `taxa+id_full.txt` and `taxa+id_full.trie`.


## Command-line split-sentences

```
alvisnlp split-sentences.plan -alias input INPUT -alias output OUTPUT
```

where *INPUT* is the path to the document to process, and *OUTPUT* the path to the sentence file.

## Command-line bash script

Execute on a folder for a source (Elsevier, PMC, PlosOne). Outputs a folder with the source name in outputAlvisNLP and generates a csv per document (named by the doi). 

```
bash execute_alvisNLP.sh sample-docs/PMC
```
