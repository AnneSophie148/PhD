# Split sentences


## Directory content

- `split-sentences.plan`: AlvisNLP plan for splitting sentences.
- `ner-taxa.plan`: AlvisNLP plan for searching taxon names.
- `xml2alvisnlp.xslt`: stylesheet for reading XML files in AlvisNLP.
- `sample-docs/`: directory containing sample documents.


## Requirements

- Install AlvisNLP.
- Files: `taxa+id_full.txt` and `taxa+id_full.trie`.


## Command-line

```
alvisnlp split-sentences.plan -alias input INPUT -alias output OUTPUT
```

where *INPUT* is the path to the document to process, and *OUTPUT* the path to the sentence file.
