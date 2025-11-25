#!/bin/bash

PATH_CORPUS="$1"
AlvisNL="../alvisnlp_install/bin/alvisnlp"

for doc in "$PATH_CORPUS"/*.xml; do
    source="$(basename "$PATH_CORPUS")"
    doi="$(basename "$doc" .xml)"

    output_dir="outputAlvisNLP/$source"
    mkdir -p "$output_dir"

    "$AlvisNL" split-sentences.plan \
        -alias input "$doc" \
        -alias output "$output_dir/${doi}_references_citing.csv"
done