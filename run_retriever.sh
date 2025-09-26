#!/bin/bash

python retrieval/passage_retrieval.py \
    --model_name_or_path facebook/contriever-msmarco --passages retrieval/corpus/psgs_w100.tsv \
    --passages_embeddings "retrieval/corpus/wikipedia_embeddings/*" \
    --query "Where is the Eiffel Tower?"  \
    --n_docs 5