#!/usr/bin/env bash
datasets=$1
for folder in ${datasets}/*; do
  mv ${folder}/evidence.txt ${folder}/unsorted_evidence.txt
  cat <( head -n 1 < ${folder}/unsorted_evidence.txt )\
      <( tail -n +2 < ${folder}/unsorted_evidence.txt | sort -k22 -t$'\t' )\
      > ${folder}/evidence.txt
done
