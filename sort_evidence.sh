#!/usr/bin/env bash
datasets=$1

for folder in ${datasets}/*/; do
  mv ${folder}/evidence.txt ${folder}/unsorted_evidence.txt

  colpos=$(head -n 1 ${folder}/unsorted_evidence.txt \
          | tr "\t" "\n" | grep -n "Raw file" | cut -d ":" -f1)

  cat <( head -n 1 < ${folder}/unsorted_evidence.txt )\
      <( tail -n +2 < ${folder}/unsorted_evidence.txt | sort -k${colpos},${colpos} -t$'\t' )\
      > ${folder}/evidence.txt
done
