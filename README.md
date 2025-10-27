# GramPred
Predict gram staining by using bacterial genomic sequences and machine learning model

## Gene prediction by prodigal
/path/to/prodigal -i example.fa -o example.gff -a example.faa -d example.fna -c -m -p meta

## Gene annotation by emapper
1. python /path/to/eggnog-mapper/eggnog-mapper-master/emapper.py --cpu 0 -i example.fna --itype CDS --output example -d bact --data_dir /path/to/eggnog-mapper/eggnog-mapper-master/data

2. cat example.emapper.annotations |sed '/#.*/d' |cut -f1,7,10,12 |awk -F"\t" '{print $1"\t"$2"\t"$3"\t"$4}' |sed '/-: -/d'|sort -k 1.3n |uniq |sed '1i\query\tCOG_category\tGOs\tKEGG_ko' >example.filtered

## Predict gram staining by GramPred
