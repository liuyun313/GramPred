# GramPred
Predict gram staining by using bacterial genomic sequences and machine learning model

## Quick start
1. Prepare genomic sequence in FASTA format, eg. example.fa.
2. Install python and required packages

    python==3.11.7
   
    scikit-learn==1.5.1
  
    joblib==1.2.0
  
    Bio==1.7.1
  
4. Predict Gram Staining
   
    python grampred.py --fasta example.fa

_% This commond will predict the Gram Staining of example.fa using compositional features only._


## Add functional features into prediction
1. Prepare genomic sequence in FASTA format, eg. example.fa.
2. Install python and required packages

   python==3.11.7
   
    scikit-learn==1.5.1
  
    joblib==1.2.0
  
    Bio==1.7.1
   
4. Install prodigal and predict genes from the genomic sequence

    conda install prodigal

    /path/to/prodigal -i example.fa -o example.gff -a example.faa -d example.fna -c -m -p meta
   
5. Install eggnog-mapper and annotate the predicted genes

    % Install eggnog-mapper using conda
   
   conda install -c bioconda eggnog-mapper

   % Download bacterial database 

   python /path/to/eggnog-mapper/download_eggnog_data.py -y -f -H -d 2 --data_dir /path/to/eggnog

   % Annotation 

   python /path/to/eggnog-mapper/emapper.py --cpu 0 -i example.fna --itype CDS --output example -d bact --data_dir /path/to/eggnog-mapper/eggnog-mapper-master/data

   % Extract GO, COG and KEGG results 

   cat example.emapper.annotations |sed '/#.*/d' |cut -f1,7,10,12 |awk -F"\t" '{print $1"\t"$2"\t"$3"\t"$4}' |sed '/-: -/d'|sort -k 1.3n |uniq |sed '1i\query\tCOG_category\tGOs\tKEGG_ko' >example.filtered
   
8. Predict gram staining by GramPred

   python grampred.py --fasta example.fa --func_file example.filtered

