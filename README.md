# GramPred
Predict gram staining by using bacterial genomic sequences and machine learning model

# Usage 
python grampred.py --fasta genome.fa --func_file annotation_file

**Input**

--fasta genomic sequence in FASTA format 

--func_file(optional) annotation results of genes from the genomic sequence

**Output**

features.csv generated features 

predict.csv prediction result 

## Quick start
1. Prepare genomic sequence in FASTA format, eg. example.fa.
2. Install python and required packages

    python==3.11.7
   
    scikit-learn==1.5.1
  
    joblib==1.2.0
  
    Bio==1.7.1
  
3. Predict Gram Staining
   
    python grampred.py --fasta example.fa

_% This commond will predict the Gram Staining of example.fa using compositional features only._


## Add functional features into prediction
1. Prepare genomic sequence in FASTA format, eg. example.fa.
2. Install python and required packages

   python==3.11.7
   
    scikit-learn==1.5.1
  
    joblib==1.2.0
  
    Bio==1.7.1
   
3. Install prodigal and predict genes from the genomic sequence

    conda install prodigal

    /path/to/prodigal -i example.fa -o example.gff -a example.faa -d example.fna -c -m -p meta
   
4. Install eggnog-mapper and annotate the predicted genes

   _% Install eggnog-mapper using conda_
   
   conda install -c bioconda eggnog-mapper

   _% Download bacterial database _

   python /path/to/eggnog-mapper/download_eggnog_data.py -y -f -H -d 2 --data_dir /path/to/eggnog

   _% Annotation_ 

   python /path/to/eggnog-mapper/emapper.py --cpu 0 -i example.fna --itype CDS --output example -d bact --data_dir /path/to/eggnog-mapper/eggnog-mapper-master/data

   _% Extract GO, COG and KEGG results _

   cat example.emapper.annotations |sed '/#.*/d' |cut -f1,7,10,12 |awk -F"\t" '{print $1"\t"$2"\t"$3"\t"$4}' |sed '/-: -/d'|sort -k 1.3n |uniq |sed '1i\query\tCOG_category\tGOs\tKEGG_ko' >example.filtered
   
5. Predict gram staining by GramPred

   python grampred.py --fasta example.fa --func_file example.filtered

## Contact us
liuyun313@jlu.edu.cn
