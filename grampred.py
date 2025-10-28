# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 10:20:34 2025

@author: 58211
"""


import os
import pandas as pd
from joblib import load
import argparse
from Bio import SeqIO
import numpy as np
#from torch import nn
from itertools import tee, product

directory = os.getcwd()

# In[] input parameters

# 创建解析器
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument("--fasta", type=str, required=True, help="genome file")
#parser.add_argument("--func", action='store_true', help="use functional features or not")
parser.add_argument("--func_file", type=str, help="annotation file")

# 解析参数
args = parser.parse_args()

# In[] computing kmer

def window(seq,n):
    els = tee(seq,n)
    for i,el in enumerate(els):
        for _ in range(i):
            next(el, None)
    return zip(*els)

def generate_feature_mapping(kmer_len):
    BASE_COMPLEMENT = {"A":"T","T":"A","G":"C","C":"G"}
    kmer_hash = {}
    counter = 0
    for kmer in product("ATGC",repeat=kmer_len):
        kmer = ''.join(kmer)
        kmer_hash[kmer] = counter
        counter += 1
        '''
        if kmer not in kmer_hash:
            kmer_hash[kmer] = counter
            rev_compl = ''.join([BASE_COMPLEMENT[x] for x in reversed(kmer)])
            kmer_hash[rev_compl] = counter
            counter += 1
        '''
    return kmer_hash


def count_kmer(filename, kmer_len):
    
    kmer_dict = generate_feature_mapping(kmer_len)

    kmers = np.ones((1,max(kmer_dict.values())+1))
    
    contigs_id = []
    

    nucl_dict = SeqIO.to_dict(SeqIO.parse(filename,"fasta"), key_function = lambda rec: rec.description) #keep whitespace in FASTA header
    
    if len(nucl_dict) == 1:

        ids = list(nucl_dict.values())[0]

        header, seq = ids.description, ids.seq

        contigs_id.append(header)

        #Initialize feature vectors. NxD where N is number of datapoints, D is number of dimentions

        for kmer_tuple in window(str(seq).upper(), kmer_len):
            try:
                kmers[0, kmer_dict["".join(kmer_tuple)]] += 1
            except KeyError:
                continue
        
    else:

        is_added = False

        for header in nucl_dict.keys():

            if 'plasmid' not in header:
                
                if is_added == False:
                    contigs_id.append(nucl_dict[header].description)     
                    is_added = True
                
                seq = nucl_dict[header].seq

                #Initialize feature vectors. NxD where N is number of datapoints, D is number of dimentions
                
                for kmer_tuple in window(str(seq).upper(), kmer_len):
                    try:
                        kmers[0, kmer_dict["".join(kmer_tuple)]] += 1
                    except KeyError:
                        continue
    
    kmers = pd.DataFrame(kmers, columns=kmer_dict.keys())
    return kmers


kmers = count_kmer(args.fasta, 4)


# In[] extract functional features
def annotation_stat(filename):

    GOs = pd.read_csv('./models/goid_name', sep='\t', header=None, index_col=0)

    anno_dict = {}

    with open(filename) as f:

        data_list = f.readlines()
        num_last = 100

        for i in range(1, len(data_list)):
            
            line = data_list[i].split('\t')
            x = line[0].split('_')
            num = int(x[-1])            
            
            if i == 1:
                species = x[0][4:] + '_' + x[1]
                anno_dict[species] = {}
            elif num < num_last:
                species = x[0][4:] + '_' + x[1]
                anno_dict[species] = {}

            num_last = num
            
            # the total number of genes per species
            anno_dict['Total'] = anno_dict.get('Total', 0) +1 
            
            # genes distribution annotated by COG
            anno_dict[line[1][0]] = anno_dict.get(line[1][0], 0) +1 

            if anno_dict[line[1][0]] != '-':
                anno_dict['cog'] = anno_dict.get('cog', 0) +1 
            
            # genes distribution annotated by Go
            if line[2] != '-':
                anno_dict['go'] = anno_dict.get('go', 0) +1 

                gos = line[2].strip().split(',')   ## Total genes number annotated to GO
                ## genes number of 3 GO domains
                for go in gos:
                    if go in GOs.index:
                        name_space = GOs.loc[go,2]
                        anno_dict[name_space] = anno_dict.get(name_space, 0) + 1
            
            # genes distribution annotated by KEGG
            a=line[3].strip()
            if a != '-':
                anno_dict['KEGG'] = anno_dict.get('KEGG', 0) +1 

    anno = pd.DataFrame(columns=['label', 'value'])

    anno.loc[len(anno)] = ['cog', anno_dict.get('cog',0)/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['go', anno_dict.get('go',0)/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['kegg', anno_dict.get('KEGG',0)/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['INFORMATION STORAGE AND PROCESSING', (anno_dict.get('J', 0) + anno_dict.get('K', 0) + anno_dict.get('L', 0))/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['CELLULAR PROCESSES AND SIGNALING', (anno_dict.get('D', 0) + anno_dict.get('M', 0) + anno_dict.get('N', 0)+anno_dict.get('O', 0) + anno_dict.get('M', 0) + anno_dict.get('U', 0) + anno_dict.get('V', 0))/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['METABOLISM', (anno_dict.get('C', 0) + anno_dict.get('E', 0) + anno_dict.get('F', 0)+anno_dict.get('G', 0) + anno_dict.get('H', 0) + anno_dict.get('I', 0) + anno_dict.get('P', 0) + anno_dict.get('Q', 0))/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['molecular_function', (anno_dict.get('molecular_function', 0))/(anno_dict.get('molecular_function', 0)+anno_dict.get('biological_process', 0)+anno_dict.get('cellular_component', 0))] 
    anno.loc[len(anno)] = ['biological_process', (anno_dict.get('biological_process', 0))/(anno_dict.get('molecular_function', 0)+anno_dict.get('biological_process', 0)+anno_dict.get('cellular_component', 0))] 
    anno.loc[len(anno)] = ['cellular_component', (anno_dict.get('cellular_component', 0))/(anno_dict.get('molecular_function', 0)+anno_dict.get('biological_process', 0)+anno_dict.get('cellular_component', 0))] 
    anno.loc[len(anno)] = ['C', (anno_dict.get('C', 0))/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['D', (anno_dict.get('D', 0))/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['E', (anno_dict.get('E', 0))/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['F', (anno_dict.get('F', 0))/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['G', (anno_dict.get('G', 0))/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['H', (anno_dict.get('H', 0))/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['I', (anno_dict.get('I', 0))/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['J', (anno_dict.get('J', 0))/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['K', (anno_dict.get('K', 0))/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['L', (anno_dict.get('L', 0))/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['M', (anno_dict.get('M', 0))/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['N', (anno_dict.get('N', 0))/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['O', (anno_dict.get('O', 0))/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['P', (anno_dict.get('P', 0))/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['Q', (anno_dict.get('Q', 0))/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['T', (anno_dict.get('T', 0))/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['U', (anno_dict.get('U', 0))/anno_dict.get('Total',0)] 
    anno.loc[len(anno)] = ['V', (anno_dict.get('V', 0))/anno_dict.get('Total',0)] 

    anno.set_index(["label"], inplace=True)
    return anno 


# In[] predict
result = pd.DataFrame(columns=['genome', 'gram'])

df2 = kmers.sum(axis=1)    

kmer = kmers.div(df2, axis='rows')
kmer = kmer.T
kmer.columns = ['value']

if args.func_file:    
    clf = load('./models/rf_ensem.joblib')    

    anno = annotation_stat(args.func_file)
    
    data = pd.concat([kmer,anno],axis=0)
    
    data.to_csv('features.csv')

    data = data.T     

    y = clf.predict(data)

    result.loc[len(result)] = [args.fasta, y] 

    result.set_index('genome', inplace=True)

    result.to_csv('predict.csv')


else:
    clf = load('./models/rf_kmer.joblib')    

    kmer.to_csv('features.csv')

    kmer = kmer.T

    y = clf.predict(kmer)

    result.loc[len(result)] = [args.fasta, y] 

    result.set_index('genome', inplace=True)

    result.to_csv('predict.csv')







