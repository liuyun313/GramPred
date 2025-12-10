# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
import os
from itertools import tee, product
from Bio import SeqIO 
class ModalityTower(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, hidden_mult: int = 2, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        hidden_dim = embed_dim * hidden_mult
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, embed_dim))
        layers.append(nn.BatchNorm1d(embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(query=query, key=context, value=context)
        x = self.norm1(query + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

class FusionBinaryClassifier(nn.Module):
    def __init__(self, kmer_dim: int, anno_dim: int, embed_dim: int = 128, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.kmer_tower = ModalityTower(input_dim=kmer_dim, embed_dim=embed_dim, hidden_mult=2, num_layers=2, dropout=dropout)
        self.anno_tower = ModalityTower(input_dim=anno_dim, embed_dim=embed_dim, hidden_mult=2, num_layers=2, dropout=dropout)
        self.cross_kmer_to_anno = CrossAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.cross_anno_to_kmer = CrossAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True, activation="gelu")
        self.fusion_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, kmer_feat: torch.Tensor, anno_feat: torch.Tensor) -> torch.Tensor:
        kmer_h = self.kmer_tower(kmer_feat)
        anno_h = self.anno_tower(anno_feat)
        kmer_tok = kmer_h.unsqueeze(1)
        anno_tok = anno_h.unsqueeze(1)
        kmer_ctx = self.cross_kmer_to_anno(query=kmer_tok, context=anno_tok)
        anno_ctx = self.cross_anno_to_kmer(query=anno_tok, context=kmer_tok)
        tokens = torch.cat([kmer_ctx, anno_ctx], dim=1)
        fused_tokens = self.fusion_encoder(tokens)
        fused = fused_tokens.mean(dim=1)
        fused = self.fusion_norm(fused)
        logit = self.classifier(fused).squeeze(-1)
        return logit

class KmerOnlyBinaryClassifier(nn.Module):
    def __init__(self, kmer_dim: int, embed_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.kmer_tower = ModalityTower(input_dim=kmer_dim, embed_dim=embed_dim, hidden_mult=2, num_layers=2, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, kmer_feat: torch.Tensor, anno_feat: torch.Tensor = None) -> torch.Tensor:
        h = self.kmer_tower(kmer_feat)
        h = self.norm(h)
        logit = self.classifier(h).squeeze(-1)
        return logit

class AnnoOnlyBinaryClassifier(nn.Module):
    def __init__(self, anno_dim: int, embed_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.anno_tower = ModalityTower(input_dim=anno_dim, embed_dim=embed_dim, hidden_mult=2, num_layers=2, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, kmer_feat: torch.Tensor = None, anno_feat: torch.Tensor = None) -> torch.Tensor:
        h = self.anno_tower(anno_feat)
        h = self.norm(h)
        logit = self.classifier(h).squeeze(-1)
        return logit

def window(seq, n):
    els = tee(seq, n)
    for i, el in enumerate(els):
        for _ in range(i):
            next(el, None)
    return zip(*els)

def generate_feature_mapping(kmer_len):
    kmer_hash = {}
    counter = 0
    for kmer in product("ATGC", repeat=kmer_len):
        kmer = ''.join(kmer)
        kmer_hash[kmer] = counter
        counter += 1
    return kmer_hash

def process_fasta_to_kmer(filename, kmer_len=4):
    """
    读取 fasta 文件并计算 K-mer 频率 (1 x 256)
    """
    print(f"Processing FASTA: {filename}")
    kmer_dict = generate_feature_mapping(kmer_len)
    
    kmers = np.ones((1, max(kmer_dict.values()) + 1)) 
    
    nucl_dict = SeqIO.to_dict(SeqIO.parse(filename, "fasta"))
    
    if len(nucl_dict) == 0:
        raise ValueError(f"No sequences found in {filename}")

    for header in nucl_dict:
        if 'plasmid' not in header.lower(): 
            seq = nucl_dict[header].seq
            for kmer_tuple in window(str(seq).upper(), kmer_len):
                kmer_str = "".join(kmer_tuple)
                if kmer_str in kmer_dict:
                    kmers[0, kmer_dict[kmer_str]] += 1
    
    kmer_df = pd.DataFrame(kmers, columns=kmer_dict.keys())
    
    kmer_sum = kmer_df.sum(axis=1).replace(0, 1e-8)
    kmer_df = kmer_df.div(kmer_sum, axis=0)
    
    return kmer_df

def process_func_file_to_anno(filename, goid_map_path):
    """
    读取功能注释文件并提取 27 维特征向量
    """
    print(f"Processing Annotation File: {filename}")
    
    if not os.path.exists(goid_map_path):
        raise FileNotFoundError(f"GO ID mapping file not found at: {goid_map_path}")
        
    GOs = pd.read_csv(goid_map_path, sep='\t', header=None, index_col=0)
    
    anno_dict = {}
    
    with open(filename) as f:
        data_list = f.readlines()
        
        for i in range(1, len(data_list)):
            line = data_list[i].strip().split('\t')
            if len(line) < 4: continue
            
            anno_dict['Total'] = anno_dict.get('Total', 0) + 1
            
            cog_val = line[1][0]
            anno_dict[cog_val] = anno_dict.get(cog_val, 0) + 1
            if cog_val != '-':
                anno_dict['cog'] = anno_dict.get('cog', 0) + 1
            
            go_val = line[2]
            if go_val != '-':
                anno_dict['go'] = anno_dict.get('go', 0) + 1
                gos = go_val.split(',')
                for go in gos:
                    go = go.strip()
                    if go in GOs.index:
                        name_space = GOs.loc[go, 2]
                        anno_dict[name_space] = anno_dict.get(name_space, 0) + 1
            
            kegg_val = line[3].strip()
            if kegg_val != '-':
                anno_dict['KEGG'] = anno_dict.get('KEGG', 0) + 1

    anno = pd.DataFrame(columns=['label', 'value'])
    total = anno_dict.get('Total', 1e-8) 
    
    def get_cnt(key): return anno_dict.get(key, 0)
    
    anno.loc[len(anno)] = ['cog', get_cnt('cog')/total]
    anno.loc[len(anno)] = ['go', get_cnt('go')/total]
    anno.loc[len(anno)] = ['kegg', get_cnt('KEGG')/total]
    
    info_store = get_cnt('J') + get_cnt('K') + get_cnt('L')
    cell_proc  = get_cnt('D') + get_cnt('M') + get_cnt('N') + get_cnt('O') + get_cnt('M') + get_cnt('U') + get_cnt('V')
    metabolism = get_cnt('C') + get_cnt('E') + get_cnt('F') + get_cnt('G') + get_cnt('H') + get_cnt('I') + get_cnt('P') + get_cnt('Q')
    
    anno.loc[len(anno)] = ['INFORMATION STORAGE AND PROCESSING', info_store/total]
    anno.loc[len(anno)] = ['CELLULAR PROCESSES AND SIGNALING', cell_proc/total]
    anno.loc[len(anno)] = ['METABOLISM', metabolism/total]
    
    mf = get_cnt('molecular_function')
    bp = get_cnt('biological_process')
    cc = get_cnt('cellular_component')
    go_denom = mf + bp + cc + 1e-8
    
    anno.loc[len(anno)] = ['molecular_function', mf/go_denom]
    anno.loc[len(anno)] = ['biological_process', bp/go_denom]
    anno.loc[len(anno)] = ['cellular_component', cc/go_denom]
    
    cog_cats = ['C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','T','U','V']
    for cat in cog_cats:
        anno.loc[len(anno)] = [cat, get_cnt(cat)/total]

    anno.set_index(["label"], inplace=True)
    anno_vec = anno.T # shape [1, 27]
    anno_vec = anno_vec.astype(float)
    return anno_vec


def main():
    parser = argparse.ArgumentParser(description="Raw Data Processing & Inference")
    # 输入文件
    parser.add_argument("--fasta", type=str, default=None, help="Input Genome FASTA file")
    parser.add_argument("--func_file", type=str, default=None, help="Input Annotation file (tab-separated)")
    
    # 配置
    parser.add_argument("--goid_map", type=str, default="./models/goid_name", help="Path to GO ID mapping file")
    parser.add_argument("--model_dir", type=str, default=".", help="Directory containing .pt weights")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    X_kmer = None
    X_anno = None
    
    df_kmer_processed = None
    df_anno_processed = None
    
    genome_id = "Unknown"

    has_fasta = args.fasta is not None
    has_func = args.func_file is not None

    if not has_fasta and not has_func:
        print("Error: Provide at least --fasta or --func_file")
        sys.exit(1)
        
    if has_fasta:
        genome_id = args.fasta 
        try:
            df_kmer_processed = process_fasta_to_kmer(args.fasta) 
            X_kmer = torch.from_numpy(df_kmer_processed.values).float().to(device)
        except Exception as e:
            print(f"Error processing FASTA: {e}")
            sys.exit(1)
            
    if has_func:
        if not has_fasta:
            genome_id = args.func_file
        try:
            df_anno_processed = process_func_file_to_anno(args.func_file, args.goid_map) 
            X_anno = torch.from_numpy(df_anno_processed.values).float().to(device)
        except Exception as e:
            print(f"Error processing Func File: {e}")
            sys.exit(1)

    model = None
    weight_file = ""

    if has_fasta and has_func:
        print("\n[Mode]: Dual-Modal (Fusion)")
        weight_file = "best_fusion.pt"
        model = FusionBinaryClassifier(
            kmer_dim=256, anno_dim=27, 
            embed_dim=128, num_heads=2, num_layers=2, dropout=0.0
        )
    
    elif has_fasta and not has_func:
        print("\n[Mode]: Single-Modal (Kmer Only)")
        weight_file = "best_kmer.pt"
        model = KmerOnlyBinaryClassifier(kmer_dim=256, embed_dim=128, dropout=0.0)
        
    elif has_func and not has_fasta:
        print("\n[Mode]: Single-Modal (Anno Only)")
        weight_file = "best_anno.pt"
        model = AnnoOnlyBinaryClassifier(anno_dim=27, embed_dim=128, dropout=0.0)

    weight_path = os.path.join(args.model_dir, weight_file)
    if not os.path.exists(weight_path):
        print(f"Error: Weight file not found: {weight_path}")
        sys.exit(1)
        
    print(f"Loading weights: {weight_path}")
    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
    except RuntimeError as e:
        print(f"Weight loading error: {e}")
        print("Tip: Ensure the model architecture matches the weights.")
        sys.exit(1)

    model.to(device)
    model.eval()

    # --- 3. 推理 ---
    print("Running inference...")
    with torch.no_grad():
        if has_fasta and has_func:
            logits = model(X_kmer, X_anno)
        elif has_fasta:
            logits = model(X_kmer, None)
        elif has_func:
            logits = model(None, X_anno)
    
    probs = torch.sigmoid(logits).cpu().item()
    pred = int(probs >= 0.5)
    
    print(f"\nResult for {genome_id}:")
    # print(f"Probability: {probs:.4f}")
    print(f"Prediction:{'gram+' if pred==1 else 'gram-'}")
    pred = 'gram+' if pred == 1 else 'gram-'
    feat_df = None

    def transpose_and_rename(df):
        df_t = df.T
        df_t.columns = ['value']  
        return df_t

    if has_fasta and has_func:
        kmer_T = transpose_and_rename(df_kmer_processed)
        anno_T = transpose_and_rename(df_anno_processed)
        feat_df = pd.concat([kmer_T, anno_T], axis=0)
        
    elif has_fasta:
        feat_df = transpose_and_rename(df_kmer_processed)
        
    elif has_func:
        feat_df = transpose_and_rename(df_anno_processed)
        
    if feat_df is not None:
        feat_df.to_csv('features.csv')
        print("Saved features.csv")
    result = pd.DataFrame(columns=['genome', 'gram'])
    result.loc[0] = [genome_id, pred]
    result.set_index('genome', inplace=True)
    result.to_csv('predict.csv')
    print("Saved predict.csv")

if __name__ == "__main__":
    main()
