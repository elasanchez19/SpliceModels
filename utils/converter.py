#!/usr/bin/env python3

"""
Convert a DNA sequence into a numerical representation.
"""

import pandas as pd
import numpy as np

def seq2matrix(sequence, normalize=True):

    '''
    Converts a DNA sequence into a physicochemical properties matrix

    Parameters
    ----------
    sequence : str

    Returns
    -------
    numpy.ndarray
    '''

    sequence = sequence.upper()

    if normalize:
        if 'U' in sequence:
            rna = pd.read_csv("utils/scaled_rna.csv", index_col=0)
            temp_list = [rna.loc[base].to_list() if base in ('A', 'C', 'G', 'U') else rna.loc['ZERO'].to_list() for base in sequence]
        else:
            dna = pd.read_csv("utils/scaled_dna.csv", index_col=0)
            temp_list = [dna.loc[base].to_list() if base in ('A', 'C', 'G', 'T') else dna.loc['ZERO'].to_list() for base in sequence]
    else:
        if 'U' in sequence:
            rna = pd.read_csv("utils/rna.csv", index_col=0)
            temp_list = [rna.loc[base].to_list() if base in ('A', 'C', 'G', 'U') else rna.loc['ZERO'].to_list() for base in sequence]
        else:
            dna = pd.read_csv("utils/dna.csv", index_col=0)
            temp_list = [dna.loc[base].to_list() if base in ('A', 'C', 'G', 'T') else dna.loc['ZERO'].to_list() for base in sequence]


    return np.array(temp_list, dtype=np.float32)

def seq2matrix_pc(sequence, normalize=True, excluded_property=None):

    sequence = sequence.upper()

    if normalize:
        if 'U' in sequence:
            df = pd.read_csv("utils/scaled_rna.csv", index_col=0)
        else:
            df = pd.read_csv("utils/scaled_dna.csv", index_col=0)
           
    else:
        if 'U' in sequence:
            df = pd.read_csv("utils/rna.csv", index_col=0)
        else:
            df = pd.read_csv("utils/dna.csv", index_col=0)


    if excluded_property is not None: 
        if isinstance(excluded_property, str):
            excluded_property = [excluded_property]
        for prop in excluded_property:
            if prop not in df.columns:
                print(df.columns)
                raise ValueError(f"The property '{prop}' does not exist")
            
        df = df.drop(columns=excluded_property)
        
    temp_list = [df.loc[base].to_list() if base in df.index else df.loc['ZERO'].to_list() for base in sequence]

    return np.array(temp_list, dtype=np.float32)


def onehotmatrix(sequence):
    '''
    Converts a DNA sequence into a one-hot encoded matrix

    Parameters
    ----------
    sequence : str
        DNA sequence to convert

    Returns
    -------
    numpy.ndarray
        One-hot encoded matrix of shape (len(sequence), 4)
    '''
    seq_data = []

    # Define encoding for each nucleotide
    encoding = {
        'A': [1, 0, 0, 0, 0],
        'C': [0, 1, 0, 0, 0],
        'G': [0, 0, 1, 0, 0],
        'T': [0, 0, 0, 1, 0],
        'N': [0, 0, 0, 0, 0]  # Encoding for unknown nucleotides
    }
    
    # Convert each nucleotide to its one-hot encoding
    seq_data = [encoding.get(nucleotide, encoding['N']) for nucleotide in sequence]
    
    # Convert list to numpy array
 
    return np.array(seq_data)

def seq2kmer(sequence, kmers_list):

    '''
    Converts a DNA sequence into k-mer representation.

    Parameters:
    - sequence (str): The input DNA sequence.
    - k (int, optional): The length of k-mers (default is 6).

    Returns:
    - numpy.array: An array with kmer counts.
    '''

    kmer_counts = []

    for kmer in kmers_list:
        counts = sequence.count(kmer)
        kmer_counts.append(counts)

    return np.array(kmer_counts)


