#!/usr/bin/env python3
# Grupo de Biología Computacional, Universidad de Antioquia, Medellín, Colombia
"""
"""

import pandas as pd
import numpy as np
from Bio import SeqIO
from pathlib import Path


#from utils import converter as cv
from . import converter as cv 
#import converter as cv
from itertools import product
from sklearn.utils import shuffle

import tensorflow as tf
from keras.utils import to_categorical
from keras import backend as K
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from keras.models import Sequential
from keras.layers import Input, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers import Adam


def fasta_to_onehot(pos_path: str, neg_path: str) -> pd.DataFrame:
    """
    Convert positive and negative FASTA files into a DataFrame with
    headers, sequences, one-hot encoding, and labels
    """
    encoding = {
        "A": [1, 0, 0, 0, 0],
        "C": [0, 1, 0, 0, 0],
        "G": [0, 0, 1, 0, 0],
        "T": [0, 0, 0, 1, 0],
        "N": [0, 0, 0, 0, 0],
    }

    def onehotmatrix(seq: str) -> np.ndarray:
        """
        Convert a DNA sequence to a one-hot encoded NumPy array
        """
        mat = np.zeros((len(seq), 5), dtype=int)
        for i, nuc in enumerate(seq.upper()):
            mat[i] = encoding.get(nuc)
        return mat
    
    def read_fasta(fasta_path: str, label: int):
        """
        Read a FASTA file and return a list of (header, seq, onehot, label)
        """
        path= Path(fasta_path)
        if not path.exists():
            raise FileNotFoundError(f'File not found: {path}')
        
        records = []
        for record in SeqIO.parse(path, 'fasta'):
            seq = str(record.seq).upper()
            records.append((record.description, seq, onehotmatrix(seq), label))
        return records

    pos = read_fasta(pos_path, 1)
    neg = read_fasta(neg_path, 0)

    df = pd.DataFrame(pos + neg, columns=['header', 'sequence', 'encoding', 'label'])
    df = pd.sample(frac=1, random_state=42).reset_index(drop=True)

    return df



def training_process(X_train, y_train, n_folds, representation):

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # List of CNN architectures
    cnns = [DeepSplicer_regu, Spliceator, SpliceFinder]

      
    # Dictionary to hold results for each CNN
    results_dict = {cnn.__name__: [] for cnn in cnns}

    # Adjust input shape if necessary
    input_shape = X_train.shape[1:]
    if input_shape == (1024,):
        input_shape = (1024, 1)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Perform k-fold cross-validation over each CNN architecture
    for train_index, val_index in kf.split(X_train):
           
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        for cnn in cnns:
            epochs = 150
            model = cnn(input_shape)
            history = model.fit(X_train_fold, y_train_fold, epochs=epochs, batch_size=32, validation_data=(X_val_fold, y_val_fold), 
                      callbacks=[early_stop], verbose=1)
            trained_epochs = len(history.epoch)
            
            results = model.evaluate(X_val_fold, y_val_fold) #verbose=0

            # Make predictions and calculate F1 score
            y_val_pred = model.predict(X_val_fold, verbose =0)
            y_val_pred_classes = np.argmax(y_val_pred, axis=1)
            f1 = f1_score(np.argmax(y_val_fold, axis=1), y_val_pred_classes, average='weighted')

            # Store results for the current fold
            results_dict[cnn.__name__].append([trained_epochs] + list(results) + [f1])
            
    # Ad all results
    all_results = []
    for model_name, metrics in results_dict.items():
        for fold_metrics in metrics:
            all_results.append([model_name]+fold_metrics)
       
    df_all_results = pd.DataFrame(all_results, columns=['architecture', 'epochs_trained','loss', 'accuracy', 'precision', 'recall', 'f1_score'])

    return df_all_results



def load_data(train_neg, train_pos):

    # Concatenate the DataFrames df_negative and df_positive into a single DataFrame
    train_data = pd.concat([train_neg, train_pos], ignore_index = True)
    train_data = shuffle(train_data)
    #test_data = pd.concat([test_neg, test_pos], ignore_index = True)
  
    # Getting lists from the Data Frame
    sequences_list_train = train_data['encoding'].to_list()
    labels_list_train = train_data['label'].to_list() 

    # Transforming lists to Numpy Arrays and redimensioning features array
    X_train = np.array(sequences_list_train)
    y_train = np.array(labels_list_train)
    
    # Transforming labels in one-hot codification 
    y_train = to_categorical(y_train, num_classes=2)
 
    return np.array(X_train),np.array(y_train)

def f1_metric(y_true, y_pred):
    
    # Convert predictions in classes
    y_pred_classes = K.argmax(y_pred, axis=-1)
    y_true_classes = K.argmax(y_true, axis=-1)
    
    # Calculate F1 Score
    return f1_score(y_true_classes, y_pred_classes, average='weighted')

def recall_m(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision



def SpliceFinder(input_shape): #SpliceFinder - https://link.springer.com/article/10.1186/s12859-022-04971-w#Sec2 

    model = Sequential([
    
        Conv1D(filters=50, kernel_size=9, strides=1, activation='relu', padding='same', input_shape=input_shape),
        
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.3),
        
        Dense(2, activation='sigmoid')  
    ])

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy', 
                  metrics=[tf.keras.metrics.Precision(),'accuracy', tf.keras.metrics.Recall()])

    return model                


def DeepSplicer_regu(input_shape):

    model = Sequential([
    # First block of normalization
    BatchNormalization(input_shape=input_shape),
    Conv1D(filters=16, kernel_size=11, padding='same', kernel_regularizer=l1(0.0001)),
    Activation('relu'), 
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    
    BatchNormalization(),
    Conv1D(filters=32, kernel_size=11, padding='same', kernel_regularizer=l1(0.0001)),
    Activation('relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
        
    BatchNormalization(),
    Conv1D(filters=64, kernel_size=21, padding='same', kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)),
    Activation('relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
        
    BatchNormalization(),
    Conv1D(filters=64, kernel_size=41, padding='same', kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)),
    Activation('relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
        
    Flatten(),
    Dense(32, activation='relu', kernel_regularizer=l2(0.0001)),
    Dropout(0.5),
    Dense(16, activation='relu', kernel_regularizer=l2(0.0001)),
    Dropout(0.5) ,

    Dense(2, activation='sigmoid')
    
    ])
    
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  weighted_metrics=[precision_m,'accuracy', recall_m ])
    return model

def DeepSplicer(input_shape):

    model = Sequential([
    # First block of normalization
    BatchNormalization(input_shape=input_shape),
    Conv1D(filters=16, kernel_size=11, padding='same', kernel_regularizer=l1(0.00007)),
    Activation('relu'), 
    MaxPooling1D(pool_size=2),
    
    BatchNormalization(),
    Conv1D(filters=32, kernel_size=11, padding='same', kernel_regularizer=l1(0.00007)),
    Activation('relu'),
    MaxPooling1D(pool_size=2),
        
    BatchNormalization(),
    Conv1D(filters=64, kernel_size=21, padding='same', kernel_regularizer=l1(0.00007)),
    Activation('relu'),
    MaxPooling1D(pool_size=2),
        
    BatchNormalization(),
    Conv1D(filters=64, kernel_size=41, padding='same', kernel_regularizer=l1(0.00007)),
    Activation('relu'),
    MaxPooling1D(pool_size=2),
        
    Flatten(),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
        
    Dense(2, activation='sigmoid')
    
    ])
    
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['precision', 'accuracy', 'recall'])
    return model
    
def Spliceator(input_shape): #https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04471-3#Sec12 

    model = Sequential([
        Conv1D(filters=16, kernel_size=7, strides=1, activation='relu', padding='same', input_shape=input_shape),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(0.2),

        Conv1D(filters=32, kernel_size=6, strides=1, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(0.2),

        Conv1D(filters=64, kernel_size=6, strides=1, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(0.2),

        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.2),

        Dense(2, activation='sigmoid') 
    ])

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',  
                  metrics=[precision_m,'accuracy', recall_m ])  

    return model 


    with open(pos_path, 'r', encoding='utf-8') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            header = record.descriotion
            sequence = str(record.seq).upper()

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

            oh_rep = cv.onehotmatrix




def fasta2df_onehot(filename, type_label=True):

    df = pd.DataFrame(columns=['header', 'sequence', 'encoding', 'label'])

    with open(filename) as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            header = record.description.title()
            sequence = str(record.seq.upper())
            oh_rep = cv.onehotmatrix(record.seq.upper())

            if type_label:
                label = 1
            else:
                label = 0

            df.loc[len(df)] = [header, sequence, oh_rep, label]
    #df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df
{}