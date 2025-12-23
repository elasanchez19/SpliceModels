#!/usr/bin/env python3

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import auc
from Bio import SeqIO
from pathlib import Path
from . import converter as cv 
from itertools import product
from sklearn.utils import shuffle
from keras.utils import to_categorical
from keras import backend as K
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from keras.models import Sequential
from keras.layers import Input, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers import Adam
from sklearn.model_selection import KFold


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
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def training_process(df, n_folds, cnns):

    X = np.array(df['encoding'].tolist())
    y = df['label'].values

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    input_shape = X.shape[1:]

    evaluation_results = {cnn.__name__: [] for cnn in cnns}
    training_histories = {cnn.__name__: [] for cnn in cnns}
    roc_curves = {cnn.__name__: [] for cnn in cnns}
    confusion_matrixs = {cnn.__name__: [] for cnn in cnns}

    for cnn in cnns:
        print(f'\n--- Training {cnn.__name__} ---')

        fold_metrics = []
        all_true = []
        all_pred = []

        for fold, (train_index, val_index) in enumerate(kf.split(X)):
            print(f'\nFold {fold+1}/{n_folds}')
        
            X_train_fold, X_val_fold = X[train_index], X[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            model = cnn(input_shape)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'precision', 'recall', 'auc'])
            history = model.fit(X_train_fold, y_train_fold, epochs=2, batch_size=32, validation_data=(X_val_fold, y_val_fold), verbose=1)
            
            training_histories[cnn.__name__].append(history.history)

            loss, accuracy, precision, recall, auc = model.evaluate(X_val_fold, y_val_fold, verbose=0)

            y_pred_prob = model.predict(X_val_fold)
            y_pred = (y_pred_prob > 0.5).astype(int)

            all_true.extend(y_val_fold)
            all_pred.extend(y_pred)

            f1 = f1_score(y_val_fold, y_pred, average='weighted')

            fold_metrics.append([loss, accuracy, precision, recall, auc, f1])

            fpr, tpr, _ = roc_curve(y_val_fold, y_pred_prob) 
            roc_curves[cnn.__name__].append((fpr, tpr))

        cm = confusion_matrix(all_true, all_pred)
        confusion_matrixs[cnn.__name__].append(cm)
      
        # Guardar el promedio del modelo
        avg_loss = np.mean([m[0] for m in fold_metrics])
        avg_acc  = np.mean([m[1] for m in fold_metrics])
        avg_preci   = np.mean([m[2] for m in fold_metrics])
        avg_recall = np.mean([m[3] for m in fold_metrics])
        avg_auc  = np.mean([m[4] for m in fold_metrics])
        avg_f1   = np.mean([m[5] for m in fold_metrics])

        evaluation_results[cnn.__name__] = [avg_loss, avg_acc, avg_preci, avg_recall, avg_auc, avg_f1]

    df_results = pd.DataFrame.from_dict(evaluation_results, orient='index', columns=['loss', 'accuracy', 'precision', 'recall', 'auc', 'f1']).reset_index().rename(columns={'index': 'cnn'})

    return {'evaluation_metrics': df_results, 
            'training_curves': training_histories,
             'roc_curves': roc_curves,
             'confusion_matrices': confusion_matrixs
               }

def SpliceFinder(input_shape): #SpliceFinder - https://link.springer.com/article/10.1186/s12859-022-04971-w#Sec2 

    model = Sequential([
    
        Conv1D(filters=50, kernel_size=9, strides=1, activation='relu', padding='same', input_shape=input_shape),
        
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.3),
        
        Dense(1, activation='sigmoid')  
    ])

    return model                


def DeepSplicer(input_shape):

    model = Sequential([
    # First block of normalization
    BatchNormalization(input_shape=input_shape),
    Conv1D(filters=16, kernel_size=11, padding='same', kernel_regularizer=l1(0.0001)),
    Activation('relu'), 
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
        
    Flatten(),
    Dense(32, activation='relu', kernel_regularizer=l2(0.0001)),
    Dropout(0.5),
    Dense(16, activation='relu', kernel_regularizer=l2(0.0001)),
    Dropout(0.5) ,

    Dense(1, activation='sigmoid')
    
    ])
    

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

        Dense(1, activation='sigmoid') 
    ])


    return model 

def plot_all_results(results, model_names, save_path="results"):

# ============================================================
# Create directories
# ============================================================
    os.makedirs(f"{save_path}/curves", exist_ok=True)
    os.makedirs(f"{save_path}/roc", exist_ok=True)
    os.makedirs(f"{save_path}/confusion", exist_ok=True)
    os.makedirs(f"{save_path}/metrics", exist_ok=True)

    eval_metrics = results['evaluation_metrics']

    # ============================================================
    # 1. CURVAS DE LOSS (PROMEDIO TRAIN + PROMEDIO VAL)
    # ============================================================
    plt.figure(figsize=(9,6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    for i, model_name in enumerate(model_names):

        histories = results['training_curves'][model_name]
    
        all_train_loss = np.array([h['loss'] for h in histories])
        all_val_loss = np.array([h['val_loss'] for h in histories])
        
        # compute mean through folds
        mean_train_loss = all_train_loss.mean(axis=0)
        mean_val_loss = all_val_loss.mean(axis=0)

        epochs = range(1, len(mean_train_loss) + 1)

        plt.plot(
            epochs,
            mean_train_loss,
            label=f"{model_name} Train",
            color=colors[i],
            linestyle="-",
            linewidth=2
        )
        plt.plot(
            epochs,
            mean_val_loss,
            label=f"{model_name} Val",
            color=colors[i],
            linestyle="--",
            linewidth=2
        )

    plt.title("Average Loss Curves Across Models")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{save_path}/curves/all_models_loss.png", dpi=300)
    plt.close()

    # ============================================================
    # 2. ROC CURVES
    # ============================================================
    
    plt.figure(figsize=(7,7))

    for model_name in model_names:

        rocs = results['roc_curves'][model_name]

        mean_fpr = np.linspace(0,1,100)
        interpolated_tprs = []

        for fpr, tpr in rocs:
            interp = np.interp(mean_fpr, fpr, tpr)
            interpolated_tprs.append(interp)

        mean_tpr = np.mean(interpolated_tprs, axis=0)
        model_auc = auc(mean_fpr, mean_tpr)

        plt.plot(mean_fpr, mean_tpr, linewidth=2, label= f'{model_name} AUC = {model_auc:.3f}')

    plt.plot([0,1], [0,1], '--', color='gray') 
    plt.title("Average ROC Curves Across Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"{save_path}/roc/all_models_roc.png", dpi=300)
    plt.close()

    
    # ============================================================
    # 3. METRICS BARPLOT (COMPARING MODELS)
    # ============================================================
    metrics = ["accuracy", "precision", "recall", "auc", "f1"]
    
    values = np.array([
         [
            eval_metrics.loc[eval_metrics["cnn"] == model, metric].values[0]
            for model in model_names
        ]
        for metric in metrics
    ])

    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)
    
    plt.figure(figsize=(10,6))
    
    for i, model_name in enumerate(model_names):
        plt.bar(x + i*width, values[:,i], width=width, label=model_name)

    plt.xticks(x + width*(len(model_names)-1)/2, metrics)
    plt.ylim(0,1)
    plt.title("Average Metrics Across Models")
    plt.ylabel("Score")
    plt.legend(title='Models')
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{save_path}/metrics/all_models_metrics.png", dpi=300)
    plt.close()



    # ============================================================
    # 4. CONFUSION MATRICES (ONE PER MODEL)
    # ============================================================
    for model_name in model_names:

        conf_mats = results["confusion_matrices"][model_name]
        total_cm = np.sum(conf_mats, axis=0)

        plt.figure(figsize=(6,5))
        sns.heatmap(total_cm, annot=True, fmt="d", cmap="Blues")

        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

        plt.savefig(f"{save_path}/confusion/{model_name}_confusion.png", dpi=300)
        plt.close()

    print(f"\n✨ All plots were successfully generated in '{save_path}/' folder!\n")