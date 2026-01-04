# Deep Learning Models for Splice Site Prediction

This repository implements a comparative analysis of three deep learning architectures
for **splice site prediction** using the same dataset and evaluation pipeline.
The project focuses on evaluating model performance through multiple metrics and visual
interpretation of results.

## 🧬 Project Overview

Splice site prediction is a fundamental task in computational genomics, as accurate
identification of exon–intron boundaries is essential for understanding gene structure
and alternative splicing mechanisms.

In this project, three convolutional neural network (CNN) architectures are trained and evaluated using the same dataset and encoding strategy in order to ensure a fair comparison.

The implemented architectures are **simplified versions** of previously proposed models. 
Some layers were intentionally omitted in order to reduce computational complexity and 
facilitate code execution and reproducibility, while preserving the core architectural 
principles of each model.

The evaluated models are:

- **Spliceator** – CNN-based architecture inspired by published literature
- **SpliceFinder** – Lightweight CNN architecture
- **DeepSplicer** – Custom CNN architecture with regularization strategies

All models are evaluated using *k-fold cross-validation*, and their performance is compared 
using multiple metrics and visualizations.

## ⚙️ Methodology
**1. Data Encoding**

DNA sequences are converted into one-hot encoding representations. Positive and negative splice site sequences are merged into a single dataset.

**2. Training Strategy**

K-Fold Cross-Validation is applied to ensure robust evaluation. Each model is trained independently on each fold. Binary cross-entropy is used as the loss function.

**3. Evaluation Metrics**

For each model, the following metrics are computed and averaged across folds:

- Loss

- Accuracy

- Precision

- Recall

- AUC (ROC)

- F1-score

## 📊 Generated Results

The pipeline automatically generates and saves:

- Average training and validation loss curves

- Mean ROC curves across folds

- Bar plots comparing evaluation metrics

- Aggregated confusion matrices

All results are stored in the results/ directory.

## 🚀 How to Run

1. Clone the repository: 
`python git clone https://github.com/your-username/your-repo-name.git`

2. Install dependencies:

`pip install -r requirements.txt`


3. Run the main script:

`python scripts/run_experiment.py`

## 📌 Requirements

- Python 3.8+

- TensorFlow / Keras

- NumPy

- Pandas

- Scikit-learn

- Matplotlib

- Seaborn

- Biopython


### 👩‍🔬 Author

Daniela Sánchez Aristizábal. Biologist – Computational Biology
Interested in AI applications for genomics.

### 📄 License

This project is released under the MIT License.
