# 🧬 Splice Acceptor Site Prediction with Deep Learning
### A Comparative Study of CNN Architectures on Human Genomic Sequences

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow"/>
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras"/>
  <img src="https://img.shields.io/badge/Bioinformatics-Genomics-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge"/>
</p>

---

## 🎯 Why This Project Matters

Pre-mRNA splicing is one of the most fundamental and tightly regulated processes in eukaryotic gene expression. During splicing, introns are removed and exons are joined to produce a mature mRNA molecule ready for translation. The precise recognition of **splice acceptor sites** — the intronic AG dinucleotide at exon-intron boundaries — is critical for maintaining the correct reading frame and, ultimately, protein function.

Errors in splice site recognition underlie **thousands of human diseases**, including many cancers, neurological disorders, and rare genetic conditions. As genomic sequencing becomes more accessible in clinical settings, computational tools capable of accurately predicting splice sites from raw sequence data are increasingly valuable.

This project implements and compares **three convolutional neural network (CNN) architectures** inspired by published literature, applying them to a curated dataset of human acceptor splice site sequences. The goal is not only predictive performance, but also **biological interpretability** — understanding *what* each model learns about genomic sequence context.

---

## 🧪 Biological Problem

**What is an acceptor splice site?**

```
5'---EXON---[GT...intron...AG]---EXON---3'
              ↑                   ↑
           Donor site         Acceptor site
```

The acceptor splice site is defined by:
- The **AG dinucleotide** at the 3' end of the intron (positions 301–302 in this dataset)
- A **polypyrimidine tract** upstream of the AG
- Regulatory elements such as **branch point sequences** and **exonic splicing enhancers (ESEs)**

**The key challenge:** Many genomic regions contain the AG dinucleotide without being true splice sites. The model must learn the broader **sequence context** — not just the motif — to discriminate functional acceptor sites from decoys.

> ⚠️ **Experimental design note:** All sequences in this dataset (positive and negative) contain the AG motif at positions 301–302. This ensures the model cannot exploit the consensus dinucleotide alone, forcing it to learn contextual features such as polypyrimidine content, branch point signals, and exonic regulatory elements.

---

## 🏗️ Repository Structure

```
splice-site-prediction/
│
├── 📂 data/
│   ├── half_acceptor_test_positive.fasta   # True acceptor splice sites (602 nt each)
│   └── half_acceptor_test_negative.fasta   # Decoy sequences with AG but no splice site
│
├── 📂 utils/
│   ├── __init__.py
│   ├── function.py      # Core pipeline: encoding, models, training, plotting
│   └── converter.py     # DNA sequence encoding utilities (one-hot, physicochemical)
│
├── 📂 scripts/
│   └── splicesite.py    # End-to-end experiment runner
│
├── 📂 results/          # Auto-generated after running the pipeline
│   ├── curves/          # Training & validation loss curves
│   ├── roc/             # Mean ROC curves per model
│   ├── metrics/         # Bar plots of evaluation metrics
│   └── confusion/       # Aggregated confusion matrices
│
├── 📓 notebook.ipynb    # Full walkthrough with biological interpretation
├── requirements.txt
└── README.md
```

---

## 🤖 Models Implemented

All three architectures operate on one-hot encoded DNA sequences of shape `(602, 5)` and are evaluated under identical conditions.

### 1. Spliceator
**Inspired by:** [Scalzitti et al., 2021 — BMC Bioinformatics](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04471-3)

A progressive CNN with **three convolutional blocks** of increasing filter counts (16 → 32 → 64), each followed by max-pooling and dropout. This hierarchical design allows the model to detect patterns at multiple scales — from short motifs to longer regulatory elements.

| Layer | Filters | Kernel | Notes |
|-------|---------|--------|-------|
| Conv1D #1 | 16 | 7 | Short motif detection |
| Conv1D #2 | 32 | 6 | Intermediate patterns |
| Conv1D #3 | 64 | 6 | Long-range context |
| Dense | 100 | — | Classification head |

> 🧬 **Biological interpretation:** Each convolutional level corresponds to a different scale of regulatory information — nucleotide context, polypyrimidine stretches, and branch point regions, respectively.

---

### 2. SpliceFinder
**Inspired by:** [Meher et al., 2022 — BMC Bioinformatics](https://link.springer.com/article/10.1186/s12859-022-04971-w)

A lightweight, **single-layer CNN** using a kernel of size 9, followed by a fully connected classifier. Simple and fast, it targets highly local sequence motifs.

| Layer | Filters | Kernel | Notes |
|-------|---------|--------|-------|
| Conv1D | 50 | 9 | Local motif detection |
| Dense | 100 | — | Classification head |

> 🧬 **Biological interpretation:** With only one convolutional layer, this model focuses on short, fixed-width sequence patterns. This design may be sufficient when the most discriminative signal is contained within a small window, but is limited in modeling broader context.

---

### 3. DeepSplicer
**Custom architecture** designed for this project.

A CNN that prioritizes **generalization through regularization**, using batch normalization, L1/L2 weight penalties, and aggressive dropout. Fewer convolutional layers than Spliceator, but greater resistance to overfitting.

| Layer | Filters | Kernel | Notes |
|-------|---------|--------|-------|
| BatchNorm | — | — | Input normalization |
| Conv1D | 16 | 11 | Wide receptive field |
| Dense #1 | 32 | — | L2 regularized |
| Dense #2 | 16 | — | L2 regularized |

> 🧬 **Biological interpretation:** The larger kernel (11 nt) gives the model a wider view of the local sequence. The heavy regularization discourages the model from memorizing specific sequence combinations, promoting robust feature extraction that may generalize across different genomic contexts.

---

## ⚙️ Methodology

### 1. Data Encoding

Each DNA sequence is converted into a **one-hot matrix** of shape `(sequence_length, 5)`:

```
A → [1, 0, 0, 0, 0]
C → [0, 1, 0, 0, 0]
G → [0, 0, 1, 0, 0]
T → [0, 0, 0, 1, 0]
N → [0, 0, 0, 0, 0]  ← unknown nucleotide
```

This encoding preserves **positional information** while making the input architecture-agnostic. The 5th channel captures ambiguous bases without introducing arbitrary numeric values.

### 2. Dataset

| Class | Count | Description |
|-------|-------|-------------|
| Positive (1) | 3,200 | True acceptor splice sites from human genome |
| Negative (0) | 3,200 | AG-containing sequences without splice activity |
| **Total** | **6,400** | Perfectly balanced dataset |

All sequences have a fixed length of **602 nucleotides**, centered on the AG dinucleotide at positions 301–302.

### 3. Training Strategy

- **K-Fold Cross-Validation** (default: 5 folds) ensures robust and unbiased performance estimates
- **Optimizer:** Adam (adaptive learning rate)
- **Loss function:** Binary cross-entropy
- Each model is trained independently on every fold

### 4. Evaluation Metrics

| Metric | Why it matters in splice site prediction |
|--------|------------------------------------------|
| Accuracy | Overall classification performance |
| Precision | How many predicted splice sites are real |
| Recall | How many true splice sites are detected |
| AUC-ROC | Discrimination ability across all thresholds |
| F1-Score | Balance between precision and recall |

> In a clinical genomics context, **recall** is particularly critical: a missed true splice site (false negative) can mean failing to identify a pathogenic splicing variant.

---

## 📊 Generated Outputs

Running the pipeline automatically generates the following visualizations in `results/`:

| Plot | Location | Description |
|------|----------|-------------|
| Loss curves | `results/curves/` | Training vs. validation loss per epoch, averaged across folds |
| ROC curves | `results/roc/` | Mean ROC curve and AUC per model |
| Metrics bar plot | `results/metrics/` | Side-by-side comparison of all models |
| Confusion matrices | `results/confusion/` | Aggregated predictions across folds per model |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/splice-site-prediction.git
cd splice-site-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
python scripts/splicesite.py
```

### 4. Explore the notebook
Open `notebook.ipynb` in Jupyter for the full walkthrough with step-by-step biological interpretation.

---

## 📦 Requirements

```
tensorflow>=2.10
numpy
pandas
scikit-learn
matplotlib
seaborn
biopython
```

Install everything with:
```bash
pip install -r requirements.txt
```

---

## 🔬 Key Biological Takeaways

This project demonstrates several important principles at the intersection of deep learning and molecular biology:

1. **The AG consensus motif is necessary but not sufficient** for splice site function. Models that rely exclusively on this dinucleotide will perform at chance level with this dataset.

2. **Convolutional filters act as sequence motif detectors.** In the context of splicing, they can learn representations of the polypyrimidine tract, branch point consensus (YNYURAY), and exonic splicing enhancers.

3. **Architectural depth matters biologically.** Deeper networks can integrate signals across longer sequence windows, reflecting the multi-scale nature of splicing regulation.

4. **Regularization improves generalization**, particularly important when training on limited genomic datasets where overfitting to training-set-specific motifs would reduce biological validity.

---

## 🔮 Future Directions

- [ ] Extend to **donor splice sites** and compare acceptor vs. donor motif landscapes
- [ ] Train on the **full dataset** (3,200 per class) with 5-fold cross-validation
- [ ] Add **attention mechanisms** or **Grad-CAM** for interpretability (which positions matter most?)
- [ ] Benchmark against existing tools (MaxEntScan, SpliceSiteFinder)
- [ ] Apply to **clinically relevant splice variants** from ClinVar

---

## 👩‍🔬 About the Author

**Daniela Sánchez Aristizábal**  
Biologist — Computational Biology  
📍 Colombia

Transitioning into computational biology with a focus on AI-driven genomics. This project is part of a portfolio series exploring deep learning applications to fundamental molecular biology problems.

- 🎥 YouTube: *[Channel name — Bioinformatics & AI Code]*
- 💼 LinkedIn: *[your-profile]*
- 🐙 GitHub: *[your-username]*

---

## 📄 License

This project is released under the **MIT License**. See `LICENSE` for details.

---

## 📚 References

1. Scalzitti, N., et al. (2021). *Spliceator: multi-species splice site prediction using convolutional neural networks.* BMC Bioinformatics, 22(1), 561. https://doi.org/10.1186/s12859-021-04471-3

2. Meher, P.K., et al. (2022). *SpliceFinder: ab initio prediction of splice sites using convolutional neural network.* BMC Bioinformatics, 23(1). https://doi.org/10.1186/s12859-022-04971-w

3. Burset, M., Seledtsov, I. A., & Solovyev, V. V. (2000). *Analysis of canonical and non-canonical splice sites in mammalian genomes.* Nucleic Acids Research, 28(21), 4364–4375.

---

<p align="center">
  <i>Built with curiosity, biology, and too much coffee. ☕🧬</i>
</p>
