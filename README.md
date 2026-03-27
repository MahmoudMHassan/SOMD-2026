# Transformer Encoders with Heuristic-Guided Contrastive Learning for Software Coreference Resolution

**Software Mention Detection (SOMD) Subtask 1 & Subtask 2 @ NSLP 2026**

---

## Introduction

This repository implements a **Cross-Document Coreference Resolution (CDCR)** system for software mentions in scientific literature. Given a collection of academic papers where software tools are referenced (e.g., "MATLAB", "SPSS", "Python"), the system determines which mentions across different documents refer to the **same software entity** and groups them into coreference clusters.

The pipeline combines **Supervised Contrastive Learning** with domain-aware heuristics to produce high-quality coreference clusters, achieving strong performance on the SOMD shared task at NSLP 2026.

### Key Components

| Component | Description |
|---|---|
| **SciBERT Encoder** | Fine-tuned [`allenai/scibert_scivocab_uncased`](https://huggingface.co/allenai/scibert_scivocab_uncased) for scientific text understanding |
| **Supervised Contrastive Loss (SupCon)** | Pulls same-cluster mentions together and pushes different-cluster mentions apart in embedding space |
| **Projection Head** | MLP (768 → 512 → 256) with BatchNorm that maps BERT representations to a compact, normalized embedding |
| **Software-Aware Heuristics** | Name canonicalization, developer conflict detection, and exact-name boosting to refine distance matrices |
| **Hierarchical Agglomerative Clustering (HAC)** | Average-linkage clustering on heuristic-adjusted cosine distances to produce final coreference clusters |

---

## System Architecture

<p align="center">
  <img src="system_architecture.png" alt="System Architecture" width="100%">
</p>

The system operates in two phases:

1. **Training Phase**: Mention texts are encoded by SciBERT and projected into a 256-dimensional embedding space. The SupCon loss trains the encoder so that mentions of the same software are pulled close together while mentions of different software are pushed apart. Early stopping with patience 4 monitors validation CoNLL F1.

2. **Inference Phase**: The trained encoder generates embeddings for unseen test mentions. A heuristic-adjusted cosine distance matrix is computed and fed into HAC to produce the final coreference clusters, saved as `clusters.json`.

---

## Project Structure

```
├── SciBERT_Coreference_subtask_1.ipynb   # Main pipeline for Subtask 1
├── SciBERT_Coreference_subtask_2.ipynb   # Main pipeline for Subtask 2
├── Experimental_runs.ipynb               # Ablations & experimental variants
├── subtask1_dataset/
│   ├── train_data.jsonl                  # Training mentions (JSONL)
│   ├── train_labels.json                 # Gold coreference clusters
│   └── test_data.jsonl                   # Test mentions (no labels)
├── subtask2_dataset/
│   ├── train_data.jsonl
│   ├── train_labels.json
│   └── test_data.jsonl
├── system_architecture.png               # Pipeline diagram
└── README.md
```

---

## Data Format

### Mentions File (`train_data.jsonl` / `test_data.jsonl`)

Each line is a JSON object representing one software mention:

```json
{
  "mention": "MATLAB",
  "mention_id": "bb3d24e7ce",
  "start": 35,
  "end": 41,
  "type": "ProgrammingEnvironment_Usage",
  "docid": "6c90086b",
  "relations": [],
  "sentence": "Instead, we applied a custom built MATLAB routine..."
}
```

| Field | Description |
|---|---|
| `mention` | The software name as it appears in the text |
| `mention_id` | Unique identifier for this mention |
| `start` / `end` | Character offsets within the sentence |
| `type` | Mention type (e.g., `ProgrammingEnvironment_Usage`, `Application_Usage`) |
| `docid` | Document identifier |
| `relations` | List of related entities (versions, developers, abbreviations) |
| `sentence` | The full sentence containing the mention |

### Labels File (`train_labels.json`)

A JSON array of arrays, where each inner array is a coreference cluster containing mention IDs:

```json
[
  ["bb3d24e7ce", "a1b2c3d4e5", "f6g7h8i9j0"],
  ["k1l2m3n4o5"],
  ["p6q7r8s9t0", "u1v2w3x4y5"]
]
```

Each inner list groups all `mention_id` values that refer to the same software entity.

---

## Requirements

Install all dependencies via pip:

```bash
pip install transformers torch scorch scipy scikit-learn tqdm pandas pytorch-metric-learning
```

- **Python** ≥ 3.10
- **PyTorch** with CUDA support (recommended for GPU acceleration)
- **[scorch](https://github.com/LoicGrobol/scorch)** for official CoNLL coreference evaluation metrics (MUC, B³, CEAFe)

---

## Running the Notebooks

### Quick Start (with provided data)

1. Open either `SciBERT_Coreference_subtask_1.ipynb` or `SciBERT_Coreference_subtask_2.ipynb` in **Google Colab** (recommended) or **Jupyter Notebook**.
2. Run all cells sequentially. The notebook will:
   - Install dependencies
   - Load training data and gold labels
   - Train the SciBERT encoder with SupCon loss (~10 epochs with early stopping)
   - Generate coreference clusters for the test set
   - Save results to `clusters.json`

### Running with Your Own Custom Data

To use your own dataset, you need to prepare your data in the format described in the [Data Format](#data-format) section above, and then modify a few variables in the notebook.

#### Step 1: Prepare Your Data Files

Create your custom files following the exact same schema:

- **`my_train_data.jsonl`** — One JSON object per line, each with fields: `mention`, `mention_id`, `start`, `end`, `type`, `docid`, `relations`, `sentence`
- **`my_train_labels.json`** — A JSON array of arrays grouping `mention_id` values into coreference clusters
- **`my_test_data.jsonl`** — Same format as the training JSONL, but for the mentions you want to cluster (no labels needed)

#### Step 2: Update the Path Variables

In the **"Execution Loop"** cell (Section 6), locate and update the following three variables at the top:

```python
# ============================================================
# SET PATHS HERE — Replace these with your own file paths
# ============================================================
TRAIN_DATA   = 'path/to/my_train_data.jsonl'    # was: 'subtask1_dataset/train_data.jsonl'
TRAIN_LABELS = 'path/to/my_train_labels.json'    # was: 'subtask1_dataset/train_labels.json'
TEST_DATA    = 'path/to/my_test_data.jsonl'      # was: 'subtask1_dataset/test_data.jsonl'
```

These are the **only** variables you must change. Everything else (model loading, training, clustering, and output) works automatically.

#### Step 3 (Optional): Tune Hyperparameters

Depending on the size and nature of your data, you may want to adjust these parameters:

| Variable | Location | Default | Description |
|---|---|---|---|
| `batch_size` | Training DataLoader | `16` | Increase for larger datasets with sufficient GPU memory |
| `max_epochs` | Training loop | `30` | Maximum training epochs before forced stop |
| `patience` | Early stopping | `4` | Epochs without improvement before stopping |
| `lr` | Optimizer | `2e-5` | Learning rate for AdamW |
| `threshold` | `cluster_with_hac()` | `0.55` | HAC distance cutoff — **lower = more clusters, higher = fewer clusters** |
| `max_length` | `collate_fn()` | `128` | Max token length for mention encoding |

#### Step 4: Run and Collect Output

After running all cells, the predicted coreference clusters will be saved to **`clusters.json`** in the working directory. The output format matches the labels format:

```json
[
  ["mention_id_1", "mention_id_2"],
  ["mention_id_3"],
  ...
]
```

### Running the Experimental Notebook

The `Experimental_runs.ipynb` notebook contains additional experimental variants including:

- **Cross-encoder** approaches
- **Hard negative mining** strategies
- **Different loss functions** and ablations

This notebook uses a `CONFIG` dictionary for centralized parameter management:

```python
CONFIG = {
    'test_data_path': 'test_data.jsonl',    # ← Replace with your test data path
    'output_path': 'clusters.json',          # ← Replace with your desired output path
}
```

To run with custom data, update `load_data()` default arguments or pass your paths directly:

```python
mentions, clusters, mention_to_cluster, cluster_to_mentions = load_data(
    data_path='path/to/my_train_data.jsonl',
    labels_path='path/to/my_train_labels.json'
)
```

---

## Evaluation

The system uses the **CoNLL coreference evaluation** metrics via the [scorch](https://github.com/LoicGrobol/scorch) library:

- **MUC** — Link-based metric
- **B³ (B-Cubed)** — Mention-based metric
- **CEAFe** — Entity-based metric
- **CoNLL F1** — Average of MUC, B³, and CEAFe (the primary metric)

If `scorch` is not installed, the notebook falls back to a built-in B³ implementation.


