# Knowledge Graph Embedding (KGE) for Disease Prediction

This repository presents our project on **Knowledge Graph Embedding (KGE)** applied to the domain of **biomedical disease prediction**, inspired by the research paper: [A Knowledge Graph Embedding Approach for Predicting Disease-Gene Associations](https://doi.org/10.1371/journal.pone.0258626).

We developed a pipeline that constructs a biomedical knowledge graph from curated datasets, generates embeddings using **TransE**, and uses a **CNN-based model** for downstream prediction tasks.

---

## Project Workflow

### Step 1: Data Collection

The raw data required to build the knowledge graph was gathered from various biomedical sources. It includes: 

- **Entity Name**
- **Description**
- **Source of Data**
- **Total Count**

---

### Step 2: Data Preprocessing

To prepare data for graph modeling, the following preprocessing steps were applied:

- Removed empty rows and NaNs
- Removed duplicates and corrected inconsistencies
- Ensured unique identifiers like `GeneID`, `GOID`, `ChemicalID`, `DiseaseID`
- Converted cleaned data into structured triples for KG

---

### Step 3: Feature Extraction

Once the Neo4j-based Knowledge Graph was constructed:

#### 1. **Triplet Extraction**
We extracted `(head, relation, tail)` triplets and organized them into:
- Train Set
- Validation Set
- Test Set  
(*stored as `.tsv` files*)

#### 2. **Knowledge Graph Embedding with TransE**
Using [PyKEEN](https://github.com/pykeen/pykeen), the triplets were embedded into a 128-dimensional vector space. The embeddings were trained with:
- Early stopping
- Model checkpointing

Output:
- Entity & relation embeddings (NumPy)
- Entity & relation ID maps (JSON)
- Saved in `KGE_TransE_ebd` directory

---

### Step 4: CNN-Based Prediction

The TransE embeddings were fed into a **Simplified CNN Model** implemented using **PyTorch**.

#### Features:
- Custom `BiomedicalKGDataset` with negative sampling
- CNN model for head/tail prediction
- Binary cross-entropy loss + L2 regularization
- Early stopping based on **Mean Reciprocal Rank (MRR)**
- Evaluation: MRR, Hits@1, Hits@3, Hits@10

The model learns to predict missing links (e.g., gene-disease associations) using KG embedding representations.

---

## Contact

For more details, please reach out to the project contributors or raise an issue in this repository.
