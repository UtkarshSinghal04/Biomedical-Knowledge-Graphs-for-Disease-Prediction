# Biomedical Knowledge Graph for Disease Prediction

This project builds a Biomedical Knowledge Graph Embedding (KGE) model to understand complex relationships between genes, diseases, chemicals, and drugs. It aims to infer disease susceptibility—especially for rare conditions like high-altitude illnesses—by leveraging established biomedical knowledge.

---

## Goal

To predict disease comorbidities and high-altitude vulnerability by embedding biomedical entities and relations into a vector space using Knowledge Graph Embeddings and a CNN-based reranker.

---

## Motivation

Direct clinical data for rare diseases like altitude sickness is sparse. We leverage sea-level biomedical knowledge (from CTD and MalaCards) to infer risk by tracing molecular and chemical pathways through a knowledge graph.

---

## Approach

### Data Sources
- CTD: Disease-Gene and Chemical-Disease associations  
- MalaCards: Disease-Drug associations  

### Graph Construction
- Parsed and cleaned large datasets using chunked Pandas
- Created triples: `(head, relation, tail)`  
- Built the graph in Neo4j using bulk import and Cypher scripts

### Core Relations
- `Disease — associated_with → Gene`
- `Disease — annotated_by → GO_Term`
- `Disease — associated_with → Chemical`
- `Disease — treated_by → Drug`

---

## Models Used

### TransE Embedding (via PyKEEN)
- Learns vector representations of entities/relations  
- Uses margin ranking loss  
- 128D embeddings, sLCWA training with negative sampling  
- Output: `entity_embeddings.npy`, `relation_embeddings.npy`

### CNN-based Reranker (PyTorch)
- Refines TransE embeddings  
- Uses 1D convolutions over stacked embeddings  
- Contrastive training with 50 negative samples per positive  
- Evaluated with MRR and Hits@K metrics  

---

## Results

| Metric    | Value   |
|-----------|---------|
| MRR       | 0.3793  |
| Hits@3    | 0.4560  |
| Hits@10   | 0.6825  |

These scores indicate strong ranking performance and generalization capability, making the model useful for biomedical risk inference.

---
