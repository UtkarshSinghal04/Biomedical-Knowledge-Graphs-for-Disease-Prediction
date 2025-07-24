import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from tqdm import tqdm
import random
import sys
from collections import defaultdict

class BiomedicalKGDataset(Dataset):
    def __init__(self, triple_file, entity_to_id, relation_to_id, negative_samples=50):
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id
        self.num_entities = len(entity_to_id)
        self.negative_samples = negative_samples
        
        # Load and filter triples
        self.triples = []
        with open(triple_file, 'r', encoding='utf-8') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                if h in entity_to_id and r in relation_to_id and t in entity_to_id:
                    self.triples.append((entity_to_id[h], relation_to_id[r], entity_to_id[t]))
        
        # Create filtered set for negative sampling
        self.positive_set = set(map(tuple, self.triples))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        pos_triple = torch.tensor([h, r, t], dtype=torch.long)
        
        # Generate negative samples
        neg_triples = []
        for _ in range(self.negative_samples):
            # Corrupt head
            neg_h = random.randint(0, self.num_entities-1)
            while (neg_h, r, t) in self.positive_set:
                neg_h = random.randint(0, self.num_entities-1)
            neg_triples.append([neg_h, r, t])
            
            # Corrupt tail
            neg_t = random.randint(0, self.num_entities-1)
            while (h, r, neg_t) in self.positive_set:
                neg_t = random.randint(0, self.num_entities-1)
            neg_triples.append([h, r, neg_t])
        
        neg_triples = torch.tensor(neg_triples, dtype=torch.long)
        labels = torch.cat([torch.ones(1), torch.zeros(len(neg_triples))], dim=0)
        return pos_triple, neg_triples, labels

def collate_fn(batch):
    pos_batch = torch.stack([item[0] for item in batch])
    neg_batch = torch.cat([item[1] for item in batch])
    labels = torch.cat([item[2] for item in batch])
    return torch.cat([pos_batch, neg_batch]), labels

class SimplifiedCNNModel(nn.Module):
    def __init__(self, num_entities, num_relations, embed_dim=200, dropout=0.3, l2_lambda=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.l2_lambda = l2_lambda
        
        # Embedding layers
        self.entity_embed = nn.Embedding(num_entities, embed_dim)
        self.rel_embed = nn.Embedding(num_relations, embed_dim)
        
        # Simplified CNN architecture to prevent overfitting
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # Simple fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.entity_embed.weight)
        nn.init.xavier_normal_(self.rel_embed.weight)

    def forward(self, triples):
        h = self.entity_embed(triples[:, 0])
        r = self.rel_embed(triples[:, 1])
        t = self.entity_embed(triples[:, 2])
        
        # Stack embeddings for convolution [batch, embed_dim, 3]
        stacked = torch.stack([h, r, t], dim=2)
        
        # Apply CNN
        conv_out = self.conv(stacked).squeeze(-1)
        
        # Apply FC layers
        return self.fc(conv_out).squeeze()

    def load_pretrained_embeddings(self, entity_emb, rel_emb):
        """Load pretrained embeddings if available"""
        if entity_emb is not None:
            # Ensure dimensions match
            if self.entity_embed.weight.shape == entity_emb.shape:
                self.entity_embed.weight.data.copy_(torch.from_numpy(entity_emb))
            else:
                print(f"Entity embedding shapes don't match. Model: {self.entity_embed.weight.shape}, Pretrained: {entity_emb.shape}")
        
        if rel_emb is not None:
            # Ensure dimensions match
            if self.rel_embed.weight.shape == rel_emb.shape:
                self.rel_embed.weight.data.copy_(torch.from_numpy(rel_emb))
            else:
                print(f"Relation embedding shapes don't match. Model: {self.rel_embed.weight.shape}, Pretrained: {rel_emb.shape}")
    
    def l2_regularization(self):
        return self.l2_lambda * (self.entity_embed.weight.norm(2)**2 + 
                                self.rel_embed.weight.norm(2)**2)

def compute_loss(scores, labels, model):
    bce_loss = nn.BCEWithLogitsLoss()(scores, labels)
    return bce_loss + model.l2_regularization()

def compute_metrics(model, test_triples, all_triples, entity_to_id, device, batch_size=512):
    """
    Compute ranking metrics like MRR and Hits@k with filtered evaluation.
    """
    model.eval()
    ranks = []
    
    # Create filtered dictionary
    filter_dict = defaultdict(set)
    for h, r, t in all_triples:
        filter_dict[(h, r)].add(t)  # For tail prediction
        filter_dict[(t, r)].add(h)  # For head prediction (with inverse relation)
    
    entity_ids = torch.arange(len(entity_to_id)).to(device)
    
    with torch.no_grad():
        for h, r, t in tqdm(test_triples, desc="Evaluating"):
            # Head prediction
            head_ranks = []
            
            for start in range(0, len(entity_to_id), batch_size):
                end = min(start + batch_size, len(entity_to_id))
                current_entities = entity_ids[start:end]
                
                # Create batch for head prediction
                hr_batch = torch.zeros((len(current_entities), 3), dtype=torch.long, device=device)
                hr_batch[:, 0] = current_entities
                hr_batch[:, 1] = r
                hr_batch[:, 2] = t
                
                # Get scores
                scores = model(hr_batch).cpu().numpy()
                
                # Filter out other true triples
                for j, e in enumerate(current_entities.cpu().numpy()):
                    if e != h and e in filter_dict.get((t, r), set()):
                        scores[j] = -np.inf
                
                # If true head is in this batch
                if start <= h < end:
                    h_idx = h - start
                    h_score = scores[h_idx]
                    h_rank = 1 + np.sum(scores > h_score)
                    head_ranks.append(h_rank)
            
            if head_ranks:
                ranks.append(min(head_ranks))
            
            # Tail prediction
            tail_ranks = []
            
            for start in range(0, len(entity_to_id), batch_size):
                end = min(start + batch_size, len(entity_to_id))
                current_entities = entity_ids[start:end]
                
                # Create batch for tail prediction
                tr_batch = torch.zeros((len(current_entities), 3), dtype=torch.long, device=device)
                tr_batch[:, 0] = h
                tr_batch[:, 1] = r
                tr_batch[:, 2] = current_entities
                
                # Get scores
                scores = model(tr_batch).cpu().numpy()
                
                # Filter out other true triples
                for j, e in enumerate(current_entities.cpu().numpy()):
                    if e != t and e in filter_dict.get((h, r), set()):
                        scores[j] = -np.inf
                
                # If true tail is in this batch
                if start <= t < end:
                    t_idx = t - start
                    t_score = scores[t_idx]
                    t_rank = 1 + np.sum(scores > t_score)
                    tail_ranks.append(t_rank)
            
            if tail_ranks:
                ranks.append(min(tail_ranks))
    
    # Calculate metrics
    ranks_array = np.array(ranks)
    mrr = np.mean(1.0 / ranks_array)
    
    hits = {}
    for k in [1, 3, 10]:
        hits[k] = np.mean(ranks_array <= k)
    
    return {'MRR': mrr, **{f'Hits@{k}': hits[k] for k in hits}}

def train_model(model, train_loader, val_triples, all_triples, entity_to_id, optimizer, device, epochs=100, patience=5):
    """
    Train the model with early stopping based on validation MRR.
    """
    model.to(device)
    best_mrr = 0.0
    patience_counter = 0
    validate_every = 1  # Validate every N epochs after the first few
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (triples, labels) in enumerate(pbar):
            triples = triples.to(device)
            labels = labels.to(device)
            
            scores = model(triples)
            loss = compute_loss(scores, labels, model)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f"{total_loss/(batch_idx+1):.4f}"})
        
        # Validation using MRR
        should_validate = (epoch + 1) % validate_every == 0
        if should_validate:
            print(f"\nValidating epoch {epoch+1}...")
            
            # Sample validation triples to speed up validation
            sampled_val_triples = random.sample(val_triples, min(len(val_triples), 500))
            metrics = compute_metrics(model, sampled_val_triples, all_triples, entity_to_id, device)
            
            print(f"Validation - MRR: {metrics['MRR']:.4f}, Hits@1: {metrics['Hits@1']:.4f}, "
                  f"Hits@3: {metrics['Hits@3']:.4f}, Hits@10: {metrics['Hits@10']:.4f}")
            
            val_mrr = metrics['MRR']
            
            if val_mrr > best_mrr:
                best_mrr = val_mrr
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pt')
                print(f"Saved new best model with MRR: {best_mrr:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} validations")
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    base_path = '/kaggle/input/embeddings-triplet-dataset'  # Adjust to your path
    
    with open(os.path.join(base_path, 'entity_to_id.json'), 'r') as f:
        entity_to_id = json.load(f)
    with open(os.path.join(base_path, 'relation_to_id.json'), 'r') as f:
        relation_to_id = json.load(f)
    
    # Try to load pretrained embeddings if they exist
    entity_emb = None
    rel_emb = None
    try:
        entity_emb = np.load(os.path.join(base_path, 'entity_embeddings_continued.npy'))
        rel_emb = np.load(os.path.join(base_path, 'relation_embeddings_continued.npy'))
        print("Loaded pretrained embeddings")
    except FileNotFoundError:
        print("Pretrained embeddings not found, will initialize randomly")
    
    # Load all triples for filtered evaluation
    all_triples = set()
    for split in ['train', 'val', 'test']:
        with open(os.path.join(base_path, f'{split}.tsv')) as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                if h in entity_to_id and r in relation_to_id and t in entity_to_id:
                    all_triples.add((
                        entity_to_id[h], relation_to_id[r], entity_to_id[t]
                    ))
    
    print(f"Loaded {len(all_triples)} total triples for filtering")
    
    # Create datasets
    train_set = BiomedicalKGDataset(
        os.path.join(base_path, 'train.tsv'),
        entity_to_id,
        relation_to_id,
        negative_samples=50  # Research-backed optimal value for biomedical KGs
    )
    
    # Load validation/test triples
    def load_triples(split):
        triples = []
        with open(os.path.join(base_path, f'{split}.tsv')) as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                if h in entity_to_id and r in relation_to_id and t in entity_to_id:
                    triples.append((
                        entity_to_id[h], relation_to_id[r], entity_to_id[t]
                    ))
        return triples
    
    val_triples = load_triples('val')
    test_triples = load_triples('test')
    
    print(f"Train triples: {len(train_set)}")
    print(f"Validation triples: {len(val_triples)}")
    print(f"Test triples: {len(test_triples)}")
    
    # Create data loader
    train_loader = DataLoader(
        train_set, batch_size=128, shuffle=True, 
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    # Get embedding dimension from pretrained embeddings if available
    embed_dim = 200
    if entity_emb is not None:
        embed_dim = entity_emb.shape[1]
        print(f"Setting embedding dimension to {embed_dim} from pretrained embeddings")
    
    # Initialize model with simplified architecture
    model = SimplifiedCNNModel(
        num_entities=len(entity_to_id),
        num_relations=len(relation_to_id),
        embed_dim=embed_dim,
        dropout=0.3,
        l2_lambda=1e-5
    )
    
    # Load pretrained embeddings if available
    if entity_emb is not None and rel_emb is not None:
        model.load_pretrained_embeddings(entity_emb, rel_emb)
    
    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train with MRR-based early stopping
    train_model(
        model, train_loader, val_triples, all_triples, 
        entity_to_id, optimizer, device, epochs=10, patience=5
    )
    
    # Final evaluation
    print("\n=== Final Test Evaluation ===")
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    test_metrics = compute_metrics(model, test_triples, all_triples, entity_to_id, device)
    
    print(f"MRR: {test_metrics['MRR']:.4f}")
    print(f"Hits@1: {test_metrics['Hits@1']:.4f}")
    print(f"Hits@3: {test_metrics['Hits@3']:.4f}")
    print(f"Hits@10: {test_metrics['Hits@10']:.4f}")

if __name__ == "__main__":
    main()
