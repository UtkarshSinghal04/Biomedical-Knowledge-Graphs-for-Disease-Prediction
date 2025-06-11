import os
import numpy as np
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import json
import pprint

dataDIR = "C:/Users/Purav Jangir/Projects/KGE-DRDO/bio-kg/triples-extraction/Base-triples"
trainPATH = os.path.join(dataDIR, "train.tsv")
valPATH = os.path.join(dataDIR, "val.tsv")
testPATH = os.path.join(dataDIR, "test.tsv")

def load_triples(file_path):
    triples = []
    with open(file_path, 'r') as file:
        for line in file:
            subj, pred, obj = line.strip().split('\t')
            triples.append((subj, pred, obj))
    return triples

train_triples = load_triples(trainPATH)
val_triplesALL = load_triples(valPATH)
test_triplesALL = load_triples(testPATH)

val_triples = val_triplesALL[:500]
test_triples = test_triplesALL[:500]

train_triples += val_triplesALL[500:] + test_triplesALL[500:]

train_factory = TriplesFactory.from_labeled_triples(
    triples=np.array(train_triples, dtype=str),
    create_inverse_triples=False  
)

entity_set = set(train_factory.entity_to_id)
relation_set = set(train_factory.relation_to_id)

def filter_known(triples, known_entities, known_relations):
    return [
        (h, r, t)
        for h, r, t in triples
        if h in known_entities and t in known_entities and r in known_relations
    ]

val_triples = filter_known(val_triples, entity_set, relation_set)
test_triples = filter_known(test_triples, entity_set, relation_set)

val_factory = TriplesFactory.from_labeled_triples(
    triples=np.array(val_triples, dtype=str),
    create_inverse_triples=False
)

test_factory = TriplesFactory.from_labeled_triples(
    triples=np.array(test_triples, dtype=str),
    create_inverse_triples=False
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

print("Starting TransE training!!!")

# print(f"Train triples: {len(train_factory.triples)}")
# print(f"Val triples: {len(val_factory.triples)}")
# print(f"Test triples: {len(test_factory.triples)}")

result = pipeline(
    training=train_factory,
    validation=val_factory,
    testing=test_factory,
    model='TransE',
    model_kwargs={
        'embedding_dim': 128
    },
    training_loop='sLCWA',
    stopper='early',
    stopper_kwargs={
        'frequency': 5,
        'patience': 3,
        'relative_delta': 0.01,
    },
    random_seed=42,
    device=device,
    training_kwargs={
        'num_epochs': 20,
        'batch_size': 64,
        'checkpoint_directory': './checkpoints',
        'checkpoint_name': 'transE_checkpoint',
        'checkpoint_frequency': 5,  
        'checkpoint_on_failure': True
    }
)

outputDIR = "KGE_TransE_ebd"
os.makedirs(outputDIR, exist_ok=True)
result.save_to_directory(outputDIR)

entityTENSOR = result.model.entity_representations[0](indices=None).detach().cpu().numpy()
relationTENSOR = result.model.relation_representations[0](indices=None).detach().cpu().numpy()

np.save(os.path.join(outputDIR, 'entity_embeddings.npy'), entityTENSOR)
np.save(os.path.join(outputDIR, 'relation_embeddings.npy'), relationTENSOR)

print(f"Embeddings saved to {outputDIR}")
print(f"Entity shape: {entityTENSOR.shape}")
print(f"Relation shape: {relationTENSOR.shape}")

with open(os.path.join(outputDIR, 'entity_to_id.json'), 'w') as f:
    json.dump(train_factory.entity_to_id, f, indent=2)

with open(os.path.join(outputDIR, 'relation_to_id.json'), 'w') as f:
    json.dump(train_factory.relation_to_id, f, indent=2)
