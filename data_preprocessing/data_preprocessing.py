import os
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

# --- Neo4j Setup ---
URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "password"
driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

# --- Configuration ---
BATCH_SIZE = 2000
ROOT_DIRS = {
    "Disease-Gene": ["DiseaseID", "GeneID", "GeneSymbol"],
    "Disease-GO": ["DiseaseID", "GOID", "GOName"],
    "Disease-Chemical": ["DiseaseID", "ChemicalID", "ChemicalName"]
}

# --- Helper Functions ---
def split_symbols(s):
    if pd.isna(s): return []
    return [x.strip() for x in str(s).split("|") if x.strip()]

def chunk_df(df, batch_size):
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i+batch_size]

def get_valid_csvs(root_folder, required_columns):
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".csv"):
                full_path = os.path.join(subdir, file)
                try:
                    df = pd.read_csv(full_path, low_memory=False)
                    df.columns = df.columns.str.strip()
                    if all(col in df.columns for col in required_columns):
                        yield full_path, df
                    else:
                        print(f"[SKIP] Missing columns in {full_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to load {full_path}: {e}")

def insert_batch(tx, batch_df, data_type):
    for _, row in batch_df.iterrows():
        if data_type == "Disease-Gene":
            tx.run("""
                MERGE (d:Disease {id: $d_id}) SET d.name = $d_name
                MERGE (g:Gene {id: $g_id}) SET g.name = $g_name
                MERGE (d)-[r:RELATED_TO]->(g)
                SET r.inferenceChemicalQty = $qty
            """,
            d_id=row['DiseaseID'],
            d_name=row.get('DiseaseName'),
            g_id=row['GeneID'],
            g_name=row['GeneSymbol'],
            qty=row.get('InferenceChemicalName') if pd.notna(row.get('InferenceChemicalName')) else None)

        elif data_type == "Disease-Chemical":
            tx.run("""
                MERGE (d:Disease {id: $d_id}) SET d.name = $d_name
                MERGE (c:Chemical {id: $c_id}) SET c.name = $c_name
                MERGE (d)-[:TREATED_BY {genes: $genes}]->(c)
            """,
            d_id=row['DiseaseID'],
            d_name=row.get('DiseaseName'),
            c_id=row['ChemicalID'],
            c_name=row['ChemicalName'],
            genes=split_symbols(row.get('InferenceGeneSymbol')))

        elif data_type == "Disease-GO":
            tx.run("""
                MERGE (d:Disease {id: $d_id}) SET d.name = $d_name
                MERGE (go:GOTerm {id: $go_id})
                    SET go.name = $go_name, go.category = $category
                MERGE (d)-[:ASSOCIATED_WITH {
                    category: $category,
                    chem_qty: $chem_qty,
                    chem_symbols: $chem_syms,
                    gene_qty: $gene_qty,
                    gene_symbols: $gene_syms
                }]->(go)
            """,
            d_id=row['DiseaseID'],
            d_name=row.get('DiseaseName'),
            go_id=row['GOID'],
            go_name=row['GOName'],
            category=data_type,
            chem_qty=row.get('InferenceChemicalQty') if pd.notna(row.get('InferenceChemicalQty')) else None,
            chem_syms=split_symbols(row.get('InferenceChemicalName')),
            gene_qty=row.get('InferenceGeneQty') if pd.notna(row.get('InferenceGeneQty')) else None,
            gene_syms=split_symbols(row.get('InferenceGeneSymbols')))

# --- Main Processing ---
with driver.session() as session:
    for data_type, required_cols in ROOT_DIRS.items():
        print(f"\n[PROCESSING] {data_type} folder")

        for file_path, df in get_valid_csvs(data_type, required_cols):
            print(f"\n--> File: {file_path}")

            for i, batch_df in enumerate(chunk_df(df, BATCH_SIZE)):
                print(f"   Processing batch {i+1} ({len(batch_df)} rows)...")
                session.execute_write(insert_batch, batch_df, data_type)

print("\n[DONE] Knowledge Graph creation complete.")
driver.close()
