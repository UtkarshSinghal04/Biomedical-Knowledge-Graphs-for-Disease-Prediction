import os
import gzip
import gdown
import pandas as pd

def download_ctd_file():
    file_id = "17HnxRmgly3jsduVtEBieBuHaYmA-ARaY"
    output_path = "CTD_genes_diseases.tsv.gz"
    url = f"https://drive.google.com/uc?id={file_id}"
    
    gdown.download(url, output_path, quiet=False)

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"File downloaded successfully: {output_path} ({size_mb:.2f} MB)")
    else:
        raise FileNotFoundError("Download failed. Check the file ID or internet connection.")

def preview_gzip_file(file_path, num_lines=48):
    with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as f:
        for _ in range(num_lines):
            print(next(f).strip())

def extract_disease_data():
    file_path = "Original_data/CTD_chemicals_diseases.csv"
    chunk_size = 10000

    disease_list = [
        "Heart Disease", "Stroke", "Hypertension", "Diabetes Mellitus", "COPD",
        "Asthma", "ILD", "Lung Cancer", "Breast Cancer", "Cervical Cancer",
        "Oral Cancer", "Liver Cancer", "Chronic Kidney Disease", "Cirrhosis",
        "Hepatitis", "Fatty Liver Disease", "Epilepsy", "Alzheimer", "Parkinson",
        "Rheumatoid Arthritis", "Lupus", "Psoriasis", "Osteoarthritis", "Tuberculosis",
        "Pneumonia", "Obesity", "Acute Mountain Sickness", "Chronic Mountain Sickness",
        "Pulmonary Edema", "Cerebral Edema", "Systemic Hypertension", "Sleep disorders",
        "retinal hemorrhage", "Deep Vein Thrombosis", "Hypoxia"
    ]

    columns = [
        "ChemicalName", "ChemicalID", "CasRN", "DiseaseName", "DiseaseID",
        "DirectEvidence", "InferenceGeneSymbol", "InferenceScore",
        "OmimIDs", "PubMedIDs"
    ]

    disease_counts = {}

    for chunk in pd.read_csv(
        file_path,
        sep=",",
        skiprows=28,
        names=columns,
        dtype=str,
        quotechar='"',
        on_bad_lines="skip",
        chunksize=chunk_size,
        keep_default_na=False,
        na_values=[""],
    ):
        for disease in disease_list:
            filtered = chunk[
                chunk["DiseaseName"].str.contains(disease, case=False, na=False, regex=True)
            ]

            if not filtered.empty:
                if disease not in disease_counts:
                    disease_counts[disease] = filtered
                else:
                    disease_counts[disease] = pd.concat(
                        [disease_counts[disease], filtered],
                        ignore_index=True
                    )

    summary_data = []

    for disease, df in disease_counts.items():
        filename = f"{disease}_filtered.csv"
        df.to_csv(filename, index=False)
        summary_data.append([disease, len(df)])
        print(f"Extracted data saved: {filename} ({len(df)} rows)")

    count_df = pd.DataFrame(summary_data, columns=["DiseaseName", "Count"])
    count_df.to_csv("Disease_Counts.csv", index=False)
    print("Summary file saved: Disease_Counts.csv")

if __name__ == "__main__":
    download_ctd_file()
    preview_gzip_file("CTD_genes_diseases.tsv.gz", num_lines=48)
    extract_disease_data()
