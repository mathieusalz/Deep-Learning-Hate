import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from datasets import Dataset, concatenate_datasets

def mlma_preprocess(path, language):
    df = pd.read_csv(path)
    df = df[df["sentiment"] != "normal"]  # Supprimer les tweets non haineux
    df = df[["tweet", "target"]]  # Garder seulement les colonnes nécessaires
    df = df.dropna()
    df['target'] = df['target'].replace({'sexual_orientation': 'gender_sexual_orientation'})
    df['target'] = df['target'].replace({'gender': 'gender_sexual_orientation'})
    df['language'] = language 
    df = df.dropna()
    return df


def hindi_preprocess(path):
    df_hindi = pd.read_csv(path, encoding='ISO-8859-1')

    # Keep only rows where 'HS' == 1
    df_hindi = df_hindi[df_hindi['HS'] == 1].copy()

    # Define mapping from binary columns to target labels
    category_mapping = {
        'HS_Race': 'origin',
        'HS_Physical': 'disability',
        'HS_Gender': 'gender_sexual_orientation',  # merged gender + sexual orientation
        'HS_Other': 'other',
        'HS_Religion': 'religion'
    }

    # Create a column for how many of those categories each tweet matches
    df_hindi['category_count'] = df_hindi[list(category_mapping.keys())].sum(axis=1)

    # Keep only rows with exactly 1 matching category
    df_hindi = df_hindi[df_hindi['category_count'] == 1].copy()

    # Assign the single matching label to a new 'target' column
    def assign_target(row):
        for col, label in category_mapping.items():
            if row[col] == 1:
                return label
        return 'other'  # fallback, shouldn't happen after filtering

    df_hindi['target'] = df_hindi.apply(assign_target, axis=1)

    # Keep only relevant columns for mBERT
    df_hindi = df_hindi[['Tweet', 'target']].rename(columns={'Tweet': 'tweet'})
    df_hindi['language'] = 'hindi'
    df_hindi.dropna()

    return df_hindi

def mhs_preprocess(path):

    # Load the dataset
    df_parquet = pd.read_parquet(path)

    # Keep only entries with hate speech score >= 2
    df_filtered_english = df_parquet[df_parquet["hatespeech"] >= 2].copy()

    # Define a function to map categories
    def map_target(row):
        categories = []
        if row.get("target_race") or row.get("target_origin"):
            categories.append("origin")
        if row.get("target_religion"):
            categories.append("religion")
        if row.get("target_gender") or row.get("target_sexuality"):
            categories.append("gender_sexual_orientation")
        if row.get("target_disability"):
            categories.append("disability")
        if len(categories) == 1:
            return categories[0]
        return None  # Either no category or multiple

    # Apply the function
    df_filtered_english["target"] = df_filtered_english.apply(map_target, axis=1)

    # Discard rows where target is None
    df_final_english = df_filtered_english[df_filtered_english["target"].notnull()][["text", "target"]]

    # Rename "text" to "tweet"
    df_final_english.rename(columns={"text": "tweet"}, inplace=True)
    df_final_english["language"] = "english"

    df_final_english.dropna()

    return df_final_english

def OSACT_preprocess(path1, path2):
    df_OSACT = pd.concat([pd.read_csv(path1), pd.read_csv(path2)])

    df_OSACT = df_OSACT[df_OSACT['HateSpeech'].isin(['HS1', 'HS2', 'HS3', 'HS4', 'HS5', 'HS6'])].copy()

    category_mapping = {
        "HS1": "origin",
        "HS2": "religion",
        "HS3": "other",       # ideology → other
        "HS4": "disability",
        "HS5": "other",       # social class → other
        "HS6": "gender_sexual_orientation"
    }
    df_OSACT["target"] = df_OSACT["HateSpeech"].map(category_mapping)

    df_OSACT["language"] = "arabic"
    df_OSACT.rename(columns={"Text": "tweet"}, inplace=True)

    df_OSACT_final = df_OSACT[["tweet", "target", "language"]].reset_index(drop=True)

    return df_OSACT_final

def preprocess():
    # Preprocess MLMA datasets
    df_en = mlma_preprocess("data/en_dataset.csv", 'english')
    df_fr = mlma_preprocess("data/fr_dataset.csv", 'french')
    df_ar = mlma_preprocess("data/ar_dataset.csv", 'arabic')
    # ---- Combine MLMA datasets
    df_mlma = pd.concat([df_en, df_fr, df_ar])

    # Preprocess Hindi dataset
    df_hindi = hindi_preprocess("data/re_dataset.csv")

    # Preprocess Measuring Hate Speech dataset
    df_mhs = mhs_preprocess("data/measuring-hate-speech.parquet")

    # Preprocess OSACT dataset
    df_OSACT = OSACT_preprocess("data/OSACT22Test.csv", "data/OSACT22Validation.csv")

    # Combine all the datasets
    df_all = pd.concat([df_mlma, df_hindi, df_mhs, df_OSACT])
    df_all = shuffle(df_all, random_state=42).reset_index(drop=True)

    # Train/Validation/Test split
    train_df, test_df = train_test_split(df_all, test_size=0.2, random_state=42, stratify=df_all["target"])

    # Save the datasets
    train_df.to_csv("processed_data/train.csv", index=False)
    test_df.to_csv("processed_data/test.csv", index=False)

if __name__ == "__main__":
    preprocess()
    print("Preprocessing complete. Datasets saved to processed_data/")