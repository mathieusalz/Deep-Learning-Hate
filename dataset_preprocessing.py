import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from datasets import Dataset, concatenate_datasets


###########################
#       MLMA Dataset      #
###########################

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df = df[df["sentiment"] != "normal"]  # Supprimer les tweets non haineux
    df = df[["tweet", "target"]]  # Garder seulement les colonnes nécessaires
    df = df.dropna()
    return df

df_en = load_and_preprocess("data/en_dataset.csv")
df_fr = load_and_preprocess("data/fr_dataset.csv")
df_ar = load_and_preprocess("data/ar_dataset.csv")


#############################
#       Hindi Dataset       #
#############################

# Load the new dataset
df_hindi = pd.read_csv("re_dataset.csv", encoding='ISO-8859-1')

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


#################################
# Measuring Hate Speech Dataset #
#################################

# Load the dataset
df_parquet = pd.read_parquet("data/measuring-hate-speech.parquet")

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

###############################
#       Final Dataset        #
###############################

df_all = pd.concat([df_en, df_fr, df_ar])

df_all['target'] = df_all['target'].replace({'sexual_orientation': 'gender_sexual_orientation'})
df_all['target'] = df_all['target'].replace({'gender': 'gender_sexual_orientation'})

df_all = pd.concat([df_all, df_hindi, df_final_english])
df_all = shuffle(df_all, random_state=42).reset_index(drop=True)

train_df, temp_df = train_test_split(df_all, test_size=0.3, random_state=42, stratify=df_all["target"])
val_df, test_df = train_test_split(temp_df, test_size=2/3, random_state=42, stratify=temp_df["target"])

train_df.to_csv("processed_data/train.csv", index=False)
val_df.to_csv("processed_data/val.csv", index=False)
test_df.to_csv("processed_data/test.csv", index=False)