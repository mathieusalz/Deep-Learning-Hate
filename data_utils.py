import pandas as pd
from datasets import Dataset

def get_data(debug = False):

    train_data = pd.read_csv("processed_data/train.csv")
    val_data = pd.read_csv("processed_data/val.csv")
    test_data = pd.read_csv("processed_data/test.csv")

    if debug:

        train_data = train_data.sample(frac = 0.01, random_state = 1)
        val_data = val_data.sample(frac = 0.01, random_state = 1)
        test_data = test_data.sample(frac = 0.01, random_state=1)

    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)
    test_dataset = Dataset.from_pandas(test_data)

    print(f"Size Training Set: {len(train_dataset)} \t Size Validation Set: {len(val_dataset)} \t Size Test Set: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset

def language_weights(data):

    lang_counts = pd.Series(data['language']).value_counts()
    lang_weights = 1.0 / lang_counts
    lang_weights = (lang_weights / lang_weights.sum()).to_dict()  # normalize

    print(lang_weights)

    return lang_weights
