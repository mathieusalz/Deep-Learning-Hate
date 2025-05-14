import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def get_data(debug = False):

    train_data = pd.read_csv("processed_data/train.csv")
    test_data = pd.read_csv("processed_data/test.csv")

    if debug:

        train_data = train_data.sample(frac = 0.01, random_state = 1)
        test_data = test_data.sample(frac = 0.01, random_state=1)

    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)

    print(f"Size Training Set: {len(train_dataset)} \t Size Test Set: {len(test_dataset)}")

    return train_dataset, test_dataset

def get_language_weights(data):

    lang_counts = pd.Series(data['language']).value_counts()
    lang_weights = 1.0 / lang_counts
    lang_weights = (lang_weights / lang_weights.sum()).to_dict()  # normalize

    print(lang_weights)

    return lang_weights

def get_dataloaders(tokenizer, label_encoder, batch_size, train_dataset, test_dataset):

    def preprocess(example):
        encoding = tokenizer(example["tweet"], truncation=True, padding=False, max_length=128)
        encoding["label"] = label_encoder.transform([example["target"]])[0]
        encoding["language"] = example["language"]
        return encoding

    train_dataset = train_dataset.map(preprocess)
    test_dataset = test_dataset.map(preprocess)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def data_collator_with_language(batch):
        languages = [example["language"] for example in batch]
        batch_without_language = [{k: v for k, v in example.items() if k != "language"} for example in batch]
        batch_padded = data_collator(batch_without_language)
        batch_padded["language"] = languages
        return batch_padded

    columns_to_remove = ["tweet", "target"]
    if "__index_level_0__" in train_dataset.column_names:
        columns_to_remove.append("__index_level_0__")

    train_dataloader = DataLoader(
        train_dataset.remove_columns(columns_to_remove),
        batch_size=batch_size, shuffle=True, collate_fn=data_collator_with_language)

    test_dataloader = DataLoader(
        test_dataset.remove_columns(columns_to_remove),
        batch_size=batch_size, collate_fn=data_collator_with_language)
    
    return train_dataloader, test_dataloader

def get_class_weights(data, device):

    class_weights = compute_class_weight(class_weight='balanced',
                                        classes=np.unique(data['language']),
                                        y=pd.Series(data['language']))
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
