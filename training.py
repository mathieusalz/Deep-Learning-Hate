import pandas as pd
from collections import defaultdict
from datasets import Dataset, concatenate_datasets

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler, DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


train_df = pd.read_csv("processed_data/train.csv")
val_df = pd.read_csv("processed_data/val.csv")
test_df = pd.read_csv("processed_data/test.csv")

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

print(f"Size Training Set : f{len(train_dataset)} \t Size Validation Set : f{len(val_dataset)} \t Size Test Set : f{len(test_dataset)}")

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Encodage des labels (string -> int)
label_encoder = LabelEncoder()
label_encoder.fit(train_dataset["target"])

def preprocess(example):
    encoding = tokenizer(example["tweet"], truncation=True, padding=False, max_length=128)
    encoding["label"] = label_encoder.transform([example["target"]])[0]
    encoding["language"] = example["language"]
    return encoding

train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def data_collator_with_language(batch):
    # Separate out 'language' from each example
    languages = [example["language"] for example in batch]
    # Remove 'language' before passing to original collator
    batch_without_language = [{k: v for k, v in example.items() if k != "language"} for example in batch]
    batch_padded = data_collator(batch_without_language)
    batch_padded["language"] = languages
    return batch_padded



train_dataloader = DataLoader(train_dataset.remove_columns(["tweet", "target", "__index_level_0__"]),
                              batch_size=16, shuffle=True, collate_fn=data_collator_with_language)
val_dataloader = DataLoader(val_dataset.remove_columns(["tweet", "target", "__index_level_0__"]),
                            batch_size=16, collate_fn=data_collator_with_language)
test_dataloader = DataLoader(test_dataset.remove_columns(["tweet", "target", "__index_level_0__"]),
                             batch_size=16, collate_fn=data_collator_with_language)

num_labels = len(label_encoder.classes_)
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=num_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Hyperparamètres de base
learning_rate = 2e-5
num_epochs = 8
weight_decay = 0.01

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_dataloader) * num_epochs
)

def evaluate(model, dataloader):
    model.eval()
    predictions, references, languages = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            lang = batch.pop("language")
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            predictions.extend(preds)
            references.extend(labels)
            languages.extend(lang)

    # Global metrics
    acc = accuracy_score(references, predictions)
    report = classification_report(references, predictions, target_names=label_encoder.classes_)

    # Per-language metrics
    lang_results = defaultdict(lambda: {"preds": [], "refs": []})
    for pred, ref, lang in zip(predictions, references, languages):
        lang_results[lang]["preds"].append(pred)
        lang_results[lang]["refs"].append(ref)

    per_language_reports = {}
    for lang, data in lang_results.items():
        acc_lang = accuracy_score(data["refs"], data["preds"])
        report_lang = classification_report(
          data["refs"],
          data["preds"],
          labels=list(range(len(label_encoder.classes_))),
          target_names=label_encoder.classes_,
          zero_division=0
        )
        per_language_reports[lang] = (acc_lang, report_lang)

    return acc, report, per_language_reports

for epoch in range(num_epochs):
    model.train()
    loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in loop:
        lang = batch.pop("language")
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        loop.set_postfix(loss=loss.item())

    # Évaluation validation
    val_acc, val_report, lang_reports = evaluate(model, val_dataloader)
    print(f"\nValidation Accuracy: {val_acc:.4f}")
    print(val_report)

    print("\nPer-language reports:")
    for lang, (acc_lang, report_lang) in lang_reports.items():
        print(f"\n🔸 Language: {lang} — Accuracy: {acc_lang:.4f}")
        print(report_lang)