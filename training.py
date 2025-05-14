import argparse
import pandas as pd
import numpy as np
from datasets import Dataset

import torch
from torch.optim import AdamW

from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler   
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from training_utils import evaluate
from data_utils import get_data, language_weights, get_dataloaders, get_class_weights


def training(eval_type, pretrain, batch_size, learning_rate, num_epochs, weight_decay, freeze = False, debug = False):
    # Load datasets
    train_dataset, val_dataset, test_dataset = get_data(debug)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_weights = get_class_weights(train_dataset, device)
    lang_weights = language_weights(train_dataset)

    # Tokenizer and label encoding
    tokenizer = BertTokenizer.from_pretrained(pretrain)

    label_encoder = LabelEncoder()
    label_encoder.fit(train_dataset["target"])

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(tokenizer, label_encoder, batch_size, train_dataset, val_dataset, test_dataset)

    num_labels = len(label_encoder.classes_)
    model = BertForSequenceClassification.from_pretrained(pretrain, num_labels=num_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if freeze:
        for param in model.bert.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in loop:
            languages  = batch.pop("language")
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            labels = batch["labels"]
            #loss = loss_fn(logits, labels)
            # Get sample weights based on language
            losses = loss_fn(logits, labels)

            sample_weights = torch.tensor([lang_weights[lang] for lang in languages], dtype=torch.float).to(device)

            # Apply weights to the per-sample losses
            loss = (losses * sample_weights).mean()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loop.set_postfix(loss=loss.item())

        val_metric = evaluate(model, val_dataloader, label_encoder, device, eval_type, "f1")

    return val_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune BERT model with configurable parameters")
    parser.add_argument("--eval_type", type=str, default="per-lang", help="Evaluation type")
    parser.add_argument("--pretrain", type=str, default="bert-base-multilingual-cased", help="Pretrained model name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--freeze", type=bool , default = False, help="Freeze BERT layers")
    parser.add_argument("--debug", type=bool , default = False, help="Debug Training")

    args = parser.parse_args()

    if args.debug:
        print("In DEBUG mode")

    training(
        eval_type=args.eval_type,
        pretrain=args.pretrain,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        freeze=args.freeze,
        debug = args.debug
    )
