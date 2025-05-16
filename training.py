import argparse
import pandas as pd
import numpy as np
from datasets import Dataset

import torch
from torch.optim import AdamW

from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler   
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from training_utils import evaluate, freeze_model
from data_utils import get_data, get_language_weights, get_dataloaders, get_class_weights


def training(eval_type, 
             pretrain, 
             batch_size, 
             learning_rate, 
             num_epochs, 
             weight_decay, 
             freeze = False, 
             debug = False, 
             classImbal = True, 
             langImbal = True,
             datasets = None):
    
    # Load datasets
    if datasets is None:
        train_dataset, test_dataset = get_data(debug)
    else:
        train_dataset, test_dataset = datasets
        print(f"Size K-FOLD Training Set: {len(train_dataset)} \t Size Test Set: {len(test_dataset)}")

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer and label encoding
    tokenizer = BertTokenizer.from_pretrained(pretrain)
    label_encoder = LabelEncoder()
    label_encoder.fit(train_dataset["target"])
    num_labels = len(label_encoder.classes_)

    # Get dataloaders
    train_dataloader, test_dataloader = get_dataloaders(tokenizer, label_encoder, batch_size, train_dataset, test_dataset)

    # Create model
    model = BertForSequenceClassification.from_pretrained(pretrain, num_labels=num_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Freeze first layers for quicker training
    if freeze:
        freeze_model(model)

    # Set up Optimizer and LR scheduling
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    # Set up loss function (w/ or w/o classImbal)
    if classImbal:
        class_weights = get_class_weights(train_dataset, device)
        loss_fn = torch.nn.CrossEntropyLoss(weight = class_weights, reduction='none')
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    # Calculate language weights - if needed
    if langImbal:
        lang_weights = get_language_weights(train_dataset)

    # Main Training Loop
    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in loop:
            languages  = batch.pop("language")
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            labels = batch["labels"]
            losses = loss_fn(logits, labels)

            if langImbal:
                sample_weights = torch.tensor([lang_weights[lang] for lang in languages], dtype=torch.float).to(device)
                loss = (losses * sample_weights).mean()
            else:
                loss = losses.mean()
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loop.set_postfix(loss=loss.item())

        val_metric = evaluate(model, train_dataloader, label_encoder, device, eval_type, "f1")

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
    parser.add_argument("--classImbal", type=bool , default = False, help="Class Imbalance")
    parser.add_argument("--langImbal", type=bool , default = False, help="Language Imbalance")

    args = parser.parse_args()

    if args.debug:
        print("In DEBUG mode")

    print("\n=== Training Configuration ===")
    print(f"Evaluation type     : {args.eval_type}")
    print(f"Pretrained model    : {args.pretrain}")
    print(f"Batch size          : {args.batch_size}")
    print(f"Learning rate       : {args.learning_rate}")
    print(f"Number of epochs    : {args.num_epochs}")
    print(f"Weight decay        : {args.weight_decay}")
    print(f"Freeze BERT layers  : {args.freeze}")
    print(f"Debug mode          : {args.debug}")
    print(f"Class imbalance     : {args.classImbal}")
    print(f"Language imbalance  : {args.langImbal}")
    print("===============================\n")
    
    training(
        eval_type=args.eval_type,
        pretrain=args.pretrain,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        freeze=args.freeze,
        debug = args.debug,
        classImbal = args.classImbal,
        langImbal = args.langImbal
    )
