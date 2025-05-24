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
from data_utils2 import get_data, get_english_data, get_language_weights, get_dataloaders, get_class_weights

from datasets.utils.logging import disable_progress_bar
from datasets.utils.logging import set_verbosity_error


def training(model,
             train_loader,
             val_loader,
             optimizer,
             label_encoder,
             device,
             loss_fn,
             num_epochs,
             eval_type,
             freeze = False, 
             debug = False):
    
    disable_progress_bar()
    set_verbosity_error()

    if debug:
        print("In DEBUG mode")

    model.to(device)

    # Freeze first layers for quicker training
    if freeze:
        freeze_model(model)

    # Set up LR scheduling
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * num_epochs
    )

    # Main Training Loop
    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable = True)
        for i,batch in enumerate(loop):
            languages = batch.pop("language")
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            labels = batch["labels"]
            losses = loss_fn(logits, labels)

            loss = losses.mean()
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if i % 250 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}", flush=True)

        val_metric = evaluate(model, val_loader, label_encoder, device, eval_type, "f1")

    return val_metric

def train_model_with_args(args):
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.pretrain == "bert-base-multilingual-cased":
        train_dataset, test_dataset = get_data(args.debug, args.smallData)
        print(f"\nLoaded multilingual data.\n")
    elif args.pretrain == "bert-base-uncased" or args.pretrain == "bert-large-uncased":
        train_dataset, test_dataset = get_english_data(args.debug, args.smallData)
        print(f"\nLoaded translated data.\n")
    else:
        train_dataset, test_dataset = get_data(args.debug, args.smallData)
        print(f"\nUnrecognized model. Loaded multilingual data.\n")

    # Tokenizer and label encoding
    tokenizer = BertTokenizer.from_pretrained(args.pretrain)
    label_encoder = LabelEncoder()
    label_encoder.fit(train_dataset["target"])
    num_labels = len(label_encoder.classes_)  

    # Get dataloaders
    train_loader, test_loader = get_dataloaders(
    tokenizer, label_encoder, args.batch_size,
    train_dataset, test_dataset,
    class_imbal=args.classImbal,
    lang_imbal=args.langImbal
)

    # Create model
    model = BertForSequenceClassification.from_pretrained(args.pretrain, num_labels=num_labels)

    #Setup optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    #Setup loss function
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    metric = training(model,
             train_loader,
             test_loader,
             optimizer,
             label_encoder,
             device,
             loss_fn,
             num_epochs=args.num_epochs,
             eval_type=args.eval_type,
             freeze=args.freeze,
             debug = args.debug,
        )

    return metric

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune BERT model with configurable parameters")
    parser.add_argument("--eval_type", type=str, default="per-lang", help="Evaluation type")
    parser.add_argument("--pretrain", type=str, default="bert-base-multilingual-cased", help="Pretrained model name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-6, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--freeze", type=bool , default = False, help="Freeze BERT layers")
    parser.add_argument("--debug", action="store_true", help="Debug Training")
    parser.add_argument("--smallData", action="store_true", help="Smaller dataset")
    parser.add_argument("--classImbal", action="store_true", default = True, help="Class Imbalance")
    parser.add_argument("--langImbal", action = "store_true", default = True, help="Language Imbalance")

    args = parser.parse_args()        

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

    train_model_with_args(args)