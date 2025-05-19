import optuna
from training2 import training
from data_utils2 import get_data, get_dataloaders
import numpy as np
from sklearn.model_selection import KFold
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW


def objective(trial):

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    classImbal = trial.suggest_categorical("classImbal", [True, False])
    langImbal = trial.suggest_categorical("langImbal", [True, False])
    #num_epochs = trial.suggest_int("num_epochs", 3, 10)
    num_epochs = 1
    pretrain = "bert-base-multilingual-cased"
    eval_type = "global"

    # Get full dataset
    full_train_dataset, _ = get_data(debug = True)  







    # Convert to numpy indices for KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    indices = np.arange(len(full_train_dataset))
    
    val_metrics = []

    for train_idx, val_idx in kf.split(indices):
        print("NEW FOLD")
        train_fold = full_train_dataset.select(train_idx.tolist())
        val_fold = full_train_dataset.select(val_idx.tolist())


        # Tokenizer and label encoding
        tokenizer = BertTokenizer.from_pretrained(pretrain)
        label_encoder = LabelEncoder()
        label_encoder.fit(full_train_dataset["target"])
        num_labels = len(label_encoder.classes_)


            # Get dataloaders
        train_loader, val_loader = get_dataloaders(
        tokenizer, label_encoder, batch_size,
        train_fold, val_fold,
        class_imbal=classImbal,
        lang_imbal=langImbal
        )

        # Create model
        model = BertForSequenceClassification.from_pretrained(pretrain, num_labels=num_labels)

        #Setup optimizer
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        #Setup loss function
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')



        val_acc = training(
            model,
            train_loader,
            val_loader,
            optimizer,
            label_encoder,
            device,
            loss_fn,
            num_epochs,
            eval_type,
            freeze = False,
        )
        val_metrics.append(val_acc)

    return np.mean(val_metrics)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    print(study.best_trial)