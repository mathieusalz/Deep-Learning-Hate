import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

from training import training
from training2 import training as training2
from data_utils import get_data
from data_utils2 import get_dataloaders


def cross_val():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "batch_size": 16,
        "learning_rate": 4e-6,
        "weight_decay": 0.1,
        "num_epochs": 5,
        "pretrain": "bert-base-multilingual-cased",
        "eval_type": "per-lang",
        "freeze": False,
    }

    full_train_dataset, _ = get_data(debug=False)
    indices = np.arange(len(full_train_dataset))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    val_metrics_1 = {
        "both": [],
        "class_only": [],
        "lang_only": [],
        "none": []
    }

    val_metrics_2 = {
        "both": [],
        "class_only": [],
        "lang_only": [],
        "none": []
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices), 1):
        print(f"\n=== FOLD {fold} ===")
        train_fold = full_train_dataset.select(train_idx.tolist())
        val_fold = full_train_dataset.select(val_idx.tolist())

        # TRAINING 1 - four variations
        for key, class_imbal, lang_imbal in [
            ("both", True, True),
            ("class_only", True, False),
            ("lang_only", False, True),
            ("none", False, False),
        ]:
            print(f"\nTRAINING 1 - ClassImbal={class_imbal}, LangImbal={lang_imbal}")
            val_acc_1 = training(
                eval_type=config["eval_type"],
                pretrain=config["pretrain"],
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                freeze=config["freeze"],
                num_epochs=config["num_epochs"],
                weight_decay=config["weight_decay"],
                datasets=(train_fold, val_fold),
                classImbal=class_imbal,
                langImbal=lang_imbal,
            )
            val_metrics_1[key].append(val_acc_1)

        # Shared setup for TRAINING 2
        tokenizer = BertTokenizer.from_pretrained(config["pretrain"])
        label_encoder = LabelEncoder()
        label_encoder.fit(full_train_dataset["target"])
        num_labels = len(label_encoder.classes_)

        for key, class_imbal, lang_imbal in [
            ("both", True, True),
            ("class_only", True, False),
            ("lang_only", False, True),
            ("none", False, False),
        ]:
            print(f"\nTRAINING 2 - ClassImbal={class_imbal}, LangImbal={lang_imbal}")
            train_loader, val_loader = get_dataloaders(
                tokenizer, label_encoder, config["batch_size"],
                train_fold, val_fold,
                class_imbal=class_imbal,
                lang_imbal=lang_imbal
            )

            model = BertForSequenceClassification.from_pretrained(config["pretrain"], num_labels=num_labels)
            optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

            val_acc_2 = training2(
                model,
                train_loader,
                val_loader,
                optimizer,
                label_encoder,
                device,
                loss_fn,
                config["num_epochs"],
                config["eval_type"],
                freeze=config["freeze"]
            )

            val_metrics_2[key].append(val_acc_2)

    print("\n=== CROSS-VALIDATION RESULTS ===")
    for key in val_metrics_1:
        print(f"TRAINING 1 ({key}) AVG ACC: {np.mean(val_metrics_1[key]):.4f}")
    for key in val_metrics_2:
        print(f"TRAINING 2 ({key}) AVG ACC: {np.mean(val_metrics_2[key]):.4f}")


if __name__ == "__main__":
    cross_val()
