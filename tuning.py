import optuna
from training import training
from data_utils import get_data
import numpy as np
from sklearn.model_selection import KFold


def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
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

        val_acc = training(
            eval_type=eval_type,
            pretrain=pretrain,
            batch_size=batch_size,
            learning_rate=learning_rate,
            freeze=True,
            num_epochs=num_epochs,
            weight_decay=weight_decay,
            datasets=(train_fold, val_fold)
        )
        val_metrics.append(val_acc)

    return np.mean(val_metrics)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    print(study.best_trial)
