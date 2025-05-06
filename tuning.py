import optuna
from training import training

def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    num_epochs = trial.suggest_int("num_epochs", 3, 10)
    pretrain = "bert-base-multilingual-cased"
    eval_type = "global"

    val_acc = training(
        eval_type=eval_type,
        pretrain=pretrain,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        weight_decay=weight_decay
    )

    return val_acc  

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    print(study.best_trial)
