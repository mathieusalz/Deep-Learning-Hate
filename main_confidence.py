import argparse
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy import stats
from scipy.stats import sem, norm
from dataset_preprocessing import preprocess_multilingual, preprocess_english
from training2 import train_model_with_args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune BERT model with configurable parameters")
    parser.add_argument("--eval_type", type=str, default="per-lang", help="Evaluation type")
    parser.add_argument("--pretrain", type=str, default="bert-base-multilingual-cased", help="Pretrained model name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--freeze", type=bool , default = False, help="Freeze BERT layers")
    parser.add_argument("--debug", action="store_true", help="Debug Training")
    parser.add_argument("--classImbal", action="store_true", help="Class Imbalance")
    parser.add_argument("--langImbal", action = "store_true", help="Language Imbalance")

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

    all_metrics = []
    number_of_trainings = 10
    seeds = list(range(number_of_trainings))
    for seed in seeds:
        print(f"Running seed {seed}")

        set_seed(seed)
        metric = train_model_with_args(args)
        all_metrics.append(metric)

    print(f"All metrics: {all_metrics}")

    confidence = 0.95 #confidence interval

    metrics = np.array(all_metrics)
    mean_metric = np.mean(metrics) #mean
    std = np.std(metrics) #standard deviation
    n=len(metrics)
    conf_int_gaussian = norm.interval(confidence, loc=mean_metric, scale=std/np.sqrt(n))
    conf_int_student = stats.t.interval(confidence, len(metrics)-1, loc=mean_metric, scale=std/np.sqrt(n))

    print(f"Mean: {mean_metric:.4f}")
    print(f"{confidence*100:.0f}% Confidence Interval (Student's t): [{conf_int_student[0]:.4f}, {conf_int_student[1]:.4f}]")
    print(f"{confidence*100:.0f}% Confidence Interval (Gaussian):    [{conf_int_gaussian[0]:.4f}, {conf_int_gaussian[1]:.4f}]")

    # Plot
    plt.figure(figsize=(8, 5))

    plt.errorbar(x=[-0.2], y=[mean_metric],
             yerr=[[mean_metric - conf_int_student[0]], [conf_int_student[1] - mean_metric]],
             fmt='o', capsize=5, label=f"{confidence*100:.0f}% CI (Student's t)", color='blue')

    plt.errorbar(x=[0.2], y=[mean_metric],
             yerr=[[mean_metric - conf_int_gaussian[0]], [conf_int_gaussian[1] - mean_metric]],
             fmt='s', capsize=5, label=f'{confidence*100:.0f}% CI (Gaussian)', color='green')

    plt.scatter([0]*len(metrics), metrics, alpha=0.6, color='gray', label='Individual runs')
    plt.xlim(-1, 1)
    plt.title("Model Performance with 95% Confidence Interval")
    plt.ylabel("Metric")
    plt.xticks([])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("confidence_intervals.png", dpi=600)
    plt.show()