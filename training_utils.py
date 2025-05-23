import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from collections import defaultdict
from tqdm import tqdm

def evaluate(model, dataloader, label_encoder, device, eval_type = "per-lang", eval_metric = "accuracy"):
    model.eval()

    if eval_type == "per-lang":
        predictions, references, languages = [], [], []
        with torch.no_grad():
            for batch in dataloader:
                lang = batch.pop("language")
                batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                predictions.extend(preds)
                references.extend(labels)
                languages.extend(lang)

        metric_global = f1_score(references, predictions, average = 'macro')
        print(f"\nGlobal Validation {eval_metric} : {metric_global:.4f}")
        
        # Global metrics
        if (eval_metric == "f1"):
            metric_global = f1_score(references, predictions, average = 'macro')
        else :
            eval_metric = "accuracy"
            metric = accuracy_score(references, predictions)
        
        
        report = classification_report(references, predictions, target_names=label_encoder.classes_)

        # Per-language metrics
        lang_results = defaultdict(lambda: {"preds": [], "refs": []})
        for pred, ref, lang in zip(predictions, references, languages):
            lang_results[lang]["preds"].append(pred)
            lang_results[lang]["refs"].append(ref)

        per_language_reports = {}
        lang_nums = 0
        f1_total_lang = 0
        for lang, data in lang_results.items():
            acc_lang = accuracy_score(data["refs"], data["preds"])
            lang_nums += 1
            f1score = f1_score(data["refs"], data["preds"], average = 'macro')
            f1_total_lang += f1score
            report_lang = classification_report(
            data["refs"],
            data["preds"],  
            labels=list(range(len(label_encoder.classes_))),
            target_names=label_encoder.classes_,
            zero_division=0
            )
            per_language_reports[lang] = (acc_lang, report_lang)
            print(f"\n🔸 Language: {lang} — F1: {f1score:.4f}")

        eval_metric = 'f1-lang average'
        metric = f1_total_lang / lang_nums

        print(f"\nValidation {eval_metric} : {metric:.4f}")
        #print(report)
        '''
        print("\nPer-language reports:")
        for lang, (acc_lang, report_lang) in per_language_reports.items():
            print(f"\n🔸 Language: {lang} — Accuracy: {acc_lang:.4f}")
            print(report_lang)
        '''
    
    else:

        predictions, references = [], []
        with torch.no_grad():
            for batch in dataloader:
                lang = batch.pop("language")
                batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                predictions.extend(preds)
                references.extend(labels)
        
        if (eval_metric == "f1"):
            metric = f1_score(references, predictions, average='macro')
        else :
            eval_metric = "accuracy"
            metric = accuracy_score(references, predictions)
        report = classification_report(references, predictions, target_names=label_encoder.classes_)

        
        print(f"\nValidation {eval_metric} : {metric:.4f}")
        #print(report)
        

    return metric

def freeze_model(model):

    for param in model.bert.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True
