import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import numpy as np
from collections import defaultdict

from spacy.lang.xx import MultiLanguage
nlp = MultiLanguage()

# Wrapper tokenizer for TfidfVectorizer
class SpacyTokenizer:
    def __call__(self, doc):
        return [
            token.text.lower()
            for token in nlp(doc)
            if not token.is_punct and not token.is_space
        ]

def load_data():
    df = pd.read_csv("processed_data/train.csv")  # contains 'tweet', 'target', 'language'
    return df

def run_cross_validation(df, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Linear SVM": LinearSVC(),
        "Multinomial NB": MultinomialNB()
    }

    global_results = defaultdict(list)  # model -> list of macro f1 scores
    per_language_scores = defaultdict(lambda: defaultdict(list))  # model -> lang -> list of f1s

    X = df["tweet"].values
    y = df["target"].values
    langs = df["language"].values

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n=== Fold {fold + 1} ===")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        langs_val = langs[val_idx]

        for model_name, model in models.items():
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(tokenizer=SpacyTokenizer(), lowercase=False, min_df=5, max_df=0.5)),
                ("clf", model)
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)

            # Global macro F1
            f1_macro = f1_score(y_val, y_pred, average='macro')
            global_results[model_name].append(f1_macro)
            print(f"{model_name} Macro F1-score: {f1_macro:.4f}")

            # Per-language F1
            lang_df = pd.DataFrame({
                "y_true": y_val,
                "y_pred": y_pred,
                "language": langs_val
            })

            for lang in lang_df["language"].unique():
                sub = lang_df[lang_df["language"] == lang]
                if len(np.unique(sub["y_true"])) > 1:  # Avoid ill-defined F1
                    f1 = f1_score(sub["y_true"], sub["y_pred"], average='macro')
                    per_language_scores[model_name][lang].append(f1)

    return global_results, per_language_scores

# ==== MAIN EXECUTION ====

df = load_data()

global_results, per_language_scores = run_cross_validation(df)

# Summary
print("\n=== Global Macro F1-score Summary ===")
for model_name, scores in global_results.items():
    print(f"{model_name}: Mean={np.mean(scores):.4f}, Std={np.std(scores):.4f}")

print("\n=== Per-language F1-scores ===")
all_langs = set()
for scores in per_language_scores.values():
    all_langs.update(scores.keys())

for model_name in per_language_scores:
    print(f"\n{model_name}:")
    lang_scores = []
    for lang in sorted(all_langs):
        scores = per_language_scores[model_name].get(lang, [])
        if scores:
            mean_score = np.mean(scores)
            lang_scores.append(mean_score)
            print(f"  {lang}: Mean F1 = {mean_score:.4f}")
        else:
            print(f"  {lang}: No data")
    if lang_scores:
        print(f"  --> Non-weighted language avg F1 = {np.mean(lang_scores):.4f}")
