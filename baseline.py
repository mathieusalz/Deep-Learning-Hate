import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load multilingual spaCy model
nlp = spacy.load("xx_ent_wiki_sm")

# Wrapper tokenizer for TfidfVectorizer
class SpacyTokenizer:
    def __call__(self, doc):
        return [
            token.text.lower()
            for token in nlp(doc)
            if not token.is_punct and not token.is_space
        ]

def load_data():
    # Load your dataset here
    df = pd.read_csv("processed_data/train.csv")  # contains 'tweet', 'target', 'language'
    
    # Optional: restrict to known languages or balance data
    return df

def train_models(X, y):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Linear SVM": LinearSVC(),
        "Multinomial NB": MultinomialNB()
    }

    results = {}

    for name, clf in models.items():
        print(f"\n--- {name} ---")
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(tokenizer=SpacyTokenizer(), lowercase=False, min_df=5, max_df=0.5)),
            ("clf", clf)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=sorted(y.unique()))
        print(report)
        results[name] = report

    return results

# ==== MAIN EXECUTION ====

df = load_data()

# You can choose to train a multilingual model or filter by language:
# df = df[df['language'] == 'en']

X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['target'], test_size=0.2, random_state=42)

results = train_models(X_train, y_train)
