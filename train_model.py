import argparse
import os
import re
import pickle
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Preprocessing
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return ' '.join(tokens)

# Main training function
def main(train_path, test_path, text_col='essay', label_col='label'):
    # Load data
    # Load train and test data (Excel or CSV)
    if train_path.lower().endswith('.xlsx'):    
        train_df = pd.read_excel(train_path)
    else:
        train_df = pd.read_csv(train_path)
    if test_path.lower().endswith(('.xlsx', '.xls')):
        test_df = pd.read_excel(test_path)
    else:
        test_df = pd.read_csv(test_path)

    # Preprocess using specified text column
    train_df['processed'] = train_df[text_col].apply(preprocess_text)
    test_df['processed'] = test_df[text_col].apply(preprocess_text)

    # Split for validation
    X = train_df['processed']
    y = train_df[label_col]
    print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples.")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")

    # Pipeline & grid
    pipeline = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    param_grid = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__min_df': [1, 5, 10],
        'vect__max_df': [0.75, 1.0],
        'clf__C': [0.01, 0.1, 1, 10]
    }
    # Randomized search on a subset for speed
    # Subsample 10k for tuning
    if len(X_train) > 10000:
        tune_idx = X_train.sample(n=10000, random_state=42).index
        X_tune, y_tune = X_train.loc[tune_idx], y_train.loc[tune_idx]
    else:
        X_tune, y_tune = X_train, y_train
    rand = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=20,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    print("Starting randomized search on subset of size", len(X_tune))
    rand.fit(X_tune, y_tune)
    best_params = rand.best_params_
    print("Best parameters:", best_params)
    # Retrain full pipeline with best parameters
    pipeline.set_params(**best_params)
    print("Retraining full pipeline on all training data...")
    pipeline.fit(X_train, y_train)

    # Save best model and vectorizer
    os.makedirs('models', exist_ok=True)
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(pipeline.named_steps['vect'], f)
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(pipeline.named_steps['clf'], f)
    print("Models saved to models/vectorizer.pkl and models/model.pkl.")

    # Evaluate on validation
    val_acc = rand.score(X_val, y_val)
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Predict test set
    preds = pipeline.predict(test_df['processed'])
    pd.DataFrame({'prediction': preds}).to_csv('test_predictions.csv', index=False)
    print("Test predictions saved to test_predictions.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', required=True)
    parser.add_argument('--test_path',  required=True)
    parser.add_argument('--text_col',    default='essay', help='name of the text column')
    parser.add_argument('--label_col',   default='label', help='name of the label column')
    args = parser.parse_args()
    main(args.train_path, args.test_path, args.text_col, args.label_col)
