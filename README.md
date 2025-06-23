# AI vs. Human Text Classifier

This project trains a classifier to distinguish between human-written and AI-generated text using TF-IDF + Logistic Regression, and exposes a Streamlit app for inference.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Unix/Mac
   .\\venv\\Scripts\\activate  # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Files

- `train_model.py`: loads data, preprocesses text, tunes hyperparameters, trains final model, and saves `vectorizer.pkl` and `model.pkl` into `models/`.
- `app.py`: Streamlit app that loads the pickles, accepts user input, and displays predictions and feature importances.
- `models/`: contains the trained TF-IDF vectorizer and classifier.
- `.vscode/launch.json`: config for debugging the Streamlit app in VSCode.

## Usage

### 1. Train the model
```bash
python train_model.py --train_path data/AI_vs_huam_train_dataset.xlsx --test_path data/Final_test_data.csv
```
This creates `models/vectorizer.pkl` and `models/model.pkl`, and writes `test_predictions.csv`.

### 2. Run Streamlit app
```bash
streamlit run app.py
```

## Project Structure
```
.
├── .gitignore
├── .vscode
│   └── launch.json
├── README.md
├── requirements.txt
├── train_model.py
├── app.py
└── models
    ├── vectorizer.pkl
    └── model.pkl
```