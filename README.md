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

### 1. Prepare environment & install dependencies
```bash
python3 -m venv venv        # create venv
source venv/bin/activate    # activate on macOS
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py \
  --train_path data/train.csv \
  --test_path  data/train.csv \
  --text_col   text \
  --label_col  generated
```
This runs a fast randomized search on a subset, retrains on full data, and saves:
- `models/vectorizer.pkl`
- `models/model.pkl`
- `test_predictions.csv`

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```

### 4. Project structure
```
.
├── .gitignore
├── .vscode
│   └── launch.json
├── README.md
├── requirements.txt
├── train_model.py
├── app.py
├── data/
│   └── train.csv
├── test_predictions.csv
└── models
    ├── vectorizer.pkl
    └── model.pkl
```