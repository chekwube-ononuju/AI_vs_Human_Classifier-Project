import streamlit as st
import pickle
import re
import os
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

@st.cache_resource
def load_model():
    # Resolve paths relative to this script
    base_dir = Path(__file__).parent
    vect_file = base_dir / 'models' / 'vectorizer.pkl'
    clf_file  = base_dir / 'models' / 'model.pkl'
    if not vect_file.exists() or not clf_file.exists():
        st.error('Model files not found. Please train the model first.')
        st.stop()
    with open(vect_file, 'rb') as vf:
        vect = pickle.load(vf)
    with open(clf_file, 'rb') as mf:
        clf = pickle.load(mf)
    return vect, clf

vect, clf = load_model()

st.title("AI vs Human Text Classifier")

inp = st.text_area("Paste your text here:", height=200)
if st.button("Classify"):
    if not inp.strip():
        st.warning("Please enter some text.")
    else:
        def preprocess(text):
            t = text.lower()
            t = re.sub(r'<[^>]+>',' ',t)
            t = re.sub(r'[^a-z\s]',' ',t)
            toks = [w for w in t.split() if w not in ENGLISH_STOP_WORDS]
            return ' '.join(toks)
        proc = preprocess(inp)
        X = vect.transform([proc])
        # Predict class and probability
        pred = clf.predict(X)[0]
        probas = clf.predict_proba(X)[0]
        # Find index of the predicted class in classifier classes_
        class_labels = list(clf.classes_)
        pred_idx = class_labels.index(pred)
        proba = probas[pred_idx]
        # Map index 1 to AI-generated, 0 to Human-written
        label = 'AI-generated' if pred_idx == 1 else 'Human-written'
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {proba:.2%}")

        # Feature importance
        fn = vect.get_feature_names_out()
        coefs = clf.coef_[0]
        toks = proc.split()
        scores = {w: coefs[vect.vocabulary_[w]] for w in toks if w in vect.vocabulary_}
        top_pos = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        top_neg = sorted(scores.items(), key=lambda x: x[1])[:5]
        st.write("**Top AI-indicative:**")
        for w,s in top_pos: st.write(f"- {w}: {s:.3f}")
        st.write("**Top Human-indicative:**")
        for w,s in top_neg: st.write(f"- {w}: {s:.3f}")