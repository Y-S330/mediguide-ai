import os
import re
from difflib import get_close_matches
from html import escape
from typing import Dict, List, Tuple, Optional, Union

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==============================
# 1) PAGE SETUP
# ==============================
st.set_page_config(page_title="MediGuide AI", page_icon="🩺", layout="wide")

# ==============================
# 2) STYLE
# ==============================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}

[data-testid="block-container"] {
    padding: 2rem 3rem;
}

.hero {
    text-align: center;
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #38bdf8, #6366f1, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
}

.input-label {
    font-size: 1.2rem;
    font-weight: 700;
    color: #38bdf8;
    margin-bottom: 0.5rem;
}

.stMultiSelect div,
.stTextArea textarea {
    background: #020617 !important;
    border: 1.5px solid #1e293b !important;
    border-radius: 16px !important;
}

.stTextArea textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, .15) !important;
}

.stButton button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
}
.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(14, 165, 233, .35);
}

.result-card {
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 18px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    transition: 0.25s ease;
}
.result-card:hover {
    transform: translateY(-4px) scale(1.01);
    box-shadow: 0 15px 40px rgba(0, 0, 0, .45);
}

.result-card.top {
    border: 1.5px solid #38bdf8;
    background: linear-gradient(135deg, rgba(56, 189, 248, .20), rgba(99, 102, 241, .08));
    box-shadow: 0 25px 80px rgba(56, 189, 248, .25);
}

.disease-name {
    font-size: 1.7rem;
    font-weight: 900;
    margin-bottom: 0.6rem;
}

.bar-bg {
    background: #1e293b;
    height: 8px;
    border-radius: 99px;
    overflow: hidden;
    margin: 0.7rem 0;
}
.bar {
    height: 8px;
    border-radius: 99px;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
}

.warn-box {
    background: rgba(251, 191, 36, .10);
    border: 1px solid rgba(251, 191, 36, .30);
    padding: 12px;
    border-radius: 14px;
    margin-bottom: 10px;
}

.low-conf {
    background: rgba(239, 68, 68, .08);
    border: 1px solid rgba(239, 68, 68, .30);
    padding: 12px;
    border-radius: 14px;
    margin: 0.8rem 0;
}

.med-conf {
    background: rgba(251, 191, 36, .10);
    border: 1px solid rgba(251, 191, 36, .30);
    padding: 12px;
    border-radius: 14px;
    margin: 0.8rem 0;
}

.good-conf {
    background: rgba(34, 197, 94, .08);
    border: 1px solid rgba(34, 197, 94, .25);
    padding: 12px;
    border-radius: 14px;
    margin: 0.8rem 0;
}

.small-note {
    color: #94a3b8;
    font-size: 0.95rem;
}

.symptom-pill {
    display: inline-block;
    background: rgba(52, 211, 153, .12);
    border: 1px solid rgba(52, 211, 153, .30);
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 0.78rem;
    color: #6ee7b7;
    margin: 3px 4px 0 0;
}

.unknown-box {
    background: rgba(251, 191, 36, .08);
    border: 1px solid rgba(251, 191, 36, .22);
    border-radius: 12px;
    padding: 10px 12px;
    color: #fbbf24;
    margin-top: 10px;
    font-size: 0.85rem;
}

.reason-box {
    background: rgba(56, 189, 248, .08);
    border: 1px solid rgba(56, 189, 248, .20);
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 10px;
}

.typo-box {
    background: rgba(168, 85, 247, .10);
    border: 1px solid rgba(168, 85, 247, .28);
    border-radius: 12px;
    padding: 10px 12px;
    color: #d8b4fe;
    margin-top: 10px;
    font-size: 0.85rem;
}

.summary-box {
    background: rgba(34, 197, 94, .08);
    border: 1px solid rgba(34, 197, 94, .20);
    border-radius: 14px;
    padding: 12px;
    margin: 0.8rem 0;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: .80rem;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(os.path.abspath(__file__))

# ==============================
# 3) CORE HELPERS
# ==============================
def find_existing_file(candidates: List[str]) -> Optional[str]:
    for name in candidates:
        path = os.path.join(BASE, name)
        if os.path.exists(path):
            return path
    return None


def clean_text_for_match(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("_", " ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = text.replace("&", " and ")
    text = re.sub(r"\bcan['’]?t\b", "cant", text)
    text = re.sub(r"\bwon['’]?t\b", "wont", text)
    text = re.sub(r"\bdoesn['’]?t\b", "doesnt", text)
    text = re.sub(r"\bdon['’]?t\b", "dont", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_disease_key(disease_name: str) -> str:
    return (
        str(disease_name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
    )


def confidence_message(top_conf: float, second_conf: float, matched_count: int) -> Tuple[str, str]:
    gap = top_conf - second_conf

    if matched_count < 2:
        return "low", "Too few symptoms detected. Add 1–2 more relevant symptoms for a better result."
    if top_conf >= 0.50:
        return "good", "Strong confidence prediction."
    if top_conf >= 0.35 and gap >= 0.08:
        return "good", "Reasonable confidence prediction."
    if top_conf >= 0.20:
        return "medium", "Moderate confidence. Symptoms may overlap, so adding more details may improve the result."
    return "low", "Low confidence. Symptoms may be too general or overlapping."


def render_prediction_summary(top_disease: str) -> None:
    st.markdown(
        f"""
        <div class="summary-box">
            <b>Most likely condition:</b> {escape(top_disease)}<br>
            Based on the provided symptoms, this is the highest probability prediction from the model.
        </div>
        """,
        unsafe_allow_html=True
    )


def render_symptom_pills(symptoms: List[str], prefix_check: bool = False) -> str:
    pills = []
    for sym in symptoms:
        label = escape(str(sym).replace("_", " ").title())
        if prefix_check:
            label = f"✓ {label}"
        pills.append(f'<span class="symptom-pill">{label}</span>')
    return "".join(pills)


def tokenize_text(text: str) -> List[str]:
    return [tok for tok in clean_text_for_match(text).split() if tok]


def build_ngrams(tokens: List[str], min_n: int = 1, max_n: int = 4) -> List[str]:
    phrases: List[str] = []
    if not tokens:
        return phrases
    max_n = min(max_n, len(tokens))
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            phrases.append(" ".join(tokens[i:i + n]))
    return phrases


def render_pre_diagnosis_hint(symptoms: List[str]) -> None:
    if not symptoms:
        return
    if len(symptoms) == 1:
        st.info("Add at least 2–3 symptoms for a more reliable prediction.")
    elif len(symptoms) > 8:
        st.warning("Too many symptoms may reduce accuracy if they belong to different illnesses.")


# ==============================
# 4) FILE DISCOVERY
# ==============================
model_file = os.path.join(BASE, "rf_model.pkl")
label_encoder_file = os.path.join(BASE, "label_encoder.pkl")
feature_columns_file = os.path.join(BASE, "feature_columns.pkl")
display_features_file = os.path.join(BASE, "display_features.pkl")

required_files = [model_file, label_encoder_file, feature_columns_file]
missing_files = [os.path.basename(f) for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error("Missing required files:")
    for f in missing_files:
        st.write(f"- {f}")
    st.stop()

description_file = find_existing_file([
    "symptom_description.csv",
    "symptom_Description.csv",
])

precaution_file = find_existing_file([
    "disease_precaution.csv",
    "Disease precaution.csv",
])

# ==============================
# 5) LOAD MODELS
# ==============================
@st.cache_resource
def load_models():
    rf_model = joblib.load(model_file)
    label_encoder = joblib.load(label_encoder_file)

    model_feature_cols = joblib.load(feature_columns_file)
    model_feature_cols = [
        str(x).strip()
        for x in model_feature_cols
        if str(x).strip() and str(x).strip().lower() != "none"
    ]

    if os.path.exists(display_features_file):
        raw_display_feature_cols = joblib.load(display_features_file)
        raw_display_feature_cols = [
            str(x).strip()
            for x in raw_display_feature_cols
            if str(x).strip() and clean_text_for_match(x) != "none"
        ]
    else:
        raw_display_feature_cols = [x.replace("_", " ") for x in model_feature_cols]

    if len(model_feature_cols) != len(raw_display_feature_cols):
        raise ValueError("Mismatch between feature_columns.pkl and display_features.pkl lengths.")

    if not hasattr(rf_model, "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba().")

    return rf_model, label_encoder, model_feature_cols, raw_display_feature_cols

# ==============================
# 6) LOAD METADATA MAPS
# ==============================
@st.cache_data
def load_maps():
    desc_map: Dict[str, str] = {}
    prec_map: Dict[str, List[str]] = {}

    if description_file is not None:
        desc_df = pd.read_csv(description_file)
        if {"Disease", "Description"}.issubset(desc_df.columns):
            desc_df["Disease"] = desc_df["Disease"].astype(str).apply(normalize_disease_key)
            desc_df["Description"] = desc_df["Description"].astype(str).str.strip()
            desc_map = dict(zip(desc_df["Disease"], desc_df["Description"]))

    if precaution_file is not None:
        prec_df = pd.read_csv(precaution_file)
        if "Disease" in prec_df.columns:
            prec_df["Disease"] = prec_df["Disease"].astype(str).apply(normalize_disease_key)
            for _, row in prec_df.iterrows():
                disease = row["Disease"]
                precautions = row[1:].dropna().astype(str).str.strip().tolist()
                prec_map[disease] = precautions

    return desc_map, prec_map

try:
    rf, le, model_features, display_features = load_models()
    desc_map, prec_map = load_maps()
except Exception as e:
    st.error(f"Failed to load app resources: {e}")
    st.stop()

# ==============================
# 7) FEATURE MAPPING
# ==============================
display_to_model: Dict[str, str] = {}
model_to_display: Dict[str, str] = {}
cleaned_to_original_display: Dict[str, str] = {}

for disp, model in zip(display_features, model_features):
    cleaned_disp = clean_text_for_match(disp)
    display_to_model[cleaned_disp] = model
    model_to_display[model] = cleaned_disp
    cleaned_to_original_display[cleaned_disp] = disp

feature_index = {model: i for i, model in enumerate(model_features)}
canonical_display_keys = list(display_to_model.keys())

# ==============================
# 8) ALIASES / NORMALIZATION RULES
# ==============================
MANUAL_ALIASES = {
    "fever": "high fever",
    "high fever": "high fever",
    "very high fever": "high fever",
    "temperature": "high fever",
    "high temperature": "high fever",
    "raised temperature": "high fever",
    "running a temperature": "high fever",
    "feverish": "high fever",
    "hot body": "high fever",
    "body hot": "high fever",
    "burning body": "high fever",
    "mild fever": "mild fever",
    "low fever": "mild fever",
    "slight fever": "mild fever",

    "chill": "chills",
    "chills": "chills",
    "shivering": "chills",
    "shivers": "chills",
    "rigors": "chills",
    "night sweats": "sweating",
    "sweat": "sweating",
    "sweats": "sweating",
    "heavy sweating": "sweating",
    "excess sweating": "sweating",
    "sweating heavily": "sweating",

    "head ache": "headache",
    "head pain": "headache",
    "pain in head": "headache",
    "migraine": "headache",
    "severe headache": "headache",
    "bad headache": "headache",
    "pain behind eyes": "headache",
    "pain behind the eyes": "headache",
    "eye pain": "headache",
    "eye pressure": "headache",

    "blurred vision": "blurred and distorted vision",
    "blurry vision": "blurred and distorted vision",
    "vision problems": "blurred and distorted vision",
    "vision issue": "blurred and distorted vision",
    "vision issues": "blurred and distorted vision",
    "distorted vision": "blurred and distorted vision",
    "light sensitivity": "visual disturbances",
    "sensitivity to light": "visual disturbances",
    "sensitive to light": "visual disturbances",
    "seeing flashing lights": "visual disturbances",
    "seeing spots": "visual disturbances",

    "tired": "fatigue",
    "tiredness": "fatigue",
    "weakness": "fatigue",
    "feeling weak": "fatigue",
    "very tired": "fatigue",
    "extreme tiredness": "fatigue",
    "exhausted": "fatigue",
    "low energy": "fatigue",
    "body weakness": "fatigue",
    "general weakness": "fatigue",
    "feeling tired": "fatigue",
    "lack of energy": "fatigue",

    "shortness of breath": "breathlessness",
    "short breath": "breathlessness",
    "difficulty breathing": "breathlessness",
    "trouble breathing": "breathlessness",
    "cant breathe": "breathlessness",
    "can t breathe": "breathlessness",
    "breathing problem": "breathlessness",
    "hard to breathe": "breathlessness",
    "breathless": "breathlessness",

    "diarrhea": "diarrhoea",
    "diarrhoea": "diarrhoea",
    "loose motion": "diarrhoea",
    "runny stool": "diarrhoea",
    "loose stools": "diarrhoea",
    "watery stool": "diarrhoea",
    "watery stools": "diarrhoea",

    "vomit": "vomiting",
    "vomiting": "vomiting",
    "throwing up": "vomiting",
    "threw up": "vomiting",
    "feel like vomiting": "nausea",
    "feeling like vomiting": "nausea",
    "want to vomit": "nausea",
    "queasy": "nausea",
    "queasiness": "nausea",
    "sick to my stomach": "nausea",

    "stomach ache": "abdominal pain",
    "stomach cramps": "abdominal pain",
    "belly pain": "abdominal pain",
    "belly ache": "abdominal pain",
    "tummy pain": "abdominal pain",
    "tummy ache": "abdominal pain",
    "abdomen pain": "abdominal pain",
    "pain in abdomen": "abdominal pain",
    "pain in stomach": "stomach pain",
    "stomach pain": "stomach pain",

    "loss of appetite": "loss of appetite",
    "no appetite": "loss of appetite",
    "not eating": "loss of appetite",
    "reduced appetite": "loss of appetite",
    "poor appetite": "loss of appetite",

    "body pain": "muscle pain",
    "body ache": "muscle pain",
    "muscle ache": "muscle pain",
    "whole body pain": "muscle pain",
    "body soreness": "muscle pain",
    "muscle soreness": "muscle pain",

    "joint ache": "joint pain",
    "bone pain": "joint pain",
    "pain in joints": "joint pain",
    "aching joints": "joint pain",
    "severe joint pain": "joint pain",

    "chest ache": "chest pain",
    "pressure in chest": "chest pain",
    "tight chest": "chest pain",
    "chest discomfort": "chest pain",

    "back ache": "back pain",
    "lower back ache": "back pain",
    "low back pain": "back pain",

    "skin rash": "rash",
    "red spots": "rash",
    "red patch": "rash",
    "red patches": "rash",
    "itchy skin": "itching",
    "skin itching": "itching",
    "itching skin": "itching",

    "fast heartbeat": "fast heart rate",
    "heart racing": "fast heart rate",
    "rapid heartbeat": "fast heart rate",
    "heart beating fast": "fast heart rate",
    "palpitations": "fast heart rate",

    "dizzy": "dizziness",
    "feeling dizzy": "dizziness",
    "lightheaded": "dizziness",
    "light headed": "dizziness",
    "faint feeling": "dizziness",
    "feel faint": "dizziness",

    "frequent urination": "polyuria",
    "urinating frequently": "polyuria",
    "pee a lot": "polyuria",
    "peeing a lot": "polyuria",
    "excessive urination": "polyuria",
    "frequent peeing": "polyuria",

    "burning urination": "burning micturition",
    "burning while urinating": "burning micturition",
    "painful urination": "burning micturition",
    "pain when urinating": "burning micturition",

    "runny nose": "runny nose",
    "blocked nose": "congestion",
    "stuffy nose": "congestion",
    "nasal congestion": "congestion",
    "sore throat": "throat irritation",
    "throat pain": "throat irritation",
    "itchy throat": "throat irritation",
    "coughing": "cough",
    "dry cough": "cough",
    "wet cough": "cough",

    "yellow eyes": "yellowish skin",
    "yellowing eyes": "yellowish skin",
    "yellow skin": "yellowish skin",
    "skin turning yellow": "yellowish skin",

    "high fever with chills": "chills",
    "muscle and bone pain": "muscle pain",
    "pain all over body": "muscle pain",
}

alias_to_display: Dict[str, str] = {}

for disp in display_features:
    cleaned = clean_text_for_match(disp)
    alias_to_display[cleaned] = cleaned
    alias_to_display[cleaned.replace(" ", "")] = cleaned
    alias_to_display[cleaned.replace(" ", "_")] = cleaned

for alias, target in MANUAL_ALIASES.items():
    alias_clean = clean_text_for_match(alias)
    target_clean = clean_text_for_match(target)

    if target_clean in display_to_model:
        alias_to_display[alias_clean] = target_clean
        alias_to_display[alias_clean.replace(" ", "")] = target_clean
        alias_to_display[alias_clean.replace(" ", "_")] = target_clean
    elif target_clean.replace(" ", "") in display_to_model:
        alias_to_display[alias_clean] = target_clean.replace(" ", "")

sorted_aliases = sorted(alias_to_display.keys(), key=len, reverse=True)

# ==============================
# 9) TEXT MATCHING
# ==============================
@st.cache_data
def build_typo_token_pool() -> List[str]:
    token_pool = set()
    for phrase in canonical_display_keys + list(alias_to_display.keys()):
        for token in tokenize_text(phrase):
            if len(token) >= 3:
                token_pool.add(token)
    return sorted(token_pool)


TYPO_TOKEN_POOL = build_typo_token_pool()


def auto_correct_text_input(raw_text: str) -> Tuple[str, List[Tuple[str, str]]]:
    cleaned = clean_text_for_match(raw_text)
    if not cleaned:
        return "", []

    tokens = cleaned.split()
    corrected_tokens: List[str] = []
    corrections: List[Tuple[str, str]] = []

    for token in tokens:
        corrected = token
        if len(token) >= 4 and token not in TYPO_TOKEN_POOL:
            close = get_close_matches(token, TYPO_TOKEN_POOL, n=1, cutoff=0.78)
            if close:
                corrected = close[0]

        corrected_tokens.append(corrected)
        if corrected != token:
            corrections.append((token, corrected))

    corrected_text = " ".join(corrected_tokens)
    return corrected_text, corrections


def add_detected_symptom(model_symptom: str, detected_model: List[str]) -> None:
    if model_symptom and model_symptom not in detected_model:
        detected_model.append(model_symptom)


def partial_token_match_candidates(cleaned_text: str, detected_model: List[str]) -> None:
    text_tokens = set(tokenize_text(cleaned_text))
    if not text_tokens:
        return

    for display_key, model_symptom in display_to_model.items():
        if model_symptom in detected_model:
            continue

        feature_tokens = set(display_key.split())
        if not feature_tokens:
            continue

        overlap = len(feature_tokens & text_tokens)
        min_needed = 1 if len(feature_tokens) == 1 else max(2, int(np.ceil(len(feature_tokens) * 0.6)))

        if overlap >= min_needed:
            add_detected_symptom(model_symptom, detected_model)


def fuzzy_phrase_match_candidates(cleaned_text: str, detected_model: List[str]) -> None:
    tokens = tokenize_text(cleaned_text)
    if not tokens:
        return

    candidate_pool = list(alias_to_display.keys())
    ngrams = build_ngrams(tokens, 1, 4)

    for phrase in ngrams:
        close = get_close_matches(phrase, candidate_pool, n=1, cutoff=0.90)
        if not close:
            continue

        matched_alias = close[0]
        display_key = alias_to_display.get(matched_alias)
        if not display_key:
            continue

        model_symptom = display_to_model.get(display_key)
        if model_symptom:
            add_detected_symptom(model_symptom, detected_model)


def closest_suggestions_for_unknown(leftover_text: str, max_suggestions: int = 6) -> List[str]:
    tokens = tokenize_text(leftover_text)
    if not tokens:
        return []

    ngrams = build_ngrams(tokens, 1, 4)
    suggestions: List[str] = []

    for phrase in ngrams:
        close = get_close_matches(phrase, canonical_display_keys, n=3, cutoff=0.78)
        for item in close:
            if item not in suggestions:
                suggestions.append(item)
            if len(suggestions) >= max_suggestions:
                return suggestions

    return suggestions


def extract_symptoms_from_text(text: str) -> Tuple[List[str], str, List[Tuple[str, str]], str]:
    corrected_text, corrections = auto_correct_text_input(text)
    cleaned = clean_text_for_match(corrected_text)
    if not cleaned:
        return [], "", corrections, corrected_text

    detected_model: List[str] = []
    remaining = cleaned

    for alias in sorted_aliases:
        display_key = alias_to_display[alias]
        if display_key not in display_to_model:
            continue

        model_symptom = display_to_model[display_key]
        pattern = r"\b" + re.escape(alias) + r"\b"

        if re.search(pattern, remaining):
            add_detected_symptom(model_symptom, detected_model)
            remaining = re.sub(pattern, " ", remaining, count=1)

    partial_token_match_candidates(cleaned, detected_model)
    fuzzy_phrase_match_candidates(cleaned, detected_model)

    remaining = re.sub(r"\s+", " ", remaining).strip()
    return detected_model, remaining, corrections, corrected_text

# ==============================
# 10) VECTOR BUILDING
# ==============================
def build_input_vector(selected_model_symptoms: List[str]) -> np.ndarray:
    input_vector = np.zeros((1, len(model_features)), dtype=np.float32)

    for symptom in selected_model_symptoms:
        if symptom in feature_index:
            input_vector[0, feature_index[symptom]] = 1.0

    return input_vector

# ==============================
# 11) MODEL PREDICTION
# ==============================
def predict_rf_core(selected_model_symptoms_tuple: Tuple[str, ...], k: int = 5) -> Union[Dict[str, str], List[Tuple[str, float]]]:
    selected_model_symptoms = list(selected_model_symptoms_tuple)
    input_vector = build_input_vector(selected_model_symptoms)

    if input_vector.sum() == 0:
        return {"error": "No valid symptoms matched the system vocabulary."}

    try:
        probabilities = rf.predict_proba(input_vector)[0]
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

    if probabilities is None or len(probabilities) == 0:
        return {"error": "Prediction failed. Empty probability output."}

    k = min(k, len(probabilities))
    top_indices = np.argsort(probabilities)[::-1][:k]

    results: List[Tuple[str, float]] = []
    for idx in top_indices:
        disease_name = le.inverse_transform([idx])[0]
        confidence = float(probabilities[idx])
        results.append((disease_name, confidence))

    if not results:
        return {"error": "Prediction failed. Please try different symptoms."}

    return results

# ==============================
# 12) DECISION LAYER
# ==============================
def evaluate_prediction(results: List[Tuple[str, float]], matched_count: int):
    if not results:
        return {"error": "Prediction failed."}

    top_conf = results[0][1]
    second_conf = results[1][1] if len(results) > 1 else 0.0
    margin = top_conf - second_conf

    flags = {
        "too_few_symptoms": matched_count < 2,
        "low_confidence": top_conf < 0.30,
        "uncertain": margin < 0.08
    }

    return {
        "results": results,
        "confidence": top_conf,
        "margin": margin,
        "flags": flags,
        "reliable": not any(flags.values())
    }

# ==============================
# 13) TOP-3 REASONING + SUGGESTIONS UI HELPERS
# ==============================
def get_original_display_from_clean(clean_name: str) -> Optional[str]:
    return cleaned_to_original_display.get(clean_name)


def convert_display_selection_to_model(selected_display_values: List[str]) -> List[str]:
    selected_model: List[str] = []
    for s in selected_display_values:
        display_key = clean_text_for_match(s)
        if display_key in display_to_model:
            selected_model.append(display_to_model[display_key])
    return list(dict.fromkeys(selected_model))


def merge_symptom_sources(selected_display_values: List[str], detected_model_values: List[str]) -> List[str]:
    selected_model = convert_display_selection_to_model(selected_display_values)
    combined = list(dict.fromkeys(selected_model + detected_model_values))
    return combined


def queue_suggestion_addition(suggestion_clean: str) -> None:
    display_value = get_original_display_from_clean(suggestion_clean)
    if not display_value:
        return

    pending = st.session_state.get("pending_selected_display_additions", [])

    if display_value not in pending:
        pending.append(display_value)

    st.session_state["pending_selected_display_additions"] = pending
    st.session_state["last_added_symptom"] = display_value
    st.session_state["show_added_message"] = True


def apply_pending_selected_display_additions() -> None:
    pending = st.session_state.get("pending_selected_display_additions", [])

    if not pending:
        return

    current = st.session_state.get("selected_display", [])
    updated = list(current)

    for item in pending:
        if item not in updated:
            updated.append(item)

    st.session_state["selected_display"] = updated
    st.session_state["pending_selected_display_additions"] = []


def render_clickable_suggestions(suggestions: List[str], button_prefix: str = "suggest_btn") -> None:
    if not suggestions:
        return

    st.markdown(
        """
        <div class="input-label" style="margin-top:1rem;">Suggested Symptoms</div>
        <div class="small-note">Clicking a suggestion will add it to the selected symptoms list above.</div>
        """,
        unsafe_allow_html=True
    )

    cols = st.columns(min(3, max(1, len(suggestions))))

    for i, suggestion_clean in enumerate(suggestions):
        pretty = suggestion_clean.replace("_", " ").title()

        with cols[i % len(cols)]:
            if st.button(f"+ {pretty}", key=f"{button_prefix}_{suggestion_clean}_{i}"):
                queue_suggestion_addition(suggestion_clean)
                st.rerun()


def build_top3_reasoning(results: List[Tuple[str, float]], combined_symptoms: List[str]) -> List[Dict[str, str]]:
    reasoning_cards: List[Dict[str, str]] = []

    if not results:
        return reasoning_cards

    symptoms_text = ", ".join(
        s.replace("_", " ").title() for s in combined_symptoms[:6]
    ) if combined_symptoms else "the recognized symptoms"

    top_conf = results[0][1]

    for i, (disease, conf) in enumerate(results[:3]):
        rank = i + 1
        gap_from_top = top_conf - conf

        if rank == 1:
            reason = (
                f"This is the top candidate because it received the highest model probability "
                f"from the recognized symptom set: {symptoms_text}."
            )
        elif gap_from_top < 0.05:
            reason = (
                f"This remains a close alternative because its score is near the top prediction, "
                f"which suggests overlapping symptom patterns in the current input."
            )
        else:
            reason = (
                f"This is still plausible, but it scored clearly below the top candidate. "
                f"That usually means only part of the recognized symptom set matches this disease pattern."
            )

        reasoning_cards.append({
            "rank": f"#{rank}",
            "disease": disease,
            "confidence": f"{conf * 100:.1f}%",
            "reason": reason
        })

    return reasoning_cards

# ==============================
# 14) SESSION STATE
# ==============================
if "selected_display" not in st.session_state:
    st.session_state["selected_display"] = []
if "pending_selected_display_additions" not in st.session_state:
    st.session_state["pending_selected_display_additions"] = []
if "free_text" not in st.session_state:
    st.session_state["free_text"] = ""
if "results" not in st.session_state:
    st.session_state["results"] = None
if "used_symptoms" not in st.session_state:
    st.session_state["used_symptoms"] = []
if "decision_margin" not in st.session_state:
    st.session_state["decision_margin"] = None
if "decision_flags" not in st.session_state:
    st.session_state["decision_flags"] = {}
if "leftover_text" not in st.session_state:
    st.session_state["leftover_text"] = ""
if "close_suggestions" not in st.session_state:
    st.session_state["close_suggestions"] = []
if "typo_corrections" not in st.session_state:
    st.session_state["typo_corrections"] = []
if "corrected_text" not in st.session_state:
    st.session_state["corrected_text"] = ""
if "top3_reasoning" not in st.session_state:
    st.session_state["top3_reasoning"] = []
if "preview_combined_symptoms" not in st.session_state:
    st.session_state["preview_combined_symptoms"] = []
if "last_added_symptom" not in st.session_state:
    st.session_state["last_added_symptom"] = None
if "show_added_message" not in st.session_state:
    st.session_state["show_added_message"] = False

# ==============================
# 15) UI HEADER
# ==============================
st.markdown("""
<div class="hero">
    <h1>🩺 MediGuide AI</h1>
    <p>AI-powered disease prediction using structured symptoms</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# 16) MAIN UI
# ==============================
apply_pending_selected_display_additions()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="input-label">Select or Describe Your Symptoms</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-box">
        ⚠️ You can use the dropdown, type symptoms naturally, or use both.<br>
        Try to enter symptoms from the same illness for better results.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get("show_added_message") and st.session_state.get("last_added_symptom"):
        added_label = st.session_state["last_added_symptom"].replace("_", " ").title()
        st.success(f"{added_label} added to selected symptoms")
        st.session_state["show_added_message"] = False

    selected_display = st.multiselect(
        "Symptoms",
        display_features,
        placeholder="Choose symptoms from the list...",
        help="Start typing to search and select symptoms from the list",
        max_selections=10,
        label_visibility="collapsed",
        key="selected_display"
    )

    free_text = st.text_area(
        "Or type symptoms naturally",
        placeholder="e.g. high fever, headache, blurred vision, frequent urination",
        height=110,
        key="free_text"
    )

    detected_model, leftover_text, typo_corrections, corrected_text = extract_symptoms_from_text(free_text)
    close_suggestions = closest_suggestions_for_unknown(leftover_text) if leftover_text else []

    selected_model_preview = convert_display_selection_to_model(selected_display)
    combined_preview_symptoms = merge_symptom_sources(selected_display, detected_model)
    st.session_state["preview_combined_symptoms"] = combined_preview_symptoms

    if free_text.strip() and typo_corrections:
        correction_text = ", ".join(
            f"{escape(old)} → {escape(new)}" for old, new in typo_corrections[:8]
        )
        st.markdown(
            f'<div class="typo-box">Auto-corrected: <b>{correction_text}</b></div>',
            unsafe_allow_html=True
        )

    if combined_preview_symptoms:
        st.markdown(
            f'''
            <div style="margin-top:.65rem">
                <div class="small-note">Recognized symptoms that will be used for diagnosis</div>
                {render_symptom_pills(combined_preview_symptoms, prefix_check=True)}
            </div>
            ''',
            unsafe_allow_html=True
        )

        source_notes = []
        if selected_model_preview:
            source_notes.append(f"{len(selected_model_preview)} from dropdown")
        if detected_model:
            source_notes.append(f"{len(detected_model)} from text")

        if source_notes:
            st.markdown(
                f'<div class="small-note" style="margin-top:.35rem">Source: {" + ".join(source_notes)}</div>',
                unsafe_allow_html=True
            )

    render_pre_diagnosis_hint(combined_preview_symptoms)

    if free_text.strip() and leftover_text:
        extra = ""
        if close_suggestions:
            pretty_suggestions = ", ".join(
                escape(s.replace("_", " ").title()) for s in close_suggestions
            )
            extra = f"<br><span style='color:#cbd5e1'>Closest matches: {pretty_suggestions}</span>"

        st.markdown(
            f'<div class="unknown-box">Some text was not recognized: <b>{escape(leftover_text)}</b>{extra}</div>',
            unsafe_allow_html=True
        )

    if close_suggestions or typo_corrections:
        combined_suggestions = list(dict.fromkeys(
            close_suggestions + [
                clean_text_for_match(new) for _, new in typo_corrections
            ]
        ))[:6]
        render_clickable_suggestions(combined_suggestions, button_prefix="left_suggest")

    b1, b2 = st.columns([3, 1])
    with b1:
        diagnose_clicked = st.button("Diagnose", use_container_width=True)

    with b2:
        clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    for key in [
        "selected_display",
        "pending_selected_display_additions",
        "free_text",
        "results",
        "used_symptoms",
        "decision_margin",
        "decision_flags",
        "leftover_text",
        "close_suggestions",
        "typo_corrections",
        "corrected_text",
        "top3_reasoning",
        "preview_combined_symptoms",
        "last_added_symptom",
        "show_added_message"
    ]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

if diagnose_clicked:
    combined_symptoms = merge_symptom_sources(selected_display, detected_model)

    st.session_state["leftover_text"] = leftover_text
    st.session_state["close_suggestions"] = close_suggestions
    st.session_state["typo_corrections"] = typo_corrections
    st.session_state["corrected_text"] = corrected_text
    st.session_state["preview_combined_symptoms"] = combined_symptoms

    if not combined_symptoms:
        st.warning("Please select symptoms or type symptoms that the system can recognize.")
        st.session_state["results"] = None
        st.session_state["used_symptoms"] = []
        st.session_state["decision_margin"] = None
        st.session_state["decision_flags"] = {}
        st.session_state["top3_reasoning"] = []
    else:
        with st.spinner("Analyzing symptoms..."):
            prediction_output = predict_rf_core(tuple(combined_symptoms), k=5)

        if isinstance(prediction_output, dict) and "error" in prediction_output:
            st.error(prediction_output["error"])
            st.session_state["results"] = None
            st.session_state["used_symptoms"] = []
            st.session_state["decision_margin"] = None
            st.session_state["decision_flags"] = {}
            st.session_state["top3_reasoning"] = []
        else:
            decision = evaluate_prediction(prediction_output, len(combined_symptoms))

            if "error" in decision:
                st.error(decision["error"])
                st.session_state["results"] = None
                st.session_state["used_symptoms"] = []
                st.session_state["decision_margin"] = None
                st.session_state["decision_flags"] = {}
                st.session_state["top3_reasoning"] = []
            else:
                st.session_state["results"] = decision.get("results")
                st.session_state["used_symptoms"] = combined_symptoms
                st.session_state["decision_margin"] = decision.get("margin")
                st.session_state["decision_flags"] = decision.get("flags", {})
                st.session_state["top3_reasoning"] = build_top3_reasoning(
                    decision.get("results") or [],
                    combined_symptoms
                )

with col1:
    if st.session_state.get("results"):
        results = st.session_state["results"]
        combined_symptoms = st.session_state["used_symptoms"]

        top_disease, top_conf = results[0]
        second_conf = results[1][1] if len(results) > 1 else 0.0
        level, msg = confidence_message(top_conf, second_conf, len(combined_symptoms))

        st.markdown(f"""
        <div class="result-card top">
            <div class="disease-name">{escape(top_disease)}</div>
            <div class="bar-bg">
                <div class="bar" style="width:{top_conf * 100:.1f}%"></div>
            </div>
            <p>{top_conf * 100:.1f}% confidence</p>
        </div>
        """, unsafe_allow_html=True)

        render_prediction_summary(top_disease)

        if level == "good":
            st.markdown(f'<div class="good-conf">{escape(msg)}</div>', unsafe_allow_html=True)
        elif level == "medium":
            st.markdown(f'<div class="med-conf">{escape(msg)}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="low-conf">{escape(msg)}</div>', unsafe_allow_html=True)

        if len(combined_symptoms) > 8:
            st.warning("Too many mixed symptoms may reduce accuracy.")

        disease_key = normalize_disease_key(top_disease)

        st.subheader("Recognized Symptoms Used")
        st.markdown(render_symptom_pills(combined_symptoms), unsafe_allow_html=True)

        st.subheader("Description")
        desc = desc_map.get(disease_key, "No description available.")
        st.markdown(f"**{escape(desc)}**")

        st.subheader("Precautions")
        precautions = prec_map.get(disease_key, [])
        if precautions:
            for precaution in precautions:
                st.success(precaution)
        else:
            st.warning("No precautions available.")

with col2:
    st.markdown('<div class="input-label">How this works</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-box">
        Enter symptoms in simple natural language.<br>
        The app tries auto typo correction, exact matching, synonym matching, token-overlap matching, and close-match recovery before prediction.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get("corrected_text") and st.session_state.get("free_text"):
        if clean_text_for_match(st.session_state["corrected_text"]) != clean_text_for_match(st.session_state["free_text"]):
            st.markdown('<div class="input-label" style="margin-top:1rem;">Corrected Input</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="typo-box"><b>{escape(st.session_state["corrected_text"])}</b></div>',
                unsafe_allow_html=True
            )

    if st.session_state.get("leftover_text"):
        st.markdown('<div class="input-label" style="margin-top:1rem;">Unmatched Text</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="unknown-box"><b>{escape(st.session_state["leftover_text"])}</b></div>',
            unsafe_allow_html=True
        )

    if st.session_state.get("close_suggestions"):
        pretty = [s.replace("_", " ").title() for s in st.session_state["close_suggestions"]]
        st.markdown('<div class="input-label" style="margin-top:1rem;">Closest Symptom Matches</div>', unsafe_allow_html=True)
        st.markdown(render_symptom_pills(pretty), unsafe_allow_html=True)

    if st.session_state.get("top3_reasoning"):
        st.markdown('<div class="input-label" style="margin-top:1rem;">Top-3 Reasoning</div>', unsafe_allow_html=True)
        for item in st.session_state["top3_reasoning"]:
            st.markdown(
                f"""
                <div class="reason-box">
                    <b>{escape(item["rank"])} — {escape(item["disease"])}</b><br>
                    <span style="color:#7dd3fc">{escape(item["confidence"])} confidence</span><br><br>
                    <span style="color:#cbd5e1">{escape(item["reason"])}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
# ==============================
# 17) FOOTER
# ==============================
st.markdown(
    '<div class="footer">Educational use only — not medical advice</div>',
    unsafe_allow_html=True
)