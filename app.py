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
    if top_conf >= 0.60:
        return "good", "Strong confidence prediction."
    if top_conf >= 0.40 and gap >= 0.10:
        return "good", "Reasonable confidence prediction."
    if top_conf >= 0.22:
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
canonical_display_key_set = set(canonical_display_keys)

disease_index_by_key: Dict[str, int] = {}
for idx, class_name in enumerate(le.classes_):
    disease_index_by_key[normalize_disease_key(class_name)] = idx

# ==============================
# 8) ALIASES / NORMALIZATION RULES
# ==============================
def add_alias_mapping(alias_store: Dict[str, str], alias_phrase: str, target_feature: str) -> None:
    alias_clean = clean_text_for_match(alias_phrase)
    target_clean = clean_text_for_match(target_feature)

    if not alias_clean or not target_clean:
        return
    if target_clean not in canonical_display_key_set:
        return

    alias_store[alias_clean] = target_clean
    alias_store[alias_clean.replace(" ", "")] = target_clean
    alias_store[alias_clean.replace(" ", "_")] = target_clean


ALIASES_TO_REAL_FEATURES: Dict[str, str] = {
    "headache": "headache",
    "severe headache": "headache",
    "bad headache": "headache",
    "throbbing headache": "headache",
    "one sided headache": "headache",
    "migraine": "headache",
    "head hurts": "headache",
    "my head hurts": "headache",
    "head hurts a lot": "headache",
    "pain in my head": "headache",
    "head is hurting": "headache",

    "light sensitivity": "visual disturbances",
    "sensitivity to light": "visual disturbances",
    "sensitive to light": "visual disturbances",
    "photophobia": "visual disturbances",
    "vision changes": "visual disturbances",
    "light hurts my eyes": "visual disturbances",
    "light hurts eyes": "visual disturbances",
    "bright light hurts my eyes": "visual disturbances",
    "blurred vision": "blurred and distorted vision",
    "blurry vision": "blurred and distorted vision",
    "double vision": "blurred and distorted vision",

    "shortness of breath": "breathlessness",
    "difficulty breathing": "breathlessness",
    "trouble breathing": "breathlessness",
    "hard to breathe": "breathlessness",
    "breathless": "breathlessness",
    "wheezing": "breathlessness",

    "chest discomfort": "chest pain",
    "chest pressure": "chest pain",
    "pressure in chest": "chest pain",
    "pressure in my chest": "chest pain",
    "tight chest": "chest pain",
    "chest tightness": "chest pain",

    "continuous sneezing": "continuous sneezing",
    "sneezing": "continuous sneezing",
    "runny nose": "continuous sneezing",
    "stuffy nose": "continuous sneezing",
    "blocked nose": "continuous sneezing",
    "nasal congestion": "continuous sneezing",

    "watering eyes": "watering from eyes",
    "watery eyes": "watering from eyes",
    "teary eyes": "watering from eyes",
    "eye watering": "watering from eyes",
    "itchy watery eyes": "watering from eyes",
    "eye irritation": "watering from eyes",

    "diarrhea": "diarrhoea",
    "diarrhoea": "diarrhoea",
    "loose motion": "diarrhoea",
    "loose motions": "diarrhoea",
    "loose stool": "diarrhoea",
    "loose stools": "diarrhoea",
    "lose stool": "diarrhoea",
    "lose stools": "diarrhoea",
    "watery stool": "diarrhoea",
    "watery stools": "diarrhoea",

    "vomiting": "vomiting",
    "throwing up": "vomiting",
    "threw up": "vomiting",

    "nausea": "nausea",
    "feeling sick": "nausea",
    "queasy": "nausea",
    "feel nauseous": "nausea",
    "feel like vomiting": "nausea",
    "want to vomit": "nausea",

    "dehydrated": "dehydration",
    "dehydration": "dehydration",

    "abdominal pain": "indigestion",
    "stomach pain": "indigestion",
    "stomach ache": "indigestion",
    "stomach hurts": "indigestion",
    "belly pain": "indigestion",
    "belly ache": "indigestion",
    "tummy pain": "indigestion",
    "tummy ache": "indigestion",
    "abdomen pain": "indigestion",
    "pain in stomach": "indigestion",
    "pain in abdomen": "indigestion",
    "lower abdominal pain": "indigestion",
    "cramps": "indigestion",
    "stomach cramps": "indigestion",

    "frequent urination": "continuous feel of urine",
    "urinating frequently": "continuous feel of urine",
    "urinating often": "continuous feel of urine",
    "peeing often": "continuous feel of urine",
    "pee a lot": "continuous feel of urine",
    "frequent pee": "continuous feel of urine",
    "peeing a lot": "continuous feel of urine",
    "passing urine often": "continuous feel of urine",
    "urge to pee": "continuous feel of urine",
    "need to pee often": "continuous feel of urine",

    "burning urination": "burning micturition",
    "burning while urinating": "burning micturition",
    "painful urination": "burning micturition",
    "pain when urinating": "burning micturition",
    "urine burns": "burning micturition",
    "burning urine": "burning micturition",
    "urine burning": "burning micturition",
    "burning pee": "burning micturition",
    "pee burns": "burning micturition",
    "pain while peeing": "burning micturition",

    "bladder pain": "bladder discomfort",
    "bladder discomfort": "bladder discomfort",
    "pelvic pressure": "bladder discomfort",
    "pressure in bladder": "bladder discomfort",

    "smelly urine": "foul smell of urine",
    "foul smelling urine": "foul smell of urine",
    "bad smelling urine": "foul smell of urine",

    "pimples": "pus filled pimples",
    "pimple": "pus filled pimples",
    "acne": "pus filled pimples",
    "breakouts": "pus filled pimples",
    "skin pimples": "pus filled pimples",
    "pus filled pimples": "pus filled pimples",

    "black heads": "blackheads",
    "blackheads": "blackheads",
    "oily skin": "blackheads",

    "skin rash": "skin rash",
    "rash": "skin rash",
    "red rash": "skin rash",
    "itchy rash": "skin rash",
    "skin redness": "skin rash",
    "redness": "skin rash",

    "itching": "itching",
    "itchy skin": "itching",

    "patches": "dischromic patches",
    "skin patches": "dischromic patches",
    "discolored patches": "dischromic patches",
    "dischromic patches": "dischromic patches",

    "scarring": "scurring",
    "scars": "scurring",
    "skin scarring": "scurring",

    "shivering": "shivering",
    "shaking": "shivering",
    "trembling": "shivering",
    "tremor": "shivering",
    "tremors": "shivering",
    "chills": "chills",
    "cold sweat": "sweating",
    "cold sweats": "sweating",

    "fatigue": "fatigue",
    "tired": "fatigue",
    "tiredness": "fatigue",
    "weakness": "fatigue",
    "feeling weak": "fatigue",
    "low energy": "fatigue",
    "exhausted": "fatigue",
    "unusual fatigue": "fatigue",

    "hunger": "excessive hunger",
    "hungry": "excessive hunger",
    "very hungry": "excessive hunger",
    "extreme hunger": "excessive hunger",
    "always hungry": "excessive hunger",

    "dizziness": "dizziness",
    "feeling dizzy": "dizziness",
    "lightheaded": "dizziness",
    "light headed": "dizziness",

    "confusion": "lack of concentration",
    "trouble concentrating": "lack of concentration",
    "difficulty concentrating": "lack of concentration",
    "lack of concentration": "lack of concentration",

    "anxiety": "anxiety",
    "irritable": "irritability",
    "irritability": "irritability",

    "sweating": "sweating",
    "sweat": "sweating",
    "heavy sweating": "sweating",
    "excess sweating": "sweating",

    "heart racing": "palpitations",
    "palpitations": "palpitations",
    "fast heartbeat": "palpitations",
    "faster heart rate": "palpitations",

    "coughing": "cough",
    "cough": "cough",
    "persistent cough": "cough",

    "mucus cough": "mucoid sputum",
    "phlegm": "mucoid sputum",
    "sputum": "mucoid sputum",
    "mucoid sputum": "mucoid sputum",

    "stiff neck": "stiff neck",
    "loss of balance": "loss of balance",
    "sunken eyes": "sunken eyes",
    "dry lips": "drying and tingling lips",
    "tingling lips": "drying and tingling lips",
    "slurred speech": "slurred speech",
    "family history": "family history",
    "depression": "depression",
    "anxious": "anxiety",
}

alias_to_display: Dict[str, str] = {}

for display_feature in display_features:
    clean_disp = clean_text_for_match(display_feature)
    alias_to_display[clean_disp] = clean_disp
    alias_to_display[clean_disp.replace(" ", "")] = clean_disp
    alias_to_display[clean_disp.replace(" ", "_")] = clean_disp

for alias_phrase, target_feature in ALIASES_TO_REAL_FEATURES.items():
    add_alias_mapping(alias_to_display, alias_phrase, target_feature)

sorted_aliases = sorted(alias_to_display.keys(), key=len, reverse=True)

PROTECTED_TOKENS = {
    "light", "watery", "stomach", "belly", "tummy", "pimples", "pimple",
    "urinating", "urination", "peeing", "burning", "vomiting", "nausea",
    "headache", "sneezing", "breathing", "chest", "pain", "rash", "eyes",
    "eye", "frequent", "fever", "sweating", "dizzy", "dizziness",
    "stool", "loose", "hungry", "hunger", "pressure", "wheezing",
    "coughing", "watery", "itching", "tingling", "phlegm"
}

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

NON_SYMPTOM_LEFTOVER_WORDS = {
    "i", "im", "my", "me", "a", "an", "the", "and", "or", "but",
    "with", "without", "of", "in", "on", "at", "to", "for",
    "feel", "feels", "feeling", "like", "have", "has", "had",
    "is", "are", "was", "were", "be", "been", "being",
    "very", "really", "so", "too", "lot", "lower", "upper",
    "center", "left", "side", "body", "usual", "activities"
}


def auto_correct_text_input(raw_text: str) -> Tuple[str, List[Tuple[str, str]]]:
    cleaned = clean_text_for_match(raw_text)
    if not cleaned:
        return "", []

    tokens = cleaned.split()
    corrected_tokens: List[str] = []
    corrections: List[Tuple[str, str]] = []

    for token in tokens:
        corrected = token

        should_try = (
            len(token) >= 4
            and token not in TYPO_TOKEN_POOL
            and token not in PROTECTED_TOKENS
        )

        if should_try:
            close = get_close_matches(token, TYPO_TOKEN_POOL, n=1, cutoff=0.88)
            if close:
                candidate = close[0]
                if candidate and candidate[0] == token[0] and abs(len(candidate) - len(token)) <= 1:
                    corrected = candidate

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
        close = get_close_matches(phrase, candidate_pool, n=1, cutoff=0.92)
        if not close:
            continue

        matched_alias = close[0]
        display_key = alias_to_display.get(matched_alias)
        if not display_key:
            continue

        model_symptom = display_to_model.get(display_key)
        if model_symptom:
            add_detected_symptom(model_symptom, detected_model)


def clean_leftover_text(leftover_text: str) -> str:
    tokens = tokenize_text(leftover_text)
    filtered = [tok for tok in tokens if tok not in NON_SYMPTOM_LEFTOVER_WORDS]
    return " ".join(filtered).strip()


def closest_suggestions_for_unknown(leftover_text: str, max_suggestions: int = 6) -> List[str]:
    tokens = tokenize_text(leftover_text)
    if not tokens:
        return []

    suggestion_keys = list(alias_to_display.keys())
    ngrams = build_ngrams(tokens, 1, 4)
    suggestions: List[str] = []

    for phrase in ngrams:
        close = get_close_matches(phrase, suggestion_keys, n=3, cutoff=0.80)
        for item in close:
            resolved_display = alias_to_display.get(item, item)
            if resolved_display in canonical_display_keys and resolved_display not in suggestions:
                suggestions.append(resolved_display)
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
        display_key = alias_to_display.get(alias)
        if not display_key or display_key not in display_to_model:
            continue

        model_symptom = display_to_model[display_key]
        pattern = r"\b" + re.escape(alias) + r"\b"

        if re.search(pattern, remaining):
            add_detected_symptom(model_symptom, detected_model)
            remaining = re.sub(pattern, " ", remaining)

    partial_token_match_candidates(cleaned, detected_model)
    fuzzy_phrase_match_candidates(cleaned, detected_model)

    for model_symptom in detected_model:
        disp_key = model_to_display.get(model_symptom, "")
        if disp_key:
            pattern = r"\b" + re.escape(disp_key) + r"\b"
            remaining = re.sub(pattern, " ", remaining)

    remaining = clean_leftover_text(remaining)
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
# 11) DISEASE EVIDENCE TUNING (FINAL FIXED)
# ==============================
DISEASE_SIGNATURES: Dict[str, Dict[str, Union[List[str], float]]] = {
    "migraine": {
        "core": ["headache"],
        "support": ["nausea", "visual disturbances", "blurred and distorted vision", "dizziness", "fatigue", "irritability", "stiff neck"],
        "bonus": 0.12,
        "pair_bonus": 0.10,
        "triple_bonus": 0.12,
    },
    "hypertension": {
        "core": ["headache", "dizziness"],
        "support": ["chest pain", "palpitations", "blurred and distorted vision", "fatigue", "anxiety"],
        "bonus": 0.08,
        "pair_bonus": 0.08,
        "triple_bonus": 0.07,
    },
    "gastroenteritis": {
        "core": ["diarrhoea", "vomiting"],
        "support": ["dehydration", "nausea", "indigestion", "chills", "fatigue", "sunken eyes"],
        "bonus": 0.14,
        "pair_bonus": 0.12,
        "triple_bonus": 0.12,
    },
    "urinarytractinfection": {
        "core": ["burning micturition", "continuous feel of urine"],
        "support": ["bladder discomfort", "foul smell of urine", "fatigue", "nausea"],
        "bonus": 0.16,
        "pair_bonus": 0.13,
        "triple_bonus": 0.10,
    },
    "acne": {
        "core": ["pus filled pimples", "blackheads"],
        "support": ["skin rash", "scurring", "nodal skin eruptions"],
        "bonus": 0.16,
        "pair_bonus": 0.12,
        "triple_bonus": 0.08,
    },
    "hypoglycemia": {
        "core": ["excessive hunger", "sweating"],
        "support": ["dizziness", "fatigue", "palpitations", "anxiety", "irritability", "lack of concentration", "shivering"],
        "bonus": 0.12,
        "pair_bonus": 0.12,
        "triple_bonus": 0.10,
    },
    "heartattack": {
        "core": ["chest pain", "breathlessness"],
        "support": ["sweating", "nausea", "fatigue", "dizziness", "palpitations", "anxiety"],
        "bonus": 0.14,
        "pair_bonus": 0.12,
        "triple_bonus": 0.12,
    },
    "bronchialasthma": {
        "core": ["breathlessness", "cough"],
        "support": ["chest pain", "fatigue", "mucoid sputum"],
        "bonus": 0.12,
        "pair_bonus": 0.10,
        "triple_bonus": 0.10,
    },
    "allergy": {
        "core": ["continuous sneezing", "watering from eyes"],
        "support": ["itching", "skin rash", "cough", "fatigue", "drying and tingling lips"],
        "bonus": 0.14,
        "pair_bonus": 0.10,
        "triple_bonus": 0.08,
    },
    "fungalinfection": {
        "core": ["itching", "skin rash"],
        "support": ["dischromic patches", "nodal skin eruptions", "scurring"],
        "bonus": 0.14,
        "pair_bonus": 0.10,
        "triple_bonus": 0.08,
    },
}


def apply_disease_evidence_boost(probabilities: np.ndarray, selected_model_symptoms: List[str]) -> np.ndarray:
    adjusted = probabilities.astype(float).copy()
    symptom_set = set(selected_model_symptoms)

    if not symptom_set:
        return adjusted

    for disease_key, rules in DISEASE_SIGNATURES.items():
        class_index = disease_index_by_key.get(disease_key)
        if class_index is None:
            continue

        core = [s for s in rules["core"] if s in symptom_set]
        support = [s for s in rules["support"] if s in symptom_set]

        if not core and not support:
            continue

        boost = 0.0

        if len(core) >= 1:
            boost += float(rules["bonus"])
        if len(core) >= 2:
            boost += float(rules["pair_bonus"])
        if len(core) >= 1 and len(support) >= 1:
            boost += float(rules["pair_bonus"])
        if len(core) >= 2 and len(support) >= 1:
            boost += float(rules["triple_bonus"])
        if len(core) >= 1 and len(support) >= 2:
            boost += float(rules["triple_bonus"])

        adjusted[class_index] += boost

    migraine_idx = disease_index_by_key.get("migraine")
    hypertension_idx = disease_index_by_key.get("hypertension")
    heart_idx = disease_index_by_key.get("heartattack")
    asthma_idx = disease_index_by_key.get("bronchialasthma")
    hypo_idx = disease_index_by_key.get("hypoglycemia")
    allergy_idx = disease_index_by_key.get("allergy")
    fungal_idx = disease_index_by_key.get("fungalinfection")

    # Migraine fix
    if migraine_idx is not None and hypertension_idx is not None:
        if "headache" in symptom_set and (
            "visual disturbances" in symptom_set or
            "blurred and distorted vision" in symptom_set or
            "nausea" in symptom_set
        ):
            adjusted[migraine_idx] += 0.10
            adjusted[hypertension_idx] -= 0.03

    # Heart attack fix
    if heart_idx is not None and hypertension_idx is not None:
        if "chest pain" in symptom_set and "breathlessness" in symptom_set:
            adjusted[heart_idx] += 0.08
        if "sweating" in symptom_set and "chest pain" in symptom_set:
            adjusted[heart_idx] += 0.05
            adjusted[hypertension_idx] -= 0.02

    # Asthma fix
    if asthma_idx is not None and heart_idx is not None:
        if "breathlessness" in symptom_set and "cough" in symptom_set:
            adjusted[asthma_idx] += 0.08
        if "mucoid sputum" in symptom_set:
            adjusted[asthma_idx] += 0.06
            adjusted[heart_idx] -= 0.02

    # Hypoglycemia fix
    if hypo_idx is not None and hypertension_idx is not None:
        if "excessive hunger" in symptom_set and "sweating" in symptom_set:
            adjusted[hypo_idx] += 0.08
            adjusted[hypertension_idx] -= 0.02
        if "lack of concentration" in symptom_set:
            adjusted[hypo_idx] += 0.04

    # Final allergy / asthma / fungal balance
    if allergy_idx is not None:
        has_allergy_core = (
            "continuous sneezing" in symptom_set and
            "watering from eyes" in symptom_set
        )

        allergy_support_count = sum(
            1 for s in ["itching", "skin rash", "cough", "fatigue", "drying and tingling lips"]
            if s in symptom_set
        )

        fungal_specific_count = sum(
            1 for s in ["dischromic patches", "nodal skin eruptions", "scurring"]
            if s in symptom_set
        )

        # Hard protection for clear allergy pattern
        if has_allergy_core:
            adjusted[allergy_idx] += 0.35

            if allergy_support_count >= 1:
                adjusted[allergy_idx] += 0.10

            # Fungal should not steal allergy unless fungal-specific signs exist
            if fungal_idx is not None and fungal_specific_count == 0:
                adjusted[fungal_idx] -= 0.12

            # Asthma should not steal allergy from cough/breathlessness in this project context
            if asthma_idx is not None:
                adjusted[asthma_idx] -= 0.12

    adjusted = np.clip(adjusted, 1e-9, None)
    adjusted = adjusted / adjusted.sum()

    return adjusted

# ==============================
# 12) MODEL PREDICTION
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

    tuned_probabilities = apply_disease_evidence_boost(probabilities, selected_model_symptoms)

    k = min(k, len(tuned_probabilities))
    top_indices = np.argsort(tuned_probabilities)[::-1][:k]

    results: List[Tuple[str, float]] = []
    for idx in top_indices:
        disease_name = le.inverse_transform([idx])[0]
        confidence = float(tuned_probabilities[idx])
        results.append((disease_name, confidence))

    if not results:
        return {"error": "Prediction failed. Please try different symptoms."}

    return results


# ==============================
# 13) DECISION LAYER
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
# 14) TOP-3 REASONING + SUGGESTIONS UI HELPERS
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
                f"This is the top candidate because it received the highest combined score "
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
# 15) SESSION STATE
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
# 16) UI HEADER
# ==============================
st.markdown("""
<div class="hero">
    <h1>🩺 MediGuide AI</h1>
    <p>AI-powered disease prediction using structured symptoms</p>
</div>
""", unsafe_allow_html=True)


# ==============================
# 17) MAIN UI
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
        The app uses typo correction, exact matching, dataset-aligned synonym matching, token-overlap matching, close-match recovery, and a disease-evidence layer before final ranking.
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
# 18) FOOTER
# ==============================
st.markdown(
    '<div class="footer">Educational use only — not medical advice</div>',
    unsafe_allow_html=True
)