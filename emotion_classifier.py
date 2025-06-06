# emotion_classifier.py

from transformers import pipeline

LABEL_MAP = {
    "sadness": "sad",
    "joy": "romantic",
    "love": "romantic",
    "anger": "teasing",
    "annoyance": "teasing",
    "neutral": "casual",
    "excitement": "casual",
    "gratitude": "supportive",
    "admiration": "supportive",
    "approval": "supportive",
    "pride": "motivational",
    "relief": "caring",
    "curiosity": "planning"
}

# Load only once when this file is imported
emo_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

def classify_emotion(text):
    preds = emo_classifier(text)[0]
    for pred in preds:
        mapped = LABEL_MAP.get(pred["label"])
        if mapped:
            return mapped
    return "casual"
