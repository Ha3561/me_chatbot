from collections import Counter 
from emotion_classifier import classify_emotion

def get_emotional_trend(history, N=5):
    # Extract user messages only
    recent_msgs = [m["content"] for m in history if m["role"] == "user"][-N:]
    
    # Classify emotions for those messages
    emotions = [classify_emotion(msg) for msg in recent_msgs]

    # Count frequencies
    counter = Counter(emotions)

    # Return most common one
    if counter:
        return counter.most_common(1)[0][0]
    else:
        return "neutral"
