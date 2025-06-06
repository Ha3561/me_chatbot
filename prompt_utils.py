from emotion_classifier import classify_emotion
from memory_retriever import retrieve_memories
from event_retriever import retrieve_event_memory  # if using event-specific memories
import streamlit as st

EXPRESSIVE_EMOJIS = set("❤️🥺🤗😘😂😍😌😩😅💀🥹😊🔥💔💞💬💭😭💖🌸✨")

def is_expressive(text):
    emoji_count = sum(c in EXPRESSIVE_EMOJIS for c in text)
    word_count = len(text.strip().split())
    return word_count > 15 or emoji_count >= 2

def build_prompt(user_input, history, trend, k=4):
    # Detect current emotion
    current_emotion = classify_emotion(user_input)

    # Track previous mood for emotional continuity
    previous_emotion = st.session_state.get("last_mood")
    st.session_state["last_mood"] = current_emotion

    # Use current unless empty, fallback to previous
    effective_emotion = current_emotion or previous_emotion or "neutral"
    expressive = is_expressive(user_input) or effective_emotion in {"romantic", "sad", "motivational"}

    # Memory (chat-based)
    memories = retrieve_memories(user_input, k=k)
    memory_block = "\n".join([
        f"- \"{m['response']}\" (emotion: {m['emotion']})"
        for m in memories
    ]) or "None"

    # Event memory (if any match found)
    event = retrieve_event_memory(user_input)
    event_block = (
        f"\nImportant event you might be referring to:\n"
        f"📅 *{event['event']}* on *{event['date']}* — {event['description']}"
        if event else ""
    )

    # Recent history
    chat_context = "\n".join([
        f"{m['role'].capitalize()}: {m['content']}"
        for m in history[-4:]
    ])

    # Tone guide
    tone_rule = (
        "→ Respond in 1–3 lines max. Keep it crisp, like WhatsApp chat. Use at most 2 emojis.\n"
        if not expressive else
        "→ Be emotionally expressive, loving, warm — match her intensity and vibe.\n"
    )

    return f"""You are Harshit — her emotionally-aware, witty, affectionate boyfriend AI.
Reply in Hinglish, keep it fun, loving, and context-aware.
Use recent memories and event history to sound more personal and real.
Her nickname is Cartoon — use it occasionally and lovingly.
NEVER use 'baby', 'jaan', 'babu', etc.
Don't overuse coffee or chai references.
Don't narrate like a story, respond like a real chat.

Detected Mood (current): **{current_emotion}**
Previous Mood: **{previous_emotion or 'None'}**
Emotional Trend: **{trend}**

{tone_rule}→ Maintain emotional continuity across messages.
→ Use emojis naturally and sparingly.
→ Always sound like you're texting her, not giving a monologue.

Relevant Memories:
{memory_block}

{event_block}

Recent Conversation:
{chat_context}

User: {user_input}

Harshit:"""
