import streamlit as st
import base64
import time
from datetime import datetime
from langchain_community.chat_models import ChatGoogleGenerativeAI

from langchain.schema import HumanMessage
from prompt_utils import build_prompt
from memory_retriever import retrieve_memories
from emotion_classifier import classify_emotion 
from emotion_trend import get_emotional_trend

# ----------------- LLM SETUP ----------------- 
api_key = "AIzaSyDT8RSdELgozTIwFecfavjk3pJqrTYP3y4"
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        google_api_key=api_key  # Replace this
    )

llm = load_llm()

# ----------------- WALLPAPER -----------------
def set_background(jpg_file):
    with open(jpg_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """, unsafe_allow_html=True)

set_background("chat_wallpaper.jpg")

# ----------------- HEADER -----------------
def load_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

dp_base64 = load_image_base64("harshit_dp.jpeg")

st.markdown(f"""
<style>
.header {{
    background-color:#202c33;
    box-shadow: 0px 1px 4px rgba(0, 0, 0, 0.3);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 15px;
    border-bottom: 1px solid #444;
    position: fixed;
    top: 55px;  /* offset from Streamlit’s top bar */
    left: 50%;
    transform: translateX(-50%);
    width: 90%;
    max-width: 800px;  /* controls the width */
    border-radius: 8px;
    z-index: 9999;
}}
.header img {{
    border-radius: 50%;
    height: 40px;
    width: 40px;
    margin-right: 15px;
}}
.header .name {{
    color: white;
    font-weight: 600;
    font-size: 16px;
    margin-bottom: 2px;
}}
.header .status {{
    color: #ccc;
    font-size: 12px;
    margin-top: -4px;
}}
.header .right {{
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 10px;
    color: #ccc;
    font-size: 16px;
    cursor: pointer;
}}
.header .right button {{
    background-color: transparent;
    border: none;
    color: #ccc;
    font-size: 14px;
    cursor: pointer;
}}

/* Chat bubbles */
.chat-bubble {{
    padding: 10px 15px;
    border-radius: 15px;
    margin-bottom: 10px;
    max-width: 75%;
    line-height: 1.4;
    word-wrap: break-word;
}}
.user-bubble {{
    background-color: #005c4b;
    color: white;
    margin-left: auto;
    text-align: left;
}}
.bot-bubble {{
    background-color: #202c33;
    color: white;
    margin-right: auto;
    text-align: left;
}}
.meta {{
    font-size: 10px;
    color: #ccc;
    text-align: right;
    margin-top: 5px;
}}
</style>

<div class="header">
    <img src="data:image/jpeg;base64,{dp_base64}" />
    <div>
        <div class="name">Harshit</div>
        <div class="status">online</div>
    </div>
    <div class="right">
        <div id="clear-chat-container"></div>
    <span class="menu-icon">⋮</span> 
    </div>
</div>


</div>
""", unsafe_allow_html=True)

# Push the clear button into the header layout
with st.container():
    clear_chat = st.button("Clear Chat", key="clear", help="Clear all messages")

if clear_chat:
    st.session_state.messages = []
    st.rerun()  # refresh the page to reset scroll position

# Add vertical space below fixed header
st.markdown("<div style='height: 100px'></div>", unsafe_allow_html=True)

# ----------------- SESSION + CLEAR CHAT -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.query_params.get("clear") or st.session_state.get("clear", False):
    st.session_state.messages = []
    st.session_state.clear = False

# ----------------- CHAT DISPLAY -----------------
for msg in st.session_state.messages:
    is_user = msg["role"] == "user"
    bubble_class = "user-bubble" if is_user else "bot-bubble"
    st.markdown(f"""
        <div class="chat-bubble {bubble_class}">
            {msg['content']}
            <div class="meta">{datetime.now().strftime('%H:%M')} ✓✓</div>
        </div>
    """, unsafe_allow_html=True)

# ----------------- CHAT INPUT -----------------
user_input = st.chat_input("Tell me something...")

if user_input:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f"""
        <div class="chat-bubble user-bubble">
            {user_input}
            <div class="meta">{datetime.now().strftime('%H:%M')} ✓✓</div>
        </div>
    """, unsafe_allow_html=True)

    # Typing simulation
    typing_placeholder = st.empty()
    with typing_placeholder.container():
        st.markdown(f"""
            <div class="chat-bubble bot-bubble">
                Harshit is typing... 💭
                <div class="meta">{datetime.now().strftime('%H:%M')}</div>
            </div>
        """, unsafe_allow_html=True)

    time.sleep(0.5) # Simulate typing delay

    # Generate response
    trend = get_emotional_trend(st.session_state.messages) 
    print(f"Detected emotional trend: {trend}")
    prompt = build_prompt(user_input, st.session_state.messages, trend)

    response = llm([HumanMessage(content=prompt)]).content

    typing_placeholder.markdown(f"""
        <div class="chat-bubble bot-bubble">
            {response}
            <div class="meta">{datetime.now().strftime('%H:%M')} ✓✓</div>
        </div>
    """, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": response})
