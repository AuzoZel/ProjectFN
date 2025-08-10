import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from collections import deque

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("Missing GOOGLE_API_KEY. Set it as an environment variable or in .env file.")
    st.stop()

genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-1.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

# Multi-turn buffer (max 2 turns)
class MultiTurnChat:
    def __init__(self, max_turns: int = 2):
        self.max_turns = max_turns
        self.buffer = deque()

    def add_user(self, text: str):
        self.buffer.append(('user', text))
        self._truncate()

    def add_assistant(self, text: str):
        self.buffer.append(('assistant', text))
        self._truncate()

    def _truncate(self):
        limit = self.max_turns * 2
        while len(self.buffer) > limit:
            self.buffer.popleft()

    def get_messages(self):
        return list(self.buffer)

    def as_generative_input(self):
        msgs = []
        system = {
            'role': 'system',
            'content': (
                'Bạn là FruitBot — một trợ lý thân thiện chuyên về trái cây. ' 
                'Trả lời ngắn gọn, chính xác, và kèm thông tin về hương vị, cách bảo quản, ' 
                'và cách chọn mua nếu có thể.'
            )
        }
        msgs.append(system)
        for role, text in self.buffer:
            if role == 'user':
                msgs.append({'role': 'user', 'content': text})
            else:
                msgs.append({'role': 'assistant', 'content': text})
        return msgs

# Streamlit UI
st.set_page_config(page_title="FruitBot", page_icon="🍓", layout="centered")
st.title("🍍 FruitBot — Chatbot về trái cây (Multi-turn)")

if 'chat' not in st.session_state:
    st.session_state['chat'] = MultiTurnChat(max_turns=2)
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

# Display chat messages
for role, msg in st.session_state['chat'].get_messages():
    if role == 'user':
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)

# Input area
with st.form(key='input_form', clear_on_submit=True):
    user_input = st.text_input("Bạn muốn hỏi về trái cây gì?", key='input')
    submitted = st.form_submit_button("Gửi")
    if submitted and user_input:
        # Add to history and call API
        st.session_state['chat'].add_user(user_input)
        messages = st.session_state['chat'].as_generative_input()
        # Call the generative model
        try:
            resp = model.generate_content(messages=messages, temperature=0.2, max_output_tokens=256)
            bot_text = resp.text if hasattr(resp, 'text') else str(resp)
        except Exception as e:
            bot_text = f"(Lỗi gọi API) {e}"
        st.session_state['chat'].add_assistant(bot_text)
        st.experimental_rerun()
