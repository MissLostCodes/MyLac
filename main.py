
import os
import asyncio
import streamlit as st
import nest_asyncio
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from modules.logger import get_logger
from modules.prompts import prompts
from modules.llm_init import create_llm

nest_asyncio.apply()

load_dotenv()
st.set_page_config(
    page_title="MyLac",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
        font-family: 'Inter', sans-serif;
    }

  
    .top-bar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: rgba(14, 17, 23, 0.95);
        backdrop-filter: blur(10px);
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem 3rem;
        border-bottom: 1px solid #222;
        z-index: 1000;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    .top-bar .stButton > button {
        background: linear-gradient(135deg, #00ffcc, #0066ff);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1.2rem;
        font-weight: bold;
        cursor: pointer;
        transition: 0.3s;
    }
    .top-bar .stButton > button:hover {
        background: linear-gradient(135deg, #00ffaa, #0044ff);
        transform: scale(1.05);
    }

    /* Add top padding so content doesn’t hide behind fixed bar */
    .block-container {
        padding-top: 6rem !important;
        padding-bottom: 7rem !important;
    }

    .main-title  {
        font-size: 5rem;
        font-weight: 1000;
        text-align: center;
        color: #00ffcc;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #d1d1d1;
        font-style: italic;
        margin-bottom: 2rem;
    }
    .chat-bubble-user {
        background: linear-gradient(135deg, #00bfa5, #00796b);
        color: white;
        padding: 0.8rem 1rem;
        border-radius: 20px 20px 5px 20px;
        max-width: 80%;
        margin-left: auto;
        margin-top: 10px;
    }
    .chat-bubble-bot {
        background: #262730;
        color: #e3e3e3;
        padding: 0.8rem 1rem;
        border-radius: 20px 20px 20px 5px;
        max-width: 80%;
        margin-right: auto;
        margin-top: 10px;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        color: #999;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

logger = get_logger("ML_AGENT")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)
embedder = OllamaEmbeddings(model="mxbai-embed-large")
OUTPUT_PARSER = StrOutputParser()


st.markdown("""
    <div class="top-bar">
        <div style="flex:1;">
    """, unsafe_allow_html=True)
st.markdown("<div class='main-title'>MyLac 🤖</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Where Machine Learning Books Come To Life ✨</div>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🟢 Start Chat", use_container_width=True):
        st.session_state.chat_active = True
        st.session_state.messages = []
        st.success("Chat started... Ask any Machine Learning-related question 🌙")
with col2:
    if st.button("🔴 End Chat", use_container_width=True):
        st.session_state.chat_active = False
        st.session_state.messages = []
        st.warning("Chat ended.")

st.markdown("</div></div>", unsafe_allow_html=True)

async def get_response(query, top_k=3):
    """Handles model call and query."""
    if "llm" not in st.session_state or st.session_state.llm is None:
        st.session_state.llm = await create_llm(
            model_provider=os.getenv("MODEL_PROVIDER", "ollama"),
            model_name=os.getenv("MODEL"),
        )

    query_emb = embedder.embed_query(query)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_emb,
        limit=top_k
    )

    points = results.points
    metadata = [
        {
            "textbook": r.payload.get("textbook", "Unknown"),
            "source_file": r.payload.get("source_file", "Unknown"),
            "page": r.payload.get("page", "N/A"),
            "score": r.score,
        }
        for r in points
    ]

    msg = f"Use the following textbook content to answer: \n\n{results}\n\n"
    system_msg = SystemMessage(content=prompts.SYSTEM_PROMPT + msg)
    human_msg = HumanMessage(content=query)
    response = await st.session_state.llm.ainvoke([system_msg, human_msg])
    parsed = OUTPUT_PARSER.invoke(response)
    return parsed, metadata

if st.session_state.get("chat_active", False):
    # Display chat history
    for msg in st.session_state.get("messages", []):
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-bubble-user'>🧑 {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-bot'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Type your message...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("🤔 Thinking..."):
            try:
                loop = asyncio.get_event_loop()
                response, meta = loop.run_until_complete(get_response(user_input))
                st.session_state.messages.append({
                    "role": "bot",
                    "content": f"{response}\n\n📘 *Context:* {meta}"
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "bot",
                    "content": f"⚠️ Error: {e}"
                })
        st.rerun()
else:
    st.info("Press **Start Chat** to begin talking with MyLac 🤖")

st.markdown("""
    <div class='footer'>
        Made with ❤️ by <b>MyLac</b> — "Where ML learns to talk."
    </div>
""", unsafe_allow_html=True)
