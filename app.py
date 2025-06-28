# ---- 1. Imports & Setup ------------------------------------------------------
import streamlit as st
import google.generativeai as genai
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
from gtts import gTTS
import base64, csv, os, json
from datetime import datetime
import numpy as np
import torch
import re, io
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt

if "history" not in st.session_state:
    st.session_state.history = []
if "view" not in st.session_state:               
    st.session_state.view = "Chat"

def vaani_reply(user_input):
    # ... (existing code up to raw)
    emotion = raw   
    happy_kw = ["birthday", "party", "excited", "cake", "gift", "celebrate"]
    if any(k in text_en.lower() for k in happy_kw):
        emotion = "joy"
    if any(k in text_en.lower() for k in confused_kw):
        emotion = "confused"
    elif any(k in text_en.lower() for k in sad_kw):
        emotion = "sadness"
    emoji = emotion_emoji.get(emotion, "🤖")

    # … prompt, Gemini, mood log (unchanged) …

    # -- TTS with BytesIO, return *three* items -----------
    buf = BytesIO()
    gTTS(final_out, lang='hi' if lang == "hi" else 'en').write_to_fp(buf)
    audio_b64 = base64.b64encode(buf.getvalue()).decode()

    return final_out, emoji, audio_b64           # only 3 ⇐

# ---- Page config & CSS -------------------------------------------------------
st.set_page_config(page_title="Vaani", page_icon="🪷", layout="centered")

st.markdown("""
<style>
html, body, [class*="css"]{
    font-family:'Segoe UI',Helvetica,sans-serif;
    font-size:18px;
    color:#F1F4FF;              /* light font */
}
[data-testid="stAppViewContainer"]{
    background:linear-gradient(135deg,#171821 0%,#292C3F 100%);
}

footer {visibility:hidden;}
[data-testid="stChatMessage"]{max-width:900px;padding:0.4rem 1rem;}

[data-testid="stChatMessage"]:has(div[data-testid="stMarkdown"]) div:nth-child(2){
    background:#2E3960;           /* user bubble dark blue */
    border-radius:14px;
    padding:0.6rem 1rem;
    margin-top:0.2rem;
}

[data-testid="stChatMessage"] .stChatMessageAvatar + div{
    background:#1F2436;           /* bot bubble */
    border:1px solid #3E4664;
    box-shadow:0 1px 3px rgba(0,0,0,0.4);
    border-radius:14px;
    padding:0.6rem 1rem;
    margin-top:0.2rem;
}
div[data-testid="baseButton-secondary"] > button {
    width: 100%;
    padding: 0.6rem 1.2rem;
    font-size: 16px;
    font-weight: 500;
    border-radius: 8px;
}
.stChatMessageAvatar img{
    border-radius:50%;
    width:40px;height:40px;
    object-fit:cover;
    border:2px solid #3E4664;
}

audio{width:180px;margin-top:0.3rem;}

::-webkit-scrollbar{width:6px;}
::-webkit-scrollbar-thumb{background:#555;border-radius:4px;}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
header[data-testid="stHeader"] {
    display: block !important;
}
</style>
""", unsafe_allow_html=True)

# ---- 2. Config / Keys --------------------------------------------------------
#genai.configure(api_key="AIzaSyCn3TcUxMWTK-y6d8i0Hwj5PGqDYwtrELU")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))   # <-- use env var for deployment
gemini = genai.GenerativeModel("gemini-1.5-flash", device='cpu')

classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",  # ← pick from table
    top_k=1,
    device=-1,
    model_kwargs={"torch_dtype": torch.float32}
)

embedder  = SentenceTransformer("all-MiniLM-L6-v2")
translator = Translator()

# ---- 3. Load RAG files -------------------------------------------------------
def load_rag(path): 
    with open(path,"r",encoding="utf-8") as f: return [ln.strip() for ln in f if ln.strip()]

rag_en, rag_hi = load_rag("rag_data.txt"), load_rag("rag_data_hi.txt")
emb_en, emb_hi = embedder.encode(rag_en), embedder.encode(rag_hi)

# ---- 4. Emojis & heuristics --------------------------------------------------
emotion_emoji = {"sadness":"😢","joy":"😊","anger":"😠","fear":"😨",
                 "love":"❤️","surprise":"😲","neutral":"😐","disgust":"🤢","confused":"😕"}
confused_kw = ["confused","lost","unclear","dont know what to do","don't know what to do",
               "what should i do","i dont get","i don't understand","i dont understand"]
sad_kw = ["off today", "off mood", "off right now",
          "not myself", "blah", "meh", "drained", "exhausted", "down today"]

IRRELEVANT_PHRASES = {"hi","hello","hey","hola","thanks","thank you","ok","okay","hmm"}
MIN_WORDS_FOR_LOG  = 1

# ---- 5. Session State --------------------------------------------------------
if "history" not in st.session_state: st.session_state.history=[]

# Handle new chat trigger via query param (modern API)
if st.query_params.get("newchat") == "1":
    st.session_state.history.clear()
    st.query_params.clear()  # Clear the URL param cleanly
    st.rerun()

# ---- 6. CSV Mood Logger ------------------------------------------------------
def log_mood(emotion, lang, text):
    if not os.path.exists("mood_log.csv"):
        with open("mood_log.csv","w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp","emotion","lang","text"])
    with open("mood_log.csv","a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.utcnow().isoformat(),emotion,lang,text])
# ---- Utility: safe gTTS with chunking ---------------------------------------
def text_to_base64_mp3(text: str, lang: str = "en") -> str:
    """Return base64-encoded MP3 for up to ~1000-char text, chunked for gTTS."""
    

    # ① Clean problematic characters
    text = re.sub(r"[`´“”]", "", text)

    # ② Split into 200-char chunks on sentence boundaries
    sentences = re.split(r"(?<=[.!?]) +", text)
    chunks, buf, current = [], [], ""
    for s in sentences:
        if len(current) + len(s) < 200:
            current += (" " if current else "") + s
        else:
            chunks.append(current); current = s
    if current: chunks.append(current)

    # ③ Synthesize each chunk
    mp3_bytes = io.BytesIO()
    for part in chunks:
        tts = gTTS(part.strip(), lang=lang)
        tts.write_to_fp(mp3_bytes)
    b64 = base64.b64encode(mp3_bytes.getvalue()).decode()
    return f"data:audio/mp3;base64,{b64}"

# ---- 7. Chatbot core ---------------------------------------------------------
def vaani_reply(user_input):
    lang     = translator.detect(user_input).lang
    text_en  = translator.translate(user_input,dest="en").text if lang!="en" else user_input
    
    raw      = classifier(text_en)[0][0]["label"].lower()
    emotion  = raw  
    happy_kw = ["birthday", "party", "excited", "cake", "gift", "celebrate"]
    if any(k in text_en.lower() for k in happy_kw):
        emotion = "joy"
    elif any(k in text_en.lower() for k in confused_kw):
        emotion = "confused"
    elif any(k in text_en.lower() for k in sad_kw):
        emotion = "sadness"

    #emotion  = "confused" if any(k in text_en.lower() for k in confused_kw) else raw
    emoji    = emotion_emoji.get(emotion,"🤖")

    rag, emb = (rag_hi, emb_hi) if lang=="hi" else (rag_en,emb_en)
    top      = rag[np.argmax(cosine_similarity(embedder.encode([text_en]), emb))]

    ctx_lines = [m for _, m, _ in st.session_state.history[-4:]]

    ctx            = "\n".join(ctx_lines)
    history_block  = f"Chat History: {ctx}\n" if ctx else ""

    prompt = (
        "You are a friendly, empathetic chatbot.\n"
        f"The user is feeling {emotion}. Respond warmly (no pet names).\n"
        "• Do not repeat any word more than twice.\n"
        "• Use paragraphs ≤ 3 lines.\n"
        f'Context: "{top}"\n{history_block}User: {text_en}'
    )
    response = gemini.generate_content(
        prompt,
        generation_config={"temperature":0.4,"top_p":0.9,"max_output_tokens":300}
    ).text.strip()

    final_out = translator.translate(response,dest=lang).text if lang!="en" else response

    # mood log
    if len(text_en.split())>=MIN_WORDS_FOR_LOG and text_en.lower() not in IRRELEVANT_PHRASES:
        log_mood(emotion,lang,text_en)
    
    buf = BytesIO()
    gTTS(
        final_out,
        lang='hi' if lang == "hi" else 'en'
    ).write_to_fp(buf)       # write MP3 directly to memory

    audio_bytes = buf.getvalue()       # <-- pure bytes object
    return final_out, emoji, audio_bytes
    
    
    

# ---- 8. Streamlit UI ---------------------------------------------------------
# ---- Floating Start New Chat button ----------------------------------------
st.markdown("""
    <style>
    #new-chat-btn {
        position: absolute;
        top: 20px;
        right: 25px;
        background-color: #2E3B6A;
        color: white;
        padding: 6px 10px;
        border-radius: 8px;
        font-size: 18px;
        border: none;
        cursor: pointer;
        z-index: 9999;
    }
    #new-chat-btn:hover {
        background-color: #445599;
    }
    </style>
    <form action="?newchat=1" method="get">
        <button id="new-chat-btn" type="submit">🔁</button>
    </form>
""", unsafe_allow_html=True)

# ---- Sidebar with Controls ---------------------------------------------------
with st.sidebar:
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #1A237E;
            color: white;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 500;
            padding: 0.5rem 1rem;
            width: 100%;
            transition: background 0.2s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #2D3A88;
        }
        [data-testid="stRadio"] > div > label > div:first-child {
        background-color: #8e44ad !important; /* Purple dot */
        border: 2px solid #ccc !important;
        }
        div[data-testid="stToggle"] > div > div {
            background-color: #8e44ad !important;
        }
        div[data-testid="stToggle"] > div > div > div {
            background-color: #f3d9ff !important;
        }

        
        div[data-testid="stToggle"] > div > div > div {
        background-color: #c09bf2 !important;  /* light purple thumb */
        }
            
        [data-testid="stToggle"] > div:hover,
        [data-testid="stRadio"] > div > label:hover > div:first-child {
        box-shadow: 0 0 0 2px rgba(142, 68, 173, 0.5);
    }        
        </style>
    """, unsafe_allow_html=True)

    if st.button("🔁  Start New Chat", use_container_width=True):
        st.session_state.history.clear()
        st.rerun()
    # -- Now comes the Controls section --
    st.header("🛠️ Controls")
    st.radio(
        "View",
        ["Chat", "Mood Dashboard"],
        horizontal=True,
        key="view",  # 🔑 This tells Streamlit to manage the value in session_state
        format_func=lambda x: "💬 Chat" if x == "Chat" else "📊 Dashboard"
  )
    # Inject CSS for purple toggle
    st.markdown("""
        <style>
        div[data-testid="stToggle"] > label div {
            background-color: #9c27b0 !important;  /* Purple when ON */
        }
        div[data-testid="stToggle"] > label input:checked + div {
            background-color: #9c27b0 !important;  /* Purple when checked */
        }
        </style>
    """, unsafe_allow_html=True)

    speak_audio = st.toggle("🔊 Auto-play voice", value=True)
    st.session_state.voice_enabled = speak_audio

    if st.checkbox("📜 Show Chat History"):
        st.subheader("Previous Messages")
        for role, msg, _ in st.session_state.get("history", []):
            label = "👤 You:" if role == "user" else "🤖 Vaani:"
            st.markdown(f"**{label}** {msg}")


if st.session_state.view == "Chat":   
    st.title("🪷  Vaani")
    st.markdown(
        '<p style="font-size:24px; '
        'font-weight:300; margin-top:-10px; color:#F1F4FF;">'
        'A voice that understands you.</p>',
        unsafe_allow_html=True
    )
    
    # Render previous history
    for role, msg, av in st.session_state.history:
        with st.chat_message("assistant" if role=="bot" else "user", avatar=av):
            st.markdown(msg)

    if not st.session_state.history:
        with st.chat_message("assistant", avatar="👩‍🦰"):
            st.markdown("👋 **Hello, I’m Vaani!**  \n"
                        "I’m here to listen and support you in English or Hindi.  \n"
                        "_Type anything to start, or **exit** to clear the chat._")

    user_input = st.chat_input("Type your message…")

    if user_input:
        if user_input.lower().strip() in ("exit","quit"):
            st.session_state.history.clear()
            st.rerun()

        # user bubble
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)
        # BOT bubble
        
        with st.chat_message("assistant", avatar="💬"):
            typing_placeholder = st.empty()
            typing_placeholder.markdown("*Vaani is typing…*")

        # ⚡ Generate reply
        reply, emoji, audio_data = vaani_reply(user_input)

        # Remove indicator & show real message
        typing_placeholder.empty()
        with st.chat_message("assistant", avatar=emoji):
            st.markdown(reply)
            if speak_audio:                       # <-- toggle check
                st.audio(audio_data, format="audio/mp3")

        
        # update history (role, msg)
        st.session_state.history.append(("user", user_input, "🧑"))
        st.session_state.history.append(("bot",  reply,       emoji))
# ---------------- MOOD DASHBOARD ----------------

elif st.session_state.view == "Mood Dashboard":
    st.title("📊 Mood Dashboard")

    if not os.path.exists("mood_log.csv"):
        st.info("No mood data yet – chat a little first!")
    else:
        df = pd.read_csv("mood_log.csv", parse_dates=["timestamp"])
        df["date"]  = df["timestamp"].dt.date
        df["month"] = df["timestamp"].dt.to_period("M")

        period = st.selectbox("📅 Aggregate by", ["Daily", "Monthly"])
        group  = "date" if period == "Daily" else "month"
        positive = {"joy", "love"}
        neutral = {"neutral", "confused", "surprise"}
        negative = {"sadness", "fear", "anger", "disgust"}

        def classify_sentiment(emotion):
            if emotion in positive:
                return "Positive"
            elif emotion in neutral:
                return "Neutral"
            else:
                return "Negative"

        df["sentiment"] = df["emotion"].apply(classify_sentiment)

        agg = df.groupby([group, "sentiment"]).size().unstack(fill_value=0)

        # ----------- LINE CHART -----------
        import matplotlib.dates as mdates
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = {
            "Positive": "#00c853",    # green
            "Neutral": "#ffd600",     # yellow
            "Negative": "#d50000"     # red
        }
        
        agg.plot(
            ax=ax,
            marker='o',               # ✅ show dots for each data point
            markersize=6,
            linewidth=2,
            color=[colors.get(col, "#888888") for col in agg.columns]
        )

        if group == "date":
            agg.index = pd.to_datetime(agg.index)
            ax.set_xticks(agg.index[::max(len(agg)//10, 1)])  # Show fewer ticks
            ax.set_xticklabels(agg.index[::max(len(agg)//10, 1)].strftime('%b %d'), rotation=45)
        
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

        ax.set_title(f"{period} Mood Trend", fontsize=18, fontweight='bold')
        ax.set_ylabel("🧠 Mood Count", fontsize=13)
        ax.set_xlabel(period, fontsize=13)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(title="Sentiment", fontsize=12, title_fontsize=13)

        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#FAFAFA')

        st.pyplot(fig)

        # ----------- PIE CHART -----------
        st.subheader("📈 Overall Mood Distribution")
        mood_counts = df["emotion"].value_counts()
        pie_colors = plt.cm.Paired(np.linspace(0, 1, len(mood_counts)))

        fig2, ax2 = plt.subplots()
        wedges, texts, autotexts = ax2.pie(
            mood_counts,
            labels=mood_counts.index.str.title(),
            autopct="%1.1f%%",
            startangle=140,
            colors=pie_colors,
            textprops={"fontsize": 12}
        )
        ax2.axis("equal")  # Equal aspect ratio = circle
        ax2.set_facecolor('#FAFAFA')
        fig2.patch.set_facecolor('#FFFFFF')
        st.pyplot(fig2)

        # ----------- TOP EMOJIS -----------
        st.subheader("💬 Top Mood Emojis")
        emoji_map = {
            "sadness":"😢","joy":"😊","anger":"😠","fear":"😨",
            "love":"❤️","surprise":"😲","neutral":"😐","disgust":"🤢","confused":"😕"
        }
        df["emoji"] = df["emotion"].map(emoji_map)
        top = df["emoji"].value_counts().head(5)

        st.markdown("<div style='font-size:28px;'>", unsafe_allow_html=True)
        for emoji, count in top.items():
            st.markdown(f"<span style='margin-right:15px;'>{emoji} × {count}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        
