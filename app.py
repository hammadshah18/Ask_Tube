import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import google.generativeai as genai
import whisper
import tempfile
import yt_dlp
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langdetect import detect
import os
import time
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

load_dotenv()

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="YT Insight — AI Video Chat",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# CUSTOM CSS
# --------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Reset & Root ─────────────────────────────────── */
:root {
    --bg:        #0a0a0f;
    --surface:   #111118;
    --card:      #16161f;
    --border:    #1e1e2e;
    --accent:    #e8ff47;
    --accent2:   #47c7ff;
    --accent3:   #ff6b47;
    --text:      #e8e8f0;
    --muted:     #6b6b80;
    --radius:    14px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
}

/* ── Streamlit chrome ──────────────────────────────── */
.stApp { background: var(--bg) !important; }
header[data-testid="stHeader"] { background: transparent !important; }
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* ── Scrollbar ─────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

/* ── Hero / Brand ──────────────────────────────────── */
.hero {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(232,255,71,0.1);
    border: 1px solid rgba(232,255,71,0.3);
    color: var(--accent);
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 0.3rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.1;
    color: var(--text);
    margin: 0 0 0.6rem;
}
.hero-title span {
    color: var(--accent);
    display: inline-block;
}
.hero-sub {
    font-size: 1rem;
    color: var(--muted);
    font-weight: 300;
    letter-spacing: 0.01em;
}

/* ── Cards ─────────────────────────────────────────── */
.glass-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.stat-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.stat-pill {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 0.35rem 0.9rem;
    font-size: 0.82rem;
    color: var(--muted);
}
.stat-pill .val {
    color: var(--accent);
    font-family: 'Syne', sans-serif;
    font-weight: 700;
}

/* ── URL Input ─────────────────────────────────────── */
div[data-testid="stTextInput"] input {
    background: #1a1a26 !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.8rem 1rem !important;
    transition: border-color 0.2s;
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(232,255,71,0.08) !important;
}

/* ── Buttons ───────────────────────────────────────── */
div[data-testid="stButton"] button {
    background: var(--accent) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.7rem 1.6rem !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    width: 100%;
}
div[data-testid="stButton"] button:hover {
    background: #d4eb3a !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(232,255,71,0.25) !important;
}
div[data-testid="stButton"] button:active { transform: translateY(0) !important; }

/* ── Sidebar labels ───────────────────────────────── */
.sidebar-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.sidebar-section {
    margin-bottom: 1.4rem;
}

/* ── Thumbnails ─────────────────────────────────────── */
.thumb-wrap {
    position: relative;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
}
.thumb-wrap img {
    width: 100%;
    display: block;
}
.thumb-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(to top, rgba(10,10,15,0.7) 0%, transparent 60%);
}
.thumb-play {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%,-50%);
    width: 52px; height: 52px;
    background: rgba(232,255,71,0.9);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    color: #0a0a0f;
}

/* ── Chat Messages ─────────────────────────────────── */
.chat-wrap {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding-bottom: 1rem;
}
.msg {
    display: flex;
    gap: 0.8rem;
    align-items: flex-start;
    animation: fadeUp 0.3s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.msg.user { flex-direction: row-reverse; }
.avatar {
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.avatar.ai   { background: rgba(232,255,71,0.12); }
.avatar.user { background: rgba(71,199,255,0.12); }
.bubble {
    max-width: 78%;
    padding: 0.85rem 1.1rem;
    border-radius: 14px;
    font-size: 0.93rem;
    line-height: 1.65;
}
.bubble.ai {
    background: var(--card);
    border: 1px solid var(--border);
    border-top-left-radius: 4px;
    color: var(--text);
}
.bubble.user {
    background: rgba(71,199,255,0.1);
    border: 1px solid rgba(71,199,255,0.2);
    border-top-right-radius: 4px;
    color: var(--text);
}
.bubble.ai pre, .bubble.ai code {
    background: rgba(0,0,0,0.3);
    border-radius: 6px;
    font-size: 0.83rem;
}

/* ── Typing cursor ─────────────────────────────────── */
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
}
.cursor {
    display: inline-block;
    width: 2px; height: 1em;
    background: var(--accent);
    margin-left: 2px;
    vertical-align: text-bottom;
    animation: blink 0.7s infinite;
}

/* ── Status pills ──────────────────────────────────── */
.status-ok {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.25);
    color: #4ade80;
    border-radius: 100px;
    padding: 0.25rem 0.8rem;
    font-size: 0.78rem;
    font-weight: 500;
}
.status-err {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.25);
    color: #f87171;
    border-radius: 100px;
    padding: 0.25rem 0.8rem;
    font-size: 0.78rem;
}

/* ── Divider ───────────────────────────────────────── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 1.4rem 0;
}

/* ── Chat input row ────────────────────────────────── */
div[data-testid="stTextInput"] { margin-bottom: 0 !important; }

/* ── Sliders ───────────────────────────────────────── */
div[data-testid="stSlider"] .stSlider > div > div > div {
    background: var(--accent) !important;
}

/* ── Spinner ───────────────────────────────────────── */
div[data-testid="stSpinner"] { color: var(--accent) !important; }

/* ── Scrollable chat area ──────────────────────────── */
.chat-scroll {
    max-height: 55vh;
    overflow-y: auto;
    padding-right: 4px;
}
.chat-scroll::-webkit-scrollbar { width: 3px; }
.chat-scroll::-webkit-scrollbar-thumb { background: var(--border); }

/* ── Summary block ─────────────────────────────────── */
.summary-box {
    background: linear-gradient(135deg, rgba(232,255,71,0.05) 0%, rgba(71,199,255,0.04) 100%);
    border: 1px solid rgba(232,255,71,0.15);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    font-size: 0.92rem;
    line-height: 1.7;
    color: var(--text);
}
.summary-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.7rem;
}

/* ── Feature pills in sidebar ──────────────────────── */
.feat {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0;
    font-size: 0.83rem;
    color: var(--muted);
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.feat:last-child { border-bottom: none; }
.feat-icon {
    width: 26px; height: 26px;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem;
    background: rgba(255,255,255,0.04);
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# CACHED RESOURCES
# --------------------------
@st.cache_resource(show_spinner=False)
def load_whisper():
    return whisper.load_model("tiny")

@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_llm():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    return genai.GenerativeModel("gemini-2.5-flash")

# --------------------------
# UTILS (same logic as FastAPI)
# --------------------------
def extract_video_id(url: str):
    parsed = urlparse(url)
    if parsed.hostname == "youtu.be":
        return parsed.path[1:]
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/")[2]
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
    return None

def whisper_transcribe_video(video_url):
    try:
        temp_dir = tempfile.gettempdir()
        audio_template = os.path.join(temp_dir, "yt_audio")
        audio_path = audio_template + ".mp3"
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_template,
            "noplaylist": True,
            "quiet": True,
            "continuedl": False,
            "nopart": True,
            "retries": 10,
            "fragment_retries": 10,
            "socket_timeout": 30,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        if not os.path.exists(audio_path):
            raise Exception("Audio file was not downloaded")
        whisper_model = load_whisper()
        result = whisper_model.transcribe(audio_path)
        transcript_text = ""
        for segment in result["segments"]:
            start = segment["start"]
            text = segment["text"]
            transcript_text += f"[{start:.2f}s] {text} "
        os.remove(audio_path)
        return transcript_text
    except Exception as e:
        st.error(f"Whisper error: {e}")
        return None

def get_transcript(video_id, video_url=None):
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id)
        text = ""
        for chunk in transcript_list:
            start = getattr(chunk, "start", 0)
            content = getattr(chunk, "text", "")
            text += f"[{start:.2f}s] {content} "
        return text, "YouTube Captions"
    except (TranscriptsDisabled, NoTranscriptFound):
        pass
    if video_url:
        result = whisper_transcribe_video(video_url)
        return result, "Whisper ASR"
    return None, None

def chunk_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def build_vector_store(chunks):
    embedder = load_embedder()
    embeddings = embedder.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, chunks

def retrieve_chunks(index, chunks, question, k=5):
    embedder = load_embedder()
    q_emb = embedder.encode([question])
    D, I = index.search(np.array(q_emb), k)
    return [chunks[i] for i in I[0]]

def summarize_transcript(transcript, llm):
    prompt = f"""
Summarize the following YouTube video transcript in 4-6 clear, informative sentences.
Capture the main topic, key points, and any notable conclusions.
Be concise but comprehensive.

Transcript:
{transcript[:4000]}
"""
    response = llm.generate_content(prompt)
    return response.text

def stream_answer(text):
    """Yield characters with fast typing effect."""
    for char in text:
        yield char
        time.sleep(0.008)

# --------------------------
# SESSION STATE
# --------------------------
for key, default in [
    ("processed", False),
    ("video_id", None),
    ("thumbnail", None),
    ("index", None),
    ("chunks", None),
    ("transcript_source", None),
    ("summary", None),
    ("chat_history", []),
    ("chunk_size", 200),
    ("top_k", 5),
    ("q_input_value", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --------------------------
# SIDEBAR
# --------------------------
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem;'>
        <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:#e8e8f0;'>
            ▶ YT<span style='color:#e8ff47;'>Insight</span>
        </div>
        <div style='font-size:0.75rem;color:#6b6b80;margin-top:0.2rem;'>AI-Powered Video Intelligence</div>
    </div>
    <div style='height:1px;background:linear-gradient(90deg,transparent,#1e1e2e,transparent);margin:0.8rem 0 1.2rem;'></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><div class="sidebar-label">Settings</div></div>', unsafe_allow_html=True)

    chunk_size = st.slider("Chunk Size", 100, 500, 200, 50,
        help="Words per chunk — smaller = more precise answers")
    top_k = st.slider("Chunks to Retrieve", 3, 10, 5, 1,
        help="More chunks = more context, slower response")

    st.session_state.chunk_size = chunk_size
    st.session_state.top_k = top_k

    st.markdown('<div style="height:1.2rem;"></div>', unsafe_allow_html=True)

    if st.session_state.processed:
        st.markdown(f"""
        <div class="glass-card" style="padding:1rem;">
            <div class="sidebar-label" style="margin-bottom:0.8rem;">Video Loaded</div>
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.6rem;">
                <span class="status-ok">● Ready</span>
            </div>
            <div style="font-size:0.78rem;color:#6b6b80;">
                Source: <span style="color:#e8ff47;">{st.session_state.transcript_source}</span><br>
                Chunks: <span style="color:#e8ff47;">{len(st.session_state.chunks)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑 Clear & Reset"):
            st.session_state.processed = False
            st.session_state.video_id = None
            st.session_state.thumbnail = None
            st.session_state.index = None
            st.session_state.chunks = None
            st.session_state.transcript_source = None
            st.session_state.summary = None
            st.session_state.chat_history = []
            st.rerun()

    st.markdown('<div style="height:1.4rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Features</div>', unsafe_allow_html=True)
    features = [
        ("📹","YouTube Captions"),
        ("🎤","Whisper ASR Fallback"),
        ("🌍","Multi-language"),
        ("🔍","Semantic FAISS Search"),
        ("🤖","Gemini 2.5 Flash"),
        ("✍️","Typing Animation"),
    ]
    feats_html = '<div style="margin-top:0.3rem;">'
    for icon, label in features:
        feats_html += f'<div class="feat"><div class="feat-icon">{icon}</div>{label}</div>'
    feats_html += '</div>'
    st.markdown(feats_html, unsafe_allow_html=True)

# --------------------------
# MAIN CONTENT
# --------------------------
st.markdown("""
<div class="hero">
    <div class="hero-badge">Powered by Gemini AI + FAISS</div>
    <h1 class="hero-title">Chat with Any<br><span>YouTube Video</span></h1>
    <p class="hero-sub">Enter a URL → get instant insights, summaries & answers</p>
</div>
""", unsafe_allow_html=True)

# ── URL Input ──────────────────────────────────────────────
col_url, col_btn = st.columns([5, 1.2])
with col_url:
    video_url = st.text_input(
        "", placeholder="🔗  Paste YouTube URL here — e.g. https://youtu.be/dQw4w9WgXcQ",
        label_visibility="collapsed"
    )
with col_btn:
    process_clicked = st.button("▶  Process")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Process Video ─────────────────────────────────────────
if process_clicked and video_url:
    vid = extract_video_id(video_url)
    if not vid:
        st.markdown('<span class="status-err">✕ Invalid YouTube URL</span>', unsafe_allow_html=True)
    else:
        with st.spinner("🔍 Fetching transcript…"):
            transcript, source = get_transcript(vid, video_url)

        if not transcript:
            st.markdown('<span class="status-err">✕ Could not retrieve transcript</span>', unsafe_allow_html=True)
        else:
            with st.spinner("🧠 Building vector index…"):
                chunks = chunk_text(transcript, st.session_state.chunk_size)
                index, chunks = build_vector_store(chunks)

            with st.spinner("✍️ Generating summary…"):
                llm = load_llm()
                summary = summarize_transcript(transcript, llm)

            st.session_state.update({
                "processed": True,
                "video_id": vid,
                "thumbnail": f"https://img.youtube.com/vi/{vid}/maxresdefault.jpg",
                "index": index,
                "chunks": chunks,
                "transcript_source": source,
                "summary": summary,
                "chat_history": [],
            })
            st.rerun()

# ── Video Panel + Chat ─────────────────────────────────────
if st.session_state.processed:
    left, right = st.columns([1, 2], gap="large")

    # Left – thumbnail + stats + summary
    with left:
        st.markdown(f"""
        <div class="thumb-wrap">
            <img src="{st.session_state.thumbnail}" alt="thumbnail"/>
            <div class="thumb-overlay"></div>
            <div class="thumb-play">▶</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-pill"><span class="val">{len(st.session_state.chunks)}</span> chunks</div>
            <div class="stat-pill"><span class="val">{st.session_state.transcript_source}</span></div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.summary:
            st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="summary-box">
                <div class="summary-title">📋 Video Summary</div>
                {st.session_state.summary}
            </div>
            """, unsafe_allow_html=True)

    # Right – chat
    with right:
        st.markdown("""
        <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;
                    color:#e8e8f0;margin-bottom:1rem;letter-spacing:0.01em;'>
            💬 Ask Anything About This Video
        </div>
        """, unsafe_allow_html=True)

        # ── Render full chat history ──────────────────────
        if st.session_state.chat_history:
            chat_html = '<div class="chat-scroll"><div class="chat-wrap">'
            for msg in st.session_state.chat_history:
                role = msg["role"]
                bubble_cls = "ai" if role == "assistant" else "user"
                icon = "✦" if role == "assistant" else "◈"
                content = msg["content"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                chat_html += f"""
                <div class="msg {bubble_cls}">
                    <div class="avatar {bubble_cls}">{icon}</div>
                    <div class="bubble {bubble_cls}">{content}</div>
                </div>"""
            chat_html += '</div></div>'
            st.markdown(chat_html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:2rem 1rem;color:#6b6b80;
                        border:1px dashed #1e1e2e;border-radius:12px;margin-bottom:1rem;">
                <div style="font-size:1.4rem;margin-bottom:0.4rem;">💬</div>
                <div style="font-size:0.85rem;">Ask your first question about this video</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Typing animation placeholder (below history) ──
        typing_placeholder = st.empty()

        # ── Question input ────────────────────────────────
        st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
        q_col, s_col = st.columns([5, 1.2])
        with q_col:
            question = st.text_input(
                "", placeholder="Ask a question about the video…",
                key="q_input",
                value=st.session_state.q_input_value,
                label_visibility="collapsed"
            )
        with s_col:
            ask_btn = st.button("Ask ✦")

        if ask_btn and question.strip():
            current_q = question.strip()

            # Save user message immediately
            st.session_state.chat_history.append({"role": "user", "content": current_q})

            # Retrieve context
            relevant = retrieve_chunks(
                st.session_state.index,
                st.session_state.chunks,
                current_q,
                k=st.session_state.top_k
            )
            context = "\n".join(relevant)

            try:
                lang = detect(current_q)
            except:
                lang = "en"

            prompt = f"""
You are a knowledgeable AI assistant answering questions about a YouTube video.
You have two sources of information to craft the best possible answer:

1. VIDEO TRANSCRIPT CONTEXT (retrieved segments):
{context}

2. YOUR OWN KNOWLEDGE: Use your broad training knowledge to enrich, explain, 
   and expand on what the transcript says. If the transcript is incomplete or 
   ambiguous, fill in gaps with accurate general knowledge. Always prioritize 
   transcript content for video-specific facts, but freely add helpful context, 
   explanations, examples, or background from your own knowledge.

User question: {current_q}

Guidelines:
- Combine transcript evidence with your own knowledge for the richest answer.
- If the transcript directly answers the question, lead with that, then enrich it.
- If the transcript doesn't cover it, answer from your knowledge and note it's general info.
- Use timestamps from the transcript when referencing specific moments.
- Answer in the SAME language as the question (Urdu → Urdu, Roman Urdu → Roman Urdu, 
  Sindhi/Roman Sindhi → same, English → English).
- Be clear, helpful, and thorough but concise.
"""
            llm = load_llm()

            # Show blinking cursor while generating
            typing_placeholder.markdown("""
            <div class="msg ai" style="margin-top:0.5rem;">
                <div class="avatar ai">✦</div>
                <div class="bubble ai"><span class="cursor"></span></div>
            </div>
            """, unsafe_allow_html=True)

            response = llm.generate_content(prompt)
            full_answer = response.text

            # Stream typing animation char by char
            displayed = ""
            for char in full_answer:
                displayed += char
                typing_placeholder.markdown(f"""
                <div class="msg ai">
                    <div class="avatar ai">✦</div>
                    <div class="bubble ai">{displayed}<span class="cursor"></span></div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.008)

            # Clear typing placeholder
            typing_placeholder.empty()

            # Clear input box, save answer, rerun to refresh
            st.session_state.q_input_value = ""
            st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
            st.rerun()

# ── Empty state ─────────────────────────────────────────────
elif not process_clicked:
    st.markdown("""
    <div style="text-align:center;padding:2.5rem 1rem 3rem;color:#6b6b80;">
        <div style="font-size:3rem;margin-bottom:1rem;">▶</div>
        <div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
                    color:#e8e8f0;margin-bottom:0.5rem;">
            No video loaded yet
        </div>
        <div style="font-size:0.88rem;">
            Paste a YouTube URL above and click <strong style="color:#e8ff47;">Process</strong> to get started
        </div>
    </div>
    <div style="display:flex;justify-content:center;gap:2rem;flex-wrap:wrap;padding-bottom:2rem;">
        <div class="glass-card" style="width:200px;text-align:center;padding:1.4rem;">
            <div style="font-size:1.8rem;margin-bottom:0.5rem;">📋</div>
            <div style="font-family:Syne,sans-serif;font-weight:700;font-size:0.85rem;
                        color:#e8e8f0;">Auto Summary</div>
            <div style="font-size:0.78rem;color:#6b6b80;margin-top:0.3rem;">
                Instant video overview
            </div>
        </div>
        <div class="glass-card" style="width:200px;text-align:center;padding:1.4rem;">
            <div style="font-size:1.8rem;margin-bottom:0.5rem;">🔍</div>
            <div style="font-family:Syne,sans-serif;font-weight:700;font-size:0.85rem;
                        color:#e8e8f0;">Semantic Search</div>
            <div style="font-size:0.78rem;color:#6b6b80;margin-top:0.3rem;">
                FAISS vector retrieval
            </div>
        </div>
        <div class="glass-card" style="width:200px;text-align:center;padding:1.4rem;">
            <div style="font-size:1.8rem;margin-bottom:0.5rem;">🌍</div>
            <div style="font-family:Syne,sans-serif;font-weight:700;font-size:0.85rem;
                        color:#e8e8f0;">Multilingual</div>
            <div style="font-size:0.78rem;color:#6b6b80;margin-top:0.3rem;">
                Urdu, Sindhi & more
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)