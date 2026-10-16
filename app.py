import os
import re
from urllib.parse import urlparse

import requests
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# CONFIGURATION

load_dotenv()

st.set_page_config(
    page_title="VidQuery AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.2);
    }

    .app-subtitle {
        color: #888888;
        font-size: 1rem;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
    }

    .status-card {
        padding: 1rem;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        margin-bottom: 1rem;
    }

    .hero-panel {
        padding: 1.5rem 1.6rem;
        border: 1px solid rgba(128,128,128,0.22);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(99,102,241,0.10), rgba(14,165,233,0.06));
        margin-bottom: 1.25rem;
    }

    .hero-eyebrow {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #818cf8;
        margin-bottom: 0.45rem;
    }

    .hero-title {
        font-size: clamp(1.7rem, 3vw, 2.35rem);
        line-height: 1.15;
        font-weight: 750;
        margin-bottom: 0.55rem;
    }

    .hero-copy {
        color: #888888;
        font-size: 1rem;
        margin-bottom: 0;
    }

    div.stButton > button {
        border-radius: 10px;
        transition: transform 120ms ease, border-color 120ms ease;
    }

    div.stButton > button:hover {
        transform: translateY(-1px);
    }

    [data-testid="stChatMessage"] {
        border-radius: 14px;
    }

    .small-muted {
        color: #888888;
        font-size: 0.86rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# SESSION STATE

DEFAULTS = {
    "vector_store": None,
    "video_id": None,
    "video_title": None,
    "transcript": None,
    "transcript_language": None,
    "summary": None,
    "messages": [],
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_video():
    st.session_state.vector_store = None
    st.session_state.video_id = None
    st.session_state.video_title = None
    st.session_state.transcript = None
    st.session_state.transcript_language = None
    st.session_state.summary = None
    st.session_state.messages = []

# MODEL CONFIGURATION

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    model_name = os.getenv("GROQ_MODEL")

    if not model_name:
        raise ValueError(
            "GROQ_MODEL is missing. Set your working model ID in .env."
        )

    return ChatGroq(
        model=model_name,
        temperature=0.2,
        api_key=api_key,
    )

# YOUTUBE URL AND VIDEO INFORMATION

def extract_video_id(url):
    url = url.strip()

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host in ("youtu.be", "www.youtu.be"):
        video_id = parsed.path.strip("/").split("/")[0]

    elif host in (
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
    ):
        if parsed.path == "/watch":
            from urllib.parse import parse_qs
            video_id = parse_qs(parsed.query).get("v", [""])[0]

        elif parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
            parts = parsed.path.strip("/").split("/")
            video_id = parts[1] if len(parts) > 1 else ""

        else:
            video_id = ""

    else:
        video_id = ""

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError(
            "Enter a valid YouTube video URL or 11-character video ID."
        )

    return video_id


def get_video_title(video_id):
    """Fetch the public YouTube title when available."""
    try:
        response = requests.get(
            "https://www.youtube.com/oembed",
            params={
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "format": "json",
            },
            timeout=8,
        )
        response.raise_for_status()
        return response.json().get("title", "YouTube video")

    except requests.RequestException:
        return "YouTube video"

# TRANSCRIPT EXTRACTION

def format_timestamp(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"



def get_transcript(video_id, preferred_languages=None):
    if preferred_languages is None:
        preferred_languages = ["en", "hi"]

    try:
        api = YouTubeTranscriptApi()

        try:
            transcript = api.fetch(
                video_id,
                languages=preferred_languages,
            )
            transcript_language = getattr(
                transcript, "language_code", "unknown"
            )

        except Exception:
            transcript_list = api.list(video_id)
            available = list(transcript_list)

            if not available:
                raise ValueError(
                    "No transcripts are available for this video."
                )

            selected = next(
                (
                    item for item in available
                    if not getattr(item, "is_generated", False)
                ),
                available[0],
            )

            transcript = selected.fetch()
            transcript_language = getattr(
                selected, "language_code", "unknown"
            )

        segments = []

        for item in transcript:
            text = getattr(item, "text", "")
            start = getattr(item, "start", 0)

            if isinstance(item, dict):
                text = item.get("text", "")
                start = item.get("start", 0)

            if text and text.strip():
                segments.append(
                    f"[{format_timestamp(start)}] {text.strip()}"
                )

        if not segments:
            raise ValueError("The transcript is empty.")

        return "\n".join(segments), transcript_language

    except Exception as error:
        raise RuntimeError(
            "Could not retrieve a transcript. The video may have "
            "captions disabled, or YouTube may be temporarily "
            "unavailable. Try another video or language."
        ) from error

# VECTOR STORE

def build_vector_store(transcript, video_id):
    document = Document(
        page_content=transcript,
        metadata={"video_id": video_id},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents([document])

    if not chunks:
        raise ValueError(
            "Could not create transcript chunks for this video."
        )

    embeddings = load_embeddings()

    return FAISS.from_documents(chunks, embeddings)

# RAG CHAIN AND SOURCES

def answer_question(question, vector_store, history=None):
    history = history or []
    recent_history = history[-8:]
    history_text = "\n".join(
        f"{item['role'].capitalize()}: {item['content']}"
        for item in recent_history
        if item.get("role") in ("user", "assistant")
    )

    llm = get_llm()
    search_question = question

    if history_text.strip():
        contextualize_prompt = ChatPromptTemplate.from_template("""
        Rewrite the latest user question as a standalone search query for a
        transcript retrieval system. Use the conversation only to resolve
        references and follow-ups. Preserve names, technical terms, and the
        language of the latest question. Return only the rewritten query.

        Conversation:
        {history}

        Latest question: {question}
        Standalone search query:
        """)
        search_question = (
            contextualize_prompt | llm | StrOutputParser()
        ).invoke({"history": history_text, "question": question}).strip()
        if not search_question:
            search_question = question

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    documents = retriever.invoke(search_question)

    if not documents:
        return (
            "I couldn't find relevant information in the transcript.",
            [],
        )

    context = "\n\n".join(doc.page_content for doc in documents)

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful multilingual assistant answering questions about a
    YouTube video's transcript.

    Rules:
    - Answer using only the supplied transcript context and relevant prior conversation.
    - Do not invent facts. If the context is insufficient, say so clearly.
    - Respond in the same language as the user's latest question.
    - Use conversation history to understand follow-up questions, but do not treat
      previous assistant claims as evidence if they are not supported by the transcript.
    - Preserve names, technical terms, and numbers accurately.
    - Do not print raw source excerpts or the retrieval context in your answer.

    Recent conversation:
    {history}

    Transcript context:
    <context>
    {context}
    </context>

    Latest question: {question}
    """)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "question": question,
        "history": history_text or "No previous conversation.",
    })

    sources = []
    for doc in documents:
        content = doc.page_content
        matches = re.findall(r"\[(\d{2,}:\d{2}(?::\d{2})?)\]", content)
        timestamp = matches[0] if matches else None
        seconds = None

        if timestamp:
            parts = [int(part) for part in timestamp.split(":")]
            if len(parts) == 3:
                seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
            else:
                seconds = parts[0] * 60 + parts[1]

        sources.append({"text": content, "timestamp": timestamp, "seconds": seconds})

    return answer, sources


def summarize_transcript(transcript):
    """Create a structured summary without sending the entire transcript at once."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=6500,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(transcript)
    if not chunks:
        raise ValueError("There is no transcript text to summarize.")

    llm = get_llm()
    map_prompt = ChatPromptTemplate.from_template("""
    Summarize this section of a YouTube transcript in concise bullet points.
    Capture the main ideas, important facts, examples, names, and numbers.
    Preserve the language of the transcript. Do not add information that is not present.

    Transcript section:
    {transcript_chunk}
    """)
    map_chain = map_prompt | llm | StrOutputParser()
    partial_summaries = []
    progress = st.progress(0, text="Reading transcript sections...")

    try:
        for index, chunk in enumerate(chunks, start=1):
            partial_summaries.append(
                map_chain.invoke({"transcript_chunk": chunk})
            )
            progress.progress(
                index / len(chunks),
                text=f"Summarizing section {index} of {len(chunks)}...",
            )
    finally:
        progress.empty()

    reduce_prompt = ChatPromptTemplate.from_template("""
    Combine the following section summaries into one shorter summary.
    Keep the most important ideas, facts, examples, and numbers. Remove repetition.
    Do not introduce new information. Preserve the language used in the summaries.

    Section summaries:
    {summaries}
    """)
    reduce_chain = reduce_prompt | llm | StrOutputParser()
    while len(partial_summaries) > 8:
        grouped = []
        for i in range(0, len(partial_summaries), 8):
            grouped.append(reduce_chain.invoke({
                "summaries": "\n\n".join(partial_summaries[i:i + 8])
            }))
        partial_summaries = grouped

    final_prompt = ChatPromptTemplate.from_template("""
    Create a useful, well-organized video summary from these notes.
    Preserve the language of the notes. Use these headings:
    ## Overview
    ## Key Takeaways
    ## Important Details
    ## Conclusion
    Keep it readable and concise. Use bullets where helpful. Do not invent details.

    Notes:
    {summaries}
    """)
    return (final_prompt | llm | StrOutputParser()).invoke({
        "summaries": "\n\n".join(partial_summaries)
    })

# FRIENDLY ERROR MESSAGES

def display_error(error, stage="answer"):
    message = str(error)
    lowered = message.lower()

    if isinstance(error, ValueError):
        st.warning(message)

    elif "429" in lowered or "rate_limit" in lowered:
        st.error(
            "The AI provider's rate limit has been reached. "
            "Wait a little and try again."
        )

    elif (
        "401" in lowered
        or "authentication" in lowered
        or "invalid api key" in lowered
    ):
        st.error(
            "AI authentication failed. Check GROQ_API_KEY "
            "in your .env file."
        )

    elif "model_not_found" in lowered or "404" in lowered:
        st.error(
            "The configured AI model was not found or is not "
            "available to your account. Check GROQ_MODEL in .env."
        )

    elif "timeout" in lowered or "connection" in lowered:
        st.error(
            "A network connection problem occurred. "
            "Check your internet connection and retry."
        )

    elif stage == "transcript":
        st.error(message)

    else:
        st.error(
            "Something went wrong while generating the answer. "
            "Please retry. If the issue continues, check the "
            "terminal for the technical error."
        )

# SIDEBAR

with st.sidebar:
    st.title("🎬 VidQuery AI")
    st.caption("Ask questions about YouTube videos.")

    st.divider()

    video_url = st.text_input(
        "YouTube video URL",
        placeholder="Paste a YouTube link or video ID here",
        key="video_url_input",
    )
    
    language_options = {
        "English": "en",
        "Hindi": "hi",
        "Chinese": "zh-Hans",
        "Spanish": "es",
        "French": "fr",
        "Japanese": "ja",
        "Any available language": "auto",
    }

    selected_language = st.selectbox(
        "Preferred transcript language",
        options=list(language_options.keys()),
        index=0,
    )

    if st.button(
        "Process video",
        type="primary",
        use_container_width=True,
    ):
        try:
            video_id = extract_video_id(video_url)

            with st.spinner(
                "Fetching transcript and building search index"
            ):
                preferred_languages = (
                    ["en", "hi"]
                    if language_options[selected_language] == "auto"
                    else [language_options[selected_language], "en", "hi"]
                )
                transcript, transcript_language = get_transcript(
                    video_id,
                    preferred_languages=preferred_languages,
                )
                vector_store = build_vector_store(
                    transcript,
                    video_id,
                )
                video_title = get_video_title(video_id)

            st.session_state.vector_store = vector_store
            st.session_state.video_id = video_id
            st.session_state.video_title = video_title
            st.session_state.transcript = transcript
            st.session_state.transcript_language = transcript_language
            st.session_state.summary = None
            st.session_state.messages = []

            st.success("Video processed successfully!")
            st.rerun()

        except Exception as error:
            display_error(error, stage="transcript")

    if st.session_state.video_id:
        st.divider()
        st.markdown("**Current video**")
        st.write(st.session_state.video_title)

        st.caption(
            f"Video ID: `{st.session_state.video_id}`"
        )
        if st.session_state.transcript_language:
            st.caption(f"Transcript language: `{st.session_state.transcript_language}`")

        if st.button("Generate video summary", use_container_width=True):
            if st.session_state.transcript:
                try:
                    with st.spinner("Summarizing the video transcript"):
                        st.session_state.summary = summarize_transcript(
                            st.session_state.transcript
                        )
                    st.success("Summary generated!")
                    st.rerun()
                except Exception as error:
                    display_error(error, stage="answer")

        st.link_button(
            "Open on YouTube ↗",
            f"https://www.youtube.com/watch?v="
            f"{st.session_state.video_id}",
            use_container_width=True,
        )

        if st.button(
            "Switch video",
            use_container_width=True,
        ):
            reset_video()
            st.rerun()

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()

# MAIN INTERFACE

st.markdown(
    """
    <div class="hero-panel">
        <div class="hero-title">Watch less. Understand more.</div>
        <p class="hero-copy">An AI that understands YouTube videos, answers questions, and remembers context.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.video_id:
    st.subheader(st.session_state.video_title or "Your video")

    with st.expander("Watch video", expanded=False):
        st.video(
            f"https://www.youtube.com/watch?v="
            f"{st.session_state.video_id}"
        )

    st.success("Video indexed and ready for questions.")

    if st.session_state.summary:
        st.markdown("### 📝 Video summary")
        with st.container(border=True):
            st.markdown(st.session_state.summary)
        st.download_button(
            "Download summary as Markdown",
            data=st.session_state.summary,
            file_name="youtube_video_summary.md",
            mime="text/markdown",
        )

else:
    st.info(
        "👈 Paste a **YouTube video URL** in the sidebar and select "
        "**Process video** to get started."
    )

    st.markdown("### What you can do")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 💬 Chat with context")
        st.caption("Ask follow-up questions without repeating yourself.")

    with col2:
        st.markdown("#### 📝 Summarize")
        st.caption("Generate a structured summary with key takeaways.")

    with col3:
        st.markdown("#### ⏱️ Jump to moments")
        st.caption("Use timestamp links to revisit relevant sections.")

# CHAT HISTORY

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander(
                f"📚 Sources ({len(message['sources'])} excerpts)",
                expanded=False,
            ):
                for index, source in enumerate(
                    message["sources"],
                    start=1,
                ):
                    timestamp = source.get("timestamp")
                    st.markdown(f"**Excerpt {index}**")

                    if timestamp and st.session_state.video_id:
                        video_link = (
                            f"https://www.youtube.com/watch?v="
                            f"{st.session_state.video_id}"
                            f"&t={source['seconds']}s"
                        )
                        st.markdown(
                            f"[▶ Jump to {timestamp}]({video_link})"
                        )

                    st.text(source["text"])

# CHAT INPUT

if st.session_state.vector_store is not None and not st.session_state.messages:
    st.markdown("#### Try asking")
    suggestion_cols = st.columns(3)
    suggestions = [
        "Summarize the main idea",
        "What are the key takeaways?",
        "What examples are mentioned?",
    ]
    for col, suggestion in zip(suggestion_cols, suggestions):
        with col:
            if st.button(suggestion, use_container_width=True):
                st.session_state.pending_question = suggestion

pending_question = st.session_state.pop("pending_question", None)
typed_question = st.chat_input(
    "Ask a question about the video",
    disabled=st.session_state.vector_store is None,
)
question = pending_question or typed_question

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Searching transcript and generating your answer"
            ):
                answer, sources = answer_question(
                    question,
                    st.session_state.vector_store,
                    history=st.session_state.messages[:-1],
                )

            st.markdown(answer)

            if sources:
                with st.expander("View sources", expanded=False):
                    for index, source in enumerate(sources, start=1):
                        timestamp = source.get("timestamp")

                        st.markdown(f"**Source {index}**")

                        if timestamp and st.session_state.video_id:
                            st.markdown(
                                f"[▶ Jump to {timestamp}]"
                                f"(https://www.youtube.com/watch?v="
                                f"{st.session_state.video_id}"
                                f"&t={source['seconds']}s)"
                            )

                        st.caption(source["text"][:500])


            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except Exception as error:
            display_error(error, stage="answer")
