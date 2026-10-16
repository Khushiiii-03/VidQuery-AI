# VidQuery-AI
VidQuery AI is an AI application that assists in understanding YouTube videos by means of natural language conversations. The application achieves this by combining Retrieval-Augmented Generation (RAG) with the processing of transcripts and the use of large language models (LLMs), enabling to ask questions about the content of the videos, create summaries and investigate the information without having to watch the video.

## Features

- **Video Question Answering:** With VidQuery AI you can ask questions and get answers which are aware of the context, using the video transcript.
- **Conversational Memory:** With VidQuery AI you can pose further questions and still retain the context of the conversation.
- **Multilingual Support:** VidQuery AI is compatible with the languages provided, including English, Hindi, Chinese, Spanish, French and Japanese.
- **Timestamp-Aware Responses:** VidQuery AI uses the timestamps from the transcript to enable you to find sections of the video.
- **Source Transparency:** With VidQuery AI you can examine the transcript excerpts which were retrieved in order to produce the answers.
- **Interactive Interface:** The assistant interacts with VidQuery AI through an interface that is based on Streamlit.

## Tech Stack

- **Language:** Python
- **Frontend:** Streamlit
- **LLM Provider:** Groq API
- **LLM Integration:** LangChain
- **Embeddings:** Hugging Face Sentence Transformers
- **Embedding Model:** sentence-transformers/all-MiniLM-v2
- **Vector Database:** FAISS
- **Core Concepts:** Retrieval-Augmented Generation (RAG) semantic search, embeddings, transcript processing, conversational AI

## How It Works

1. **Video Input:** The user enters a YouTube video URL.
2. **Transcript Extraction:** VidQuery AI obtains the transcript that is available and retains the timestamps where this is possible.
3. **Text Processing:** VidQuery AI breaks up the transcript into sections in order that retrieval may be carried out efficiently.
4. **Generation of embeddings:** VidQuery AI converts the text chunks into vector embeddings by using a sentence transformer model.
5. **Vector Search:** FAISS locates the sections of the transcript that are relevant to the user's question.
6. **Contextual Response Generation:** VidQuery AI forwards the retrieved content together with the conversation context to the LLM through Groq.
7. **Answer and Sources:** The VidQuery AI system shows an answer that has been generated together with extracts from the relevant source.

## Use Cases

- Summarizing videos and lectures
- Finding specific information in long videos
- Asking follow‑up questions about tutorials
- Exploring video content in supported languages
- Accelerating research and learning through conversational search

## Future Scope

- Put the application on a public hosting service.
- Include automated tests and enhance the error handling.
- Enhance the quality of retrieval by using chunking and ranking strategies.
- Include support for playlists and multiple-video knowledge bases.
- Give the lengths and formats of the summaries.
- Enhance the accuracy of citations and the assessment of answers.
