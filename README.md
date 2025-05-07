# Askora - AI Video Summarization and Learning Platform

---

## 🚀 Overview

Askora is a full-stack AI-powered web application that transforms YouTube playlists into interactive and personalized learning experiences. Built with a **React + Vite frontend** and a **FastAPI backend**, Askora leverages state-of-the-art AI models for video summarization, contextual Q\&A, voice interaction, and intelligent quiz generation.

It is designed for **students, educators, and lifelong learners** who want to engage actively with content — rather than passively watching.

---

## 🧠 Core Features

### 🎥 1. AI Video Summarization

* Users are presented with curated YouTube playlists.
* Audio from videos is extracted and transcribed using **Whisper**.
* Transcripts are cleaned and summarized via **GPT** into structured articles.

### 💬 2. Interactive Q\&A

* Ask any question about the playlist content.
* System uses **FAISS** + **OpenAI embeddings** for semantic retrieval.
* A LangChain-based **ReAct agent** routes queries through a cybersecurity QA tool.
* Irrelevant questions are smartly rejected.

### 📚 3. Playlist Management

* Browse pre-summarized playlists.
* Instantly access articles without reprocessing delays.

### 🧪 4. Smart Knowledge Quizzes

* Quizzes are generated using **GPT**.
* Instant feedback with correction.
* Enhances memory and understanding.

### 🗣️ 5. AI Voice Interaction

* Ask questions using your voice.
* Get answers read aloud via **ElevenLabs** TTS.

---

## ⚙️ Technical Stack

### Frontend

* **React + Vite**
* Tailwind CSS for styling
* Framer Motion for animations
* Modular architecture (Hero, About, Quiz, Chat, etc.)

### Backend

* **FastAPI** REST API
* Exposes endpoints like `/ask`, `/generate-question`
* Handles vectorstore (FAISS), similarity scoring, and RAG logic

### Vectorstore + RAG

* Transcripts are split and embedded
* **FAISS** handles fast semantic retrieval
* Top chunks + query are passed to **GPT**

### Agent Architecture

* Built using **LangChain’s ReAct agent**
* Uses custom `CyberSecurityQA_tool` to ensure relevance
* Filters unrelated queries and logs evaluations

---

## 👩‍💻 User Journey

### 🎯 Landing Page

* Users begin by selecting a playlist.

### 🧾 Dashboard

* View summaries and article content
* Track learning progress

### 🤖 Q\&A Chatbot

* Ask questions via text or voice
* Agent retrieves and answers using context

### ❓ Quiz Section

* AI-generated questions from playlist content
* Evaluated in real-time

### 📂 Knowledge Base

* Saved articles and interactions
* Encourages long-term personalized learning

---

## 🎓 Use Cases

* **Students:** Summarize lectures, ask questions, test knowledge
* **Educators:** Build mini-courses from video content
* **Lifelong Learners:** Engage with complex content more efficiently

---

## 📊 Evaluation

We used the [RAG Evaluator](https://github.com/AIAnytime/rag-evaluator), a Python library for evaluating RAG-based systems.

✅ It was selected because it compares generated responses directly against reference content — ideal for retrieval-based systems like Askora.

📐 The tool evaluates across multiple global metrics. We focused on:

* **BERT F1:** Measures semantic similarity between response and reference.
* **Perplexity:** Measures how fluent and predictable the response is.
* **Diversity:** Measures how repetitive or unique the response is.
* **Racial Bias:** Ensures generated responses avoid biased phrasing.

📎 Askora’s responses were tested using in-scope and out-of-scope questions to validate both accuracy and responsible fallback behavior.

🖼️ Example Output:

* Table of metric results
* Summary chart visual
* In-scope and out-of-scope response samples

---

## 📦 Project Structure

```
Askora-AI-Learning-Assistant/
│
├── core_models/              # Summarization + RAG code notebooks
│   ├── cybersec_rag_agent.ipynb
│   └── totur_course_summarization.ipynb
│
├── results_and_docs/        # Evaluation results, report, presentation, and media
│   ├── rag_eval_results.csv
│   ├── Askora - Report.pdf
│   ├── Askora - Presentation.pdf
│
├── deployment/              # FastAPI backend app (deployment-ready)
│   ├── main.py
│   └── demo_video.mp4
│
├── failed_experiments/      # Archive of early test notebooks
│   └── rag_test_old.ipynb
│
└── README.md

```

---

## 🌐 Try Askora

🧪 Explore the deployed platform here:
[🔗 Try the Askora AI Learning Assistant](https://askora-ashy.vercel.app/)

---

## 📌 Summary

Askora transforms how learners engage with video content by turning passive playlists into structured, searchable, and interactive learning hubs. It integrates summarization, semantic search, AI agents, and evaluation — all in one powerful tool.

Built for the future of education: **modular, intelligent, and human-centered.**
