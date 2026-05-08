# AI Bug Intelligence Dashboard

An AI-powered dashboard that analyzes bug reports using LLMs and embeddings to classify issues, detect duplicates, and generate executive-level insights and prioritization recommendations.

---

## 🚀 Overview

Bug triage is often manual, noisy, and difficult to prioritize at scale.

This project demonstrates how modern AI techniques can transform raw issue data into actionable insights by combining:

* LLM-based classification and summarization
* Embeddings-based semantic similarity for duplicate detection
* Interactive visualization and reporting

This application uses OpenAI APIs for inference, combining LLM-based classification with embeddings-based similarity analysis

I intentionally used a workflow-oriented architecture instead of agents because the problem domain was deterministic and sequential. A workflow provided better reliability, predictability, and debuggability.

---

## 📸 Screenshots

*(Add 1–2 screenshots of your dashboard here)*

---

## 🧠 Key Features

### 1. AI-Powered Classification

Automatically categorizes bugs into themes (e.g., Auth, Payments, UI, Performance) using an LLM.

### 2. Duplicate Detection (Embeddings)

Uses semantic similarity (embeddings + cosine similarity) to identify related or duplicate issues beyond simple keyword matching.

### 3. Trend Analysis

Aggregates and visualizes:

* Bug count by category
* High-impact problem areas
* Severity distribution

### 4. Executive Summary & Recommendations

Generates a concise summary of:

* Key problem areas
* Risks and customer impact
* Prioritized next actions

### 5. Exportable Report

Download analysis results for sharing with stakeholders.

---

## 🏗️ Architecture

```
CSV file
   ↓
Pandas dataframe
   ↓
LLM classification
   ↓
Categories added to dataset
   ↓
Embeddings generation
   ↓
Similarity analysis
   ↓
Executive summary + recommendations
   ↓
Streamlit dashboard
```

## System Flow

1. User uploads CSV bug data
2. Pandas processes the dataset
3. LLM classifies bugs into categories
4. Python aggregates counts and trends
5. Embeddings compute semantic similarity
6. LLM generates executive recommendations
7. Streamlit displays insights and reports

---

## ⚙️ Tech Stack

* Python
* Streamlit
* OpenAI (LLM + embeddings)
* Pandas
* Scikit-learn (cosine similarity)

---

## 📂 Project Structure

```
ai-bug-intelligence-dashboard/
├── app.py
├── bugs.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ Running Locally

1. Clone the repo:

   git clone https://github.com/YOUR_USERNAME/ai-bug-intelligence-dashboard.git
   cd ai-bug-intelligence-dashboard

2. Create a virtual environment:

   python3 -m venv .venv
   source .venv/bin/activate

3. Install dependencies:

   pip install -r requirements.txt

4. Set your API key:

   export OPENAI_API_KEY="your_api_key_here"

5. Run the app:

   streamlit run app.py
   .venv/bin/streamlit run app.py

---

## 🧠 Key Design Decisions

* LLMs for classification and summarization
  → flexible and expressive for unstructured data

* Embeddings for similarity (not LLMs)
  → faster, cheaper, and deterministic

* Low-temperature configuration for classification
  → ensures consistent outputs

* Workflow-based design (not agents)
  → predictable, reliable, and production-friendly

---

## 📌 Future Improvements

* Add caching to reduce API calls and latency
* Introduce evaluation metrics for classification quality
* Support real-time data ingestion (e.g., Jira API)
* Add confidence scoring for AI outputs

---

## 💡 Why this matters

This project illustrates how combining deterministic retrieval (embeddings) with generative AI (LLMs) enables more reliable and practical AI systems — a pattern widely used in production environments.

---

## 👤 Author

Gustavo Varejao
Senior Technical Program Manager

