# AI-Powered Product Review Intelligence Platform

## Overview

This project is an end-to-end NLP system that transforms large-scale product reviews into structured, explainable insights and an interactive AI copilot.

The system combines:

* Transformer-based sentiment classification
* Topic modeling for interpretability
* Product-level aggregation of insights
* An evidence-grounded LLM assistant

The goal is to simulate a real-world scenario where businesses need to extract actionable signals from unstructured customer feedback at scale.

---

## Key Features

* **Fine-tuned DistilBERT sentiment classifier**
* **Product-level aggregation of insights** (pros %, cons %, top reviews)
* **Topic modeling (BERTopic)** to extract themes from reviews
* **AI Copilot (LLM)** that answers questions using grounded evidence
* **Optimized inference pipeline** using batching + mixed precision
* **Interactive Streamlit application**

---

## System Architecture

```
Raw Reviews
    ↓
Data Cleaning & Preprocessing
    ↓
DistilBERT Sentiment Classification
    ↓
Batch Inference (GPU Optimized)
    ↓
Product-Level Aggregation
    ↓
Topic Modeling (BERTopic)
    ↓
Evidence Review Extraction
    ↓
LLM Copilot (Grounded Q&A)
    ↓
Streamlit UI
```

---

## Machine Learning Details

### Sentiment Classification

* Model: `DistilBERT (distilbert-base-uncased)`
* Task: Binary sentiment classification
* Loss Function: **Class-weighted CrossEntropyLoss**
* Motivation: Handle strong class imbalance in dataset

### Evaluation

* Accuracy: ~97%
* Macro F1 Score: ~0.89
* Strong minority-class performance (negative sentiment)

> Accuracy alone was insufficient due to dataset imbalance. Macro F1 and recall were prioritized.

---

### Topic Modeling

* Method: **BERTopic**
* Output:

  * Positive topics per product
  * Negative topics per product
* Purpose:

  * Provide interpretability
  * Explain *why* sentiment exists

---

## AI Copilot (LLM Layer)

The system includes a conversational assistant that answers product-related questions.

### Key Design Decision:

The LLM is **strictly grounded** in:

* Sentiment outputs
* Topic modeling results
* Evidence reviews

This prevents hallucination and ensures responses are:

* Explainable
* Traceable
* Reliable

---

## ⚡ Performance Optimizations

* Batched inference using `DataLoader`
* Mixed precision (`torch.cuda.amp`)
* Cached data loading with `st.cache_data`
* Precomputed ML artifacts stored in CSV format

---

## Data Sources

* Amazon Product Reviews Dataset
* Preprocessed and cleaned:

  * Removed duplicates
  * Handled missing values
  * Normalized product identifiers

---

## Application

The Streamlit app allows users to:

* Browse products
* View sentiment breakdowns
* Explore topics
* Read evidence-based reviews
* Ask questions via AI Copilot

---

## Running the App Locally

```bash
# Create environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

---

## Deployment

This project is designed to be deployed as a production-style service using:

* Render (recommended for always-on hosting)
* Streamlit Cloud (for lightweight demos)

---

## Future Improvements

* Multi-class sentiment classification (positive / neutral / negative)
* Active learning loop for model improvement
* Real-time streaming of new reviews
* Model quantization for faster inference
* Feedback loop from user interactions

---

## Key Takeaways

* Building ML systems requires more than model training — aggregation and interpretability are critical
* Combining structured ML outputs with LLMs enables powerful, reliable AI systems
* Guardrails are essential when integrating LLMs into production workflows

---

## Author

**Monish Yarapathineni**
AI Engineer | Applied AI | NLP Systems

