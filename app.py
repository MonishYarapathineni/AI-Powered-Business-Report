# app.py
import os
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="AI Product Review Catalog", layout="wide")

# ----------------------------
# Load env (dotenv) + OpenAI key
# ----------------------------
load_dotenv()  # loads .env in current directory

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import re
import html
def safe_render_review(title: str, text: str):
    """
    Render review text safely.
    Uses st.write normally, falls back to text_area if rendering glitches.
    """
    st.markdown(title)

    if not text or not isinstance(text, str):
        st.write("No review available.")
        return

    # Heuristic: very long lines or suspicious unicode → text_area
    if len(text) > 1200 or any(c in text for c in ["\u2028", "\u2029", "\u200b"]):
        st.text_area(
            label="",
            value=text,
            height=260,
            disabled=True
        )
    else:
        st.write(text)
def clean_review_for_display(x: str) -> str:
    if x is None:
        return ""
    x = str(x)

    # common mojibake fixes (CSV/encoding artifacts)
    x = (x.replace("â€™", "'")
           .replace("â€œ", '"')
           .replace("â€�", '"')
           .replace("â€“", "-")
           .replace("â€”", "-")
           .replace("Â", ""))   # leftover non-breaking-space marker

    # normalize whitespace + remove invisible/control chars
    x = x.replace("\xa0", " ")              # non-breaking spaces
    x = x.replace("\u200b", "")             # zero-width space
    x = x.replace("\u2028", " ").replace("\u2029", " ")
    x = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", x)  # control chars
    x = re.sub(r"\s+", " ", x).strip()      # collapse whitespace

    return x

# ----------------------------
# Styles
# ----------------------------
st.markdown(
    """
<style>
/* premium CTA card */
.cta-card {
  border: 1px solid rgba(255,255,255,0.10);
  background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(168,85,247,0.12));
  border-radius: 16px;
  padding: 18px 18px;
  margin: 6px 0 12px 0;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}
.cta-title {
  font-size: 18px;
  font-weight: 800;
  margin: 0;
}
.cta-sub {
  margin: 6px 0 0 0;
  opacity: 0.90;
  font-size: 13px;
  line-height: 1.35;
}
.cta-badges span{
  display:inline-block;
  font-size: 12px;
  padding: 4px 10px;
  border-radius: 999px;
  margin-right: 6px;
  margin-top: 10px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
}

/* make sidebar inputs feel tighter */
section[data-testid="stSidebar"] .stTextInput, 
section[data-testid="stSidebar"] .stSelectbox {
  margin-bottom: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Title
# ----------------------------
st.title("🛍️ AI-Powered Product Review Catalog")
st.caption("LLM-generated Pros/Cons + Topic Modeling + Evidence Reviews + AI Copilot Q&A")

# ----------------------------
# Data loading
# ----------------------------
@st.cache_data
def load_data():
    topics_df = pd.read_csv("data/amazon_topic_modeling.csv")
    summary_df = pd.read_csv("data/productReviewSummary.csv")
    top_df = pd.read_csv("data/top_reviews_per_product.csv")
    return topics_df, summary_df, top_df

topics_df, summary_df, top_df = load_data()

# ----------------------------
# Clean / normalize inputs
# ----------------------------

# If productReviewSummary.csv has an unnamed index column, drop it
for col in list(summary_df.columns):
    if str(col).lower().startswith("unnamed"):
        summary_df = summary_df.drop(columns=[col])

# Normalize product-name columns + strip whitespace
summary_df["productName"] = summary_df["productName"].astype(str).str.strip()
top_df["name"] = top_df["name"].astype(str).str.strip()
topics_df["product"] = topics_df["product"].astype(str).str.strip()

# Remove rows where product name is nan-like (string "nan" or empty)
summary_df = summary_df[summary_df["productName"].str.lower().ne("nan")]
summary_df = summary_df[summary_df["productName"].str.len() > 0]

top_df = top_df[top_df["name"].str.lower().ne("nan")]
top_df = top_df[top_df["name"].str.len() > 0]

topics_df = topics_df[topics_df["product"].str.lower().ne("nan")]
topics_df = topics_df[topics_df["product"].str.len() > 0]

# Join keys
summary_df["_key"] = summary_df["productName"]
top_df["_key"] = top_df["name"]
topics_df["_key"] = topics_df["product"]

# Topic text preference: generatedTopic else Representation
topics_df["topic_text"] = topics_df["generatedTopic"].fillna(topics_df["Representation"]).astype(str)

# Aggregate topics per product (unique, preserve order)
topics_agg = (
    topics_df.groupby(["_key", "isPositiveSentiment"])["topic_text"]
    .apply(lambda x: list(dict.fromkeys([t for t in x.tolist() if str(t).strip()])))
    .reset_index()
)

pos_topics = (
    topics_agg[topics_agg["isPositiveSentiment"] == True][["_key", "topic_text"]]
    .rename(columns={"topic_text": "pos_topics"})
)
neg_topics = (
    topics_agg[topics_agg["isPositiveSentiment"] == False][["_key", "topic_text"]]
    .rename(columns={"topic_text": "neg_topics"})
)

topics_final = pos_topics.merge(neg_topics, on="_key", how="outer")

# Merge into one table
merged = (
    summary_df.merge(top_df, on="_key", how="left")
    .merge(topics_final, on="_key", how="left")
)

# Ensure list columns exist as lists
merged["pos_topics"] = merged["pos_topics"].apply(lambda x: x if isinstance(x, list) else [])
merged["neg_topics"] = merged["neg_topics"].apply(lambda x: x if isinstance(x, list) else [])

# ----------------------------
# Sidebar: Search + Select
# ----------------------------
st.sidebar.header("🔎 Browse Products")
search = st.sidebar.text_input("Search product", "")

filtered = merged.copy()
if search.strip():
    filtered = filtered[
        filtered["productName"].str.lower().str.contains(search.lower(), na=False)
    ]

if filtered.empty:
    st.warning("No products match your search.")
    st.stop()

product_list = filtered["productName"].unique().tolist()
selected_product = st.sidebar.selectbox("Select a product", product_list)

# Grab row (single product)
row = filtered[filtered["productName"] == selected_product].iloc[0]

# ----------------------------
# Chat state (reset when product changes)
# ----------------------------
# --- Chat session state ---

if "pending_q" not in st.session_state:
    st.session_state.pending_q = None


if "active_product" not in st.session_state:
    st.session_state.active_product = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Reset chat when product changes
if st.session_state.active_product != selected_product:
    st.session_state.active_product = selected_product
    st.session_state.chat_history = []
    st.session_state.pending_q = None   

# ----------------------------
# Context builder (for OpenAI)
# ----------------------------
def build_product_context(r):
    pros_pct = r.get("pros_percentage", np.nan)
    cons_pct = r.get("cons_percentage", np.nan)

    pros = r.get("pros", "")
    cons = r.get("cons", "")

    pos_topics_list = r.get("pos_topics", []) or []
    neg_topics_list = r.get("neg_topics", []) or []

    text_pos = r.get("reviews.text_pos", "")
    text_neg = r.get("reviews.text_neg", "")

    # Build a compact, grounded context block
    context = f"""
PRODUCT
- Name: {selected_product}

SENTIMENT SUMMARY (precomputed)
- Pros %: {pros_pct if pd.notna(pros_pct) else "N/A"}
- Cons %: {cons_pct if pd.notna(cons_pct) else "N/A"}

PROS (LLM summary)
{pros if isinstance(pros, str) and pros.strip() else "N/A"}

CONS (LLM summary)
{cons if isinstance(cons, str) and cons.strip() else "N/A"}

TOPICS (from topic modeling)
- Positive topics: {", ".join(pos_topics_list[:12]) if pos_topics_list else "N/A"}
- Negative topics: {", ".join(neg_topics_list[:12]) if neg_topics_list else "N/A"}

EVIDENCE REVIEWS (most helpful)
- Top positive review: {text_pos if isinstance(text_pos, str) and text_pos.strip() else "N/A"}
- Top negative review: {text_neg if isinstance(text_neg, str) and text_neg.strip() else "N/A"}
""".strip()
    return context

# ----------------------------
# OpenAI call
# ----------------------------
def ask_openai(question: str, context_text: str) -> str:
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY not found. Add it to your .env as OPENAI_API_KEY=... and restart Streamlit."

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        system = (
            "You are an evidence-grounded product review assistant.\n"
            "You MUST answer using only the provided context.\n"
            "If the context does not contain enough information, say what is missing and give a cautious answer.\n"
            "Do not invent specs, features, or claims not supported by the evidence reviews/topics/pros-cons.\n"
            "Keep answers concise (3–7 sentences) unless asked for more detail.\n"
        )

        user = f"""CONTEXT:
{context_text}

QUESTION:
{question}
"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.4,
            max_tokens=250,
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        return f"OpenAI error: {e}"

# ----------------------------
# Modal chat (Option B)
# ----------------------------
def chat_modal(context_text: str):
    # Streamlit >= 1.32 has st.dialog; if not, fall back to expander-ish container
    dialog_fn = getattr(st, "dialog", None)

    def render_chat_body():
        st.caption(
            "🔒 Evidence-grounded answers • 🧾 Uses topics + top reviews + pros/cons • ♻️ Resets when product changes"
        )

        # -------------------------
        # Preset question buttons
        # -------------------------
        q1, q2, q3 = st.columns(3)

        if q1.button("What do people like most?", use_container_width=True):
            st.session_state.pending_q = "What do people like most about this product?"

        if q2.button("Biggest complaints?", use_container_width=True):
            st.session_state.pending_q = "What are the biggest complaints about this product?"

        if q3.button("Should I buy it?", use_container_width=True):
            st.session_state.pending_q = (
                "Should I buy this product? Who is it best for and who should avoid it?"
            )

        st.divider()

        # -------------------------
        # Chat history
        # -------------------------
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # -------------------------
        # Resolve question (preset OR typed)
        # -------------------------
        typed_q = st.chat_input("Ask a question about this product…")

        q = None
        if st.session_state.pending_q:
            q = st.session_state.pending_q
            st.session_state.pending_q = None
        elif typed_q:
            q = typed_q

        # -------------------------
        # Answer
        # -------------------------
        if q:
            # user message
            st.session_state.chat_history.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)

            # assistant message
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    a = ask_openai(q, context_text)
                st.markdown(a)

            st.session_state.chat_history.append({"role": "assistant", "content": a})
    if dialog_fn:
        @st.dialog("🤖 AI Product Copilot (Evidence-Based)")
        def _dlg():
            render_chat_body()
        _dlg()
    else:
        st.info("Your Streamlit version doesn't support popups (st.dialog). Showing Copilot inline.")
        with st.container(border=True):
            render_chat_body()

# ----------------------------
# Main layout
# ----------------------------
left, right = st.columns([1, 2], gap="large")

with right:
    # --- Premium Copilot CTA (at top) ---
    st.markdown(
        """
<div class="cta-card">
  <div class="cta-title">✨ AI Product Copilot</div>
  <div class="cta-sub">
    Ask anything about this product — answers are grounded in <b>topic modeling</b>, <b>pros/cons summaries</b>,
    and <b>most helpful evidence reviews</b>.
  </div>
  <div class="cta-badges">
    <span>✅ Evidence-grounded</span>
    <span>⚡ Fast Q&A</span>
    <span>🧠 ML + LLM</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    context_text = build_product_context(row)

    # Launch modal
    if st.button("💬 Launch AI Copilot", use_container_width=True):
        chat_modal(context_text)

    # Trust caption
    st.caption("🔒 Evidence-grounded answers • 🧾 Uses review evidence • ♻️ Resets per product")
    st.divider()

with left:
    st.subheader("📌 Sentiment Breakdown")
    pros_pct = row.get("pros_percentage", np.nan)
    cons_pct = row.get("cons_percentage", np.nan)

    if pd.notna(pros_pct) and pd.notna(cons_pct):
        st.metric("Pros %", f"{pros_pct:.1f}%")
        st.metric("Cons %", f"{cons_pct:.1f}%")
    else:
        st.info("No percentage data available for this product.")

    st.subheader("✅ Pros")
    st.write(row.get("pros", "No pros summary available."))

    st.subheader("⚠️ Cons")
    st.write(row.get("cons", "No cons summary available."))

with right:
    st.subheader(f"🧾 {selected_product}")

    # Collapse sections so Copilot feels “primary”
    with st.expander("🧩 Topics Mentioned in Reviews", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**👍 Positive Topics**")
            if row["pos_topics"]:
                for t in row["pos_topics"][:10]:
                    st.markdown(f"- {t}")
            else:
                st.write("No positive topics found.")
        with c2:
            st.markdown("**👎 Negative Topics**")
            if row["neg_topics"]:
                for t in row["neg_topics"][:10]:
                    st.markdown(f"- {t}")
            else:
                st.write("No negative topics found.")

with st.expander("⭐ Evidence Reviews (Most Helpful)", expanded=True):
    colA, colB = st.columns(2)

    pos_text = clean_review_for_display(row.get("reviews.text_pos", ""))
    neg_text = clean_review_for_display(row.get("reviews.text_neg", ""))

    with colA:
        safe_render_review("**👍 Top Positive Review**", pos_text)

    with colB:
        safe_render_review("**👎 Top Negative Review**", neg_text)
st.divider()
st.caption("This app uses precomputed ML artifacts: product-level pros/cons summaries, topic modeling outputs, and top evidence reviews.")