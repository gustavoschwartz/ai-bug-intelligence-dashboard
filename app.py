import json
import math
import os

import pandas as pd
import streamlit as st
from openai import OpenAI


DEFAULT_CSV = "bugs.csv"


def load_bug_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(DEFAULT_CSV)


def bug_rows_for_prompt(df):
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "id": row.get("id"),
                "title": row.get("title"),
                "description": row.get("description"),
                "severity": row.get("severity"),
                "component": row.get("component"),
                "customer_impact": row.get("customer_impact"),
            }
        )
    return rows


def analyze_bugs_with_llm(df):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    prompt = f"""
You are analyzing product bug reports.

Classify each bug into one clear theme such as Login, Payments, UI,
Performance, Search, Notifications, Cart, Profile, or another concise category.

Return only valid JSON in this exact shape.

Important:
- Include one bug_categories entry for EVERY bug report provided.
- Use the exact id from each bug report.
- Do not skip any bug.
- Do not return only examples.
- category must be a short label such as Login, Payments, Cart, Search, Notifications, Performance, Profile, UI, or Other.

{{
  "bug_categories": [
    {{"id": 1, "category": "Login"}}
  ],
  "executive_summary": "Short plain-English summary for leadership.",
  "prioritized_recommendations": [
    "Fix checkout-blocking payment and cart defects first."
  ],
  "suggested_next_actions": [
    "Assign owners for the highest-severity issues within 24 hours."
  ],
  "severity_customer_impact_reasoning": "Explain how severity and customer impact shaped the recommendations."
}}

Base the recommendations and next actions on severity, customer impact, and
the concentration of bugs by theme. Prioritize customer-blocking, revenue-impacting,
and high-severity issues before cosmetic or low-impact issues.

Bug reports:
{json.dumps(bug_rows_for_prompt(df), indent=2)}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=1,
        #GVAR temperature=0.2,
    )

    return json.loads(response.choices[0].message.content)


def bug_text_for_embeddings(df):
    texts = []
    for _, row in df.iterrows():
        title = str(row.get("title", "") or "")
        description = str(row.get("description", "") or "")
        texts.append(f"{title}\n\n{description}")
    return texts


def generate_bug_embeddings(df):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    response = client.embeddings.create(
        model=model,
        input=bug_text_for_embeddings(df),
    )

    return [item.embedding for item in response.data]


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot_product / (norm_a * norm_b)


def find_similar_bugs(df, embeddings, top_n=2):
    rows = []
    bugs = df.reset_index(drop=True)

    for bug_index, bug in bugs.iterrows():
        scores = []

        for other_index, other_bug in bugs.iterrows():
            if bug_index == other_index:
                continue

            scores.append(
                {
                    "bug_id": bug.get("id"),
                    "bug_title": bug.get("title"),
                    "similar_bug_id": other_bug.get("id"),
                    "similar_bug_title": other_bug.get("title"),
                    "similarity": cosine_similarity(
                        embeddings[bug_index],
                        embeddings[other_index],
                    ),
                }
            )

        rows.extend(sorted(scores, key=lambda item: item["similarity"], reverse=True)[:top_n])

    duplicates_df = pd.DataFrame(rows)
    if not duplicates_df.empty:
        duplicates_df["similarity"] = duplicates_df["similarity"].round(3)
    return duplicates_df


def add_categories_to_data(df, analysis):
    category_by_id = {
        str(item["id"]): item["category"]
        for item in analysis.get("bug_categories", [])
    }

    df = df.copy()
    df["AI_Category"] = df["id"].astype(str).map(category_by_id).fillna("Uncategorized")
    return df


st.set_page_config(page_title="AI Bug Intelligence Dashboard", layout="wide")

st.title("AI Bug Intelligence Dashboard")

uploaded_file = st.file_uploader("Upload bug CSV", type=["csv"])
bugs_df = load_bug_data(uploaded_file)

st.subheader("Bug Reports")
st.dataframe(bugs_df, width='stretch')

if st.button("Analyze Bugs", type="primary"):
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Set OPENAI_API_KEY before running analysis.")
        st.stop()

    with st.spinner("Analyzing bugs with LLM..."):
        analysis = analyze_bugs_with_llm(bugs_df)
        returned_ids = {str(item["id"]) for item in analysis.get("bug_categories", [])}
        expected_ids = set(bugs_df["id"].astype(str))
        missing_ids = expected_ids - returned_ids
        if missing_ids:
            st.warning(f"LLM did not return categories for bug IDs: {sorted(missing_ids)}")
        analyzed_df = add_categories_to_data(bugs_df, analysis)
        embeddings = generate_bug_embeddings(bugs_df)
        similar_bugs_df = find_similar_bugs(bugs_df, embeddings)

    st.subheader("Analyzed Bug Reports")
    st.dataframe(analyzed_df, width='stretch')
    st.download_button(
        label="Download Analyzed Bugs CSV",
        data=analyzed_df.to_csv(index=False),
        file_name="analyzed_bugs.csv",
        mime="text/csv",
    )

    category_counts = analyzed_df["AI_Category"].value_counts().reset_index()
    category_counts.columns = ["AI_Category", "Bug Count"]

    st.subheader("Count of Bugs per Category")
    st.dataframe(category_counts, width='stretch')
    st.bar_chart(category_counts, x="AI_Category", y="Bug Count")

    st.subheader("Top 3 Problem Areas")
    top_problem_areas = (analyzed_df["AI_Category"].value_counts().head(3).index.tolist())
    for index, area in enumerate(top_problem_areas, start=1):
        st.write(f"{index}. {area}")

    st.subheader("Executive Summary")
    st.write(analysis.get("executive_summary", "No summary returned."))

    st.subheader("Prioritized Recommendations")
    for recommendation in analysis.get("prioritized_recommendations", []):
        st.write(f"- {recommendation}")

    st.subheader("Suggested Next Actions")
    for action in analysis.get("suggested_next_actions", []):
        st.write(f"- {action}")

    st.subheader("Reasoning")
    st.write(
        analysis.get(
            "severity_customer_impact_reasoning",
            "No severity and customer-impact reasoning returned.",
        )
    )

    st.subheader("Potential Duplicate Bugs")
    st.dataframe(similar_bugs_df, width='stretch')
