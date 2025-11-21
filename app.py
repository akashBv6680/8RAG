import streamlit as st
import requests

# List of selectable RAG architectures
RAG_TYPES = [
    "RAG-Token",
    "RAG-Sequence",
    "FiD (Fusion-in-Decoder)",
    "HyDE",
    "Atlas",
    "OpenRAG",
    "Multi-hop RAG",
    "Tool-Augmented RAG"
]

RAG_DESCRIPTIONS = {
    "RAG-Token": "Retrieval for every token generation; maximizes reference granularity.",
    "RAG-Sequence": "Retrieves once per sequence; faster, suitable for document answers.",
    "FiD (Fusion-in-Decoder)": "Fuses multiple retrieved docs in the decoder for richer context.",
    "HyDE": "Generates hypothetical docs, then searches for closest matches.",
    "Atlas": "Handles retrieval + fine-tuning, excels with long, multi-step reasoning.",
    "OpenRAG": "Open-source RAG frameworks for customizable architectures.",
    "Multi-hop RAG": "Supports multi-step queries that require chaining retrievals.",
    "Tool-Augmented RAG": "Integrates retrievals from external tools (search, DB, APIs)."
}

def query_gemini(prompt: str) -> str:
    api_key = st.secrets["GEMINI_API_KEY"]
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7}
    }
    response = requests.post(f"{endpoint}?key={api_key}", json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    result = response.json()
    return result["candidates"][0]["content"]["parts"][0]["text"]

st.title("RAG Architecture Playground with Gemini Flash 2.5")

selected_rag = st.selectbox("Select RAG Architecture", RAG_TYPES)
st.write(f"**About {selected_rag}**: {RAG_DESCRIPTIONS[selected_rag]}")

user_query = st.text_area("Enter your question/query")

if st.button("Run Gemini Flash 2.5"):
    if not user_query:
        st.error("Please enter a query.")
    else:
        rag_prefix = f"Use the '{selected_rag}' RAG architecture. {RAG_DESCRIPTIONS[selected_rag]}\n\nQuery: {user_query}\n\n"
        try:
            result = query_gemini(rag_prefix)
            st.markdown("### Gemini Flash 2.5 Response")
            st.write(result)
        except Exception as e:
            st.error(f"Gemini API error: {e}")
