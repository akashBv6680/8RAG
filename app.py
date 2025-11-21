import streamlit as st
import requests
import time

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
    "HyDE": "Generates hypothetical documents, then searches for closest matches.",
    "Atlas": "Handles retrieval + fine-tuning, excels with long, multi-step reasoning.",
    "OpenRAG": "Open-source RAG frameworks for customizable architectures.",
    "Multi-hop RAG": "Supports multi-step queries that require chaining retrievals.",
    "Tool-Augmented RAG": "Integrates retrievals from external tools (search, DB, APIs)."
}

def query_gemini(prompt: str) -> str:
    # IMPORTANT: Updated to the correct supported model gemini-2.5-flash-preview-09-2025
    model_name = "gemini-2.5-flash-preview-09-2025"
    api_key = st.secrets["GEMINI_API_KEY"] # Assumes API key is set in Streamlit secrets
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        # Using a low temperature as this is often desired for RAG grounding tasks
        "generationConfig": {"temperature": 0.2} 
    }

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(f"{endpoint}?key={api_key}", json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            # Extract text from the response
            if result.get("candidates") and result["candidates"][0].get("content"):
                return result["candidates"][0]["content"]["parts"][0]["text"]
            
            # If response is successful but structure is unexpected, raise an error
            raise ValueError("Unexpected API response structure.")

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1 and (response.status_code == 429 or response.status_code >= 500):
                # Handle rate limiting (429) or server errors (5xx) with exponential backoff
                sleep_time = 2 ** attempt
                time.sleep(sleep_time)
                continue
            else:
                raise e # Re-raise if it's not a retryable error or max retries reached
        except Exception as e:
            # Handle other errors like JSON parsing or ValueError from unexpected structure
            raise e
            
    # Should be unreachable if logic is correct, but for completeness
    raise Exception("Failed to get response from Gemini API after multiple retries.")


# Updated title to reflect the correct model
st.title("RAG Architecture Playground with Gemini 2.5 Flash (Preview)")

selected_rag = st.selectbox("Select RAG Architecture", RAG_TYPES)
st.markdown(f"**About {selected_rag}**: {RAG_DESCRIPTIONS[selected_rag]}")

user_query = st.text_area("Enter your question/query")

# Updated button text
if st.button("Run Gemini 2.5 Flash (Preview)"):
    if not user_query:
        st.error("Please enter a query.")
    else:
        # Construct the prompt to instruct the model on the desired RAG behavior
        rag_prefix = (
            f"INSTRUCTION: Simulate using the '{selected_rag}' RAG architecture. "
            f"({RAG_DESCRIPTIONS[selected_rag]}). "
            f"Answer the following query by explicitly stating how the selected RAG method "
            f"would influence the retrieval or generation process for this specific query.\n\n"
        )
        full_prompt = rag_prefix + f"QUERY: {user_query}"

        with st.spinner(f"Running query using {selected_rag}..."):
            try:
                result = query_gemini(full_prompt)
                st.markdown("### Gemini 2.5 Flash (Preview) Response")
                st.info(f"The model has simulated the behavior of the **{selected_rag}** architecture.")
                st.write(result)
            except Exception as e:
                st.error(f"Gemini API error: {e}")
