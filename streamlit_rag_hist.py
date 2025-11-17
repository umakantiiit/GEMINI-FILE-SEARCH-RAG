"""
Streamlit app: Gemini File Search + simple chat UI (updated)

Changes in this version:
- Removed metadata handling entirely (no custom_metadata sent to the API).
- Implemented an ephemeral per-browser-session "conversation memory" that accumulates previous Q&A pairs in the prompt.
  - The prompt format sent to Gemini contains two sections: PREVIOUSLY_ASKED and CURRENTLY_ASKING.
  - After each question/answer exchange the Q&A pair is appended to the session memory.
  - Memory is stored in `st.session_state` and will reset when the user closes the browser or the session expires.

How to run:
1. pip install streamlit google-genai python-docx PyPDF2 pandas
2. streamlit run streamlit_gemini_file_search.py

"""

import time
from pathlib import Path
from typing import List, Dict

import streamlit as st

# You may need to install the package that contains `genai`.
# pip install google-genai
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None


def filename_display_name(filename: str) -> str:
    """Return the filename without extension to use for display_name"""
    return Path(filename).stem


def build_prompt(history: List[Dict[str, str]], current_question: str) -> str:
    """Build the prompt with PREVIOUSLY_ASKED and CURRENTLY_ASKING sections.
    history is a list of dicts with keys: 'question' and 'answer'.
    """
    prev_lines: List[str] = []
    for item in history:
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        if q or a:
            prev_lines.append(f"Q: {q}\nA: {a}")

    previously = "\n".join(prev_lines)
    prompt = (
        "There are 2 sections in this. Previously asked and currently asking. While answering you need to check the previously asked then answer the query that is present in the currently asking section.\n"
        "PREVIOUSLY_ASKED:\n\n"
        f"{previously}\n\n"
        "CURRENTLY_ASKING:\n"
        f"{current_question}"
    )
    return prompt


# Streamlit UI
st.set_page_config(page_title="Gemini File Search — Streamlit", layout="centered")
st.title("Welcome To Gemini RAG")

st.markdown(
    """
This demo lets you upload one document (no metadata), create a File Search store,
then ask questions. The app keeps an ephemeral per-browser-session memory of prior Q&A
that is placed into the prompt for subsequent queries. Memory resets when the browser session ends.
"""
)

# API key input (in the middle)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    api_key = st.text_input("Gemini API Key", type="password", help="Paste your Gemini (Google GenAI) API key here")

# File uploader (max 1)
uploaded_file = st.file_uploader(
    "Upload ONE document (pdf, csv, json, docx, txt)",
    type=["pdf", "csv", "json", "docx", "txt"],
    accept_multiple_files=False,
)

# Upload button
start_upload = st.button("Upload Data And Start Chatting", key="upload_btn")

# Session state initialization
if 'client' not in st.session_state:
    st.session_state['client'] = None
if 'file_search_store_name' not in st.session_state:
    st.session_state['file_search_store_name'] = None
if 'uploaded_filename' not in st.session_state:
    st.session_state['uploaded_filename'] = None
if 'file_search_ready' not in st.session_state:
    st.session_state['file_search_ready'] = False
# ephemeral conversation memory: list of {'question':..., 'answer':...}
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []


# When upload button is clicked
if start_upload:
    if not api_key:
        st.error("Please enter your Gemini API key before uploading.")
    elif not uploaded_file:
        st.error("Please upload a file (max 1) to continue.")
    else:
        # initialize client
        if genai is None or types is None:
            st.error("The required GenAI library is not installed or failed to import.\nRun: pip install google-genai")
        else:
            try:
                client = genai.Client(api_key=api_key)
                st.session_state['client'] = client
            except Exception as e:
                st.error(f"Failed to create GenAI client: {e}")
                client = None

            if client:
                try:
                    # Create file search store with display name set to filename stem
                    st.info("Creating File Search store...")
                    file_search_store = client.file_search_stores.create(
                        config={"display_name": filename_display_name(uploaded_file.name)}
                    )
                    st.session_state['file_search_store_name'] = file_search_store.name
                    st.session_state['uploaded_filename'] = uploaded_file.name

                    # Save uploaded file to a temp path to pass to API if needed
                    tmp_dir = Path("./.tmp_uploaded_files")
                    tmp_dir.mkdir(exist_ok=True)
                    tmp_path = tmp_dir / uploaded_file.name
                    with open(tmp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    st.info(f"Uploading and importing file into File Search store: {uploaded_file.name}")
                    # NOTE: No custom_metadata is passed per user's request
                    operation = client.file_search_stores.upload_to_file_search_store(
                        file=str(tmp_path),
                        file_search_store_name=file_search_store.name,
                        config={
                            "display_name": filename_display_name(uploaded_file.name),
                        },
                    )

                    # Wait for operation to complete (basic poll)
                    with st.spinner('Importing file to Gemini File Search (this may take some time)...'):
                        poll_seconds = 2
                        max_wait = 300  # seconds
                        waited = 0
                        while not getattr(operation, 'done', False) and waited < max_wait:
                            time.sleep(poll_seconds)
                            waited += poll_seconds
                            try:
                                operation = client.operations.get(operation)
                            except Exception:
                                # some SDKs surface operation differently; break to avoid infinite loop
                                break

                    if getattr(operation, 'done', False):
                        st.success("File successfully uploaded and imported. You can now ask questions about it below.")
                        st.session_state['file_search_ready'] = True
                        # reset any prior conversation (fresh upload)
                        st.session_state['conversation_history'] = []
                    else:
                        st.warning(
                            "Upload operation did not report completion within the app's wait period. The file may still be importing in the background. You can try to ask questions; the store may become available shortly."
                        )
                        st.session_state['file_search_ready'] = True

                except Exception as e:
                    st.exception(e)


# Chat UI — only enabled after a successful upload
st.markdown("---")
if st.session_state.get('file_search_ready'):
    st.subheader("Chat with your document")
    question = st.text_area("Ask a question about the uploaded file:")
    if st.button("Get answer"):
        if not api_key or not st.session_state.get('client'):
            st.error("Missing client or API key. Re-enter your API key and press 'Upload Data And Start Chatting' again.")
        elif not st.session_state.get('file_search_store_name'):
            st.error("File Search store is not available. Upload again.")
        elif not question or not question.strip():
            st.error("Please enter a question.")
        else:
            client = st.session_state['client']
            file_search_store_name = st.session_state['file_search_store_name']
            try:
                # Build prompt including previous history
                prompt = build_prompt(st.session_state['conversation_history'], question)

                # Call generate_content (example from user's snippet)
                response = client.models.generate_content(
                    model="gemini-2.5-flash-preview-09-2025",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[
                            types.Tool(
                                file_search=types.FileSearch(
                                    file_search_store_names=[file_search_store_name]
                                )
                            )
                        ]
                    ),
                )

                # Extract answer text; SDKs differ so try a couple of attributes
                answer_text = getattr(response, 'text', None)
                if answer_text is None:
                    # try other likely places
                    answer_text = getattr(response, 'output', None)
                if answer_text is None:
                    # fallback: stringify full response
                    st.write(response)
                else:
                    st.success("Answer:")
                    st.write(answer_text)

                    # Append Q&A to session conversation history so next prompt includes it
                    st.session_state['conversation_history'].append({'question': question, 'answer': answer_text})

            except Exception as e:
                st.exception(e)
else:
    st.info("Upload a file and click 'Upload Data And Start Chatting' to enable the chat interface.")


# Small housekeeping: provide instructions and optional debug info
st.markdown("---")
with st.expander("How this works (technical)"):
    st.write(
        """
        1. Enter your Gemini API key in the top box.
        2. Upload a single file (pdf/csv/json/docx/txt). No metadata is requested.
        3. Click Upload Data And Start Chatting — the app will create a File Search store and import the file.
        4. After import, use the chat box to ask questions. The app builds a prompt containing two sections: PREVIOUSLY_ASKED (which contains prior Q&A pairs from your current browser session) and CURRENTLY_ASKING (your new question). The combined prompt is sent to Gemini with the File Search tool enabled.
        5. The conversation memory lives only in your browser session (Streamlit session_state) and resets when you close the browser tab or the session expires.
        """
    )

if st.checkbox("Show debug session state"):
    st.json({
        'file_search_store_name': st.session_state.get('file_search_store_name'),
        'uploaded_filename': st.session_state.get('uploaded_filename'),
        'conversation_history': st.session_state.get('conversation_history')
    })

# End of app
