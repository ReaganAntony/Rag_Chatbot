import streamlit as st
import requests
import os
from pathlib import Path
from datetime import datetime

API_BASE = "http://localhost:8000"

@st.cache_data(ttl=300)
def fetch_documents():
    r = requests.get(f"{API_BASE}/documents")
    if r.status_code == 200:
        return r.json()
    else:
        return []

st.set_page_config(page_title="PDF RAG Demo", layout="wide")

st.title("PDF RAG — Upload, Index, Chat")

# Initialize session state
if 'tab' not in st.session_state:
    st.session_state.tab = "Upload & Index"
if 'question' not in st.session_state:
    st.session_state.question = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'refresh_docs' not in st.session_state:
    st.session_state.refresh_docs = False

# Tab selection without rerun
tab_options = ["Upload & Index", "Documents", "Chat"]
tab = st.radio("Go to:", tab_options, index=tab_options.index(st.session_state.tab) if st.session_state.tab in tab_options else 0, key="tab_selector")
if tab != st.session_state.tab:
    st.session_state.tab = tab
    st.session_state.refresh_docs = True

if tab == "Upload & Index":
    st.header("Upload PDFs")
    uploaded = st.file_uploader("Choose PDF(s)", accept_multiple_files=True, type=["pdf"])
    auto_process = st.checkbox("Automatically process and index after upload", value=True)
    if st.button("Upload"):
        if not uploaded:
            st.warning("Choose at least one PDF")
        else:
            for f in uploaded:
                files = {"file": (f.name, f.getvalue(), "application/pdf")}
                r = requests.post(f"{API_BASE}/upload", files=files)
                if r.status_code == 200:
                    doc = r.json()
                    st.success("Document uploaded and indexed successfully")
                    fetch_documents.clear()  # Clear cache
                else:
                    st.error(f"Upload failed: {r.text}")

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go to Documents", key="nav_docs_upload"):
            st.session_state.tab = "Documents"
            st.rerun()
    with col2:
        if st.button("Go to Chat", key="nav_chat_upload"):
            st.session_state.tab = "Chat"
            st.rerun()

elif tab == "Documents":
    st.header("Documents")
    docs = fetch_documents()
    if docs:
        # Group documents by date
        date_groups = {}
        for d in docs:
            upload_date = datetime.fromisoformat(d['uploaded_at']).date()
            if upload_date not in date_groups:
                date_groups[upload_date] = []
            date_groups[upload_date].append(d)

        for date, docs_list in sorted(date_groups.items(), reverse=True):
            st.subheader(f"Uploaded on {date}")
            for d in docs_list:
                with st.expander(f"{d['filename']} "):
                    st.write(d)
                    if st.button("Delete", key=f"del_{d['doc_id']}"):
                        rr = requests.delete(f"{API_BASE}/documents/{d['doc_id']}")
                        if rr.status_code == 200:
                            st.success("Deleted")
                            fetch_documents.clear()  # Clear cache
                        else:
                            st.error(rr.text)
    else:
        st.info("No documents uploaded yet. Go to Upload & Index to add documents.")
    # Navigation buttons removed - using radio buttons instead

elif tab == "Chat":
    st.header("Chat with documents")
    # Fetch documents
    docs = fetch_documents()

    # Group documents by filename
    doc_groups = {}
    for d in docs:
        filename = d['filename']
        upload_date = datetime.fromisoformat(d['uploaded_at']).date()
        if filename not in doc_groups:
            doc_groups[filename] = {'dates': [], 'count': 0}
        doc_groups[filename]['dates'].append(upload_date)
        doc_groups[filename]['count'] += 1

    # Create two columns
    col1, col2 = st.columns(2)

    # Left column: Your Uploaded Documents
    with col1:
        if docs:
            st.subheader("Your Uploaded Documents")
            for filename, info in doc_groups.items():
                dates_str = ", ".join(sorted(set(str(d) for d in info['dates'])))
                badge = " (Multiple Uploads)" if info['count'] > 1 else ""
                with st.expander(f"{filename}{badge}"):
                    st.write(f"Uploaded on: {dates_str}")
        else:
            st.info("Upload documents to see personalized suggestions.")

    # Right column: Suggested Questions
    with col2:
        st.subheader("Suggested Questions")
        with st.container():
            st.info("Based on your documents, here are some questions you can ask:")
            suggested_questions = [
                "What is the main topic of the document?",
                "Summarize the key points.",
                "Explain [specific term] from the document."
            ]
            for q in suggested_questions:
                if st.button(q, key=f"sugg_{q.replace(' ', '_').replace('[', '').replace(']', '')}"):
                    st.session_state.question = q

    # Chat history disabled as per user request

    mode = st.selectbox("Mode", ["KB (all indexed docs)", "Single PDF"])
    doc_id = None
    if mode == "Single PDF":
        if docs:
            opt = {f"{d['filename']} (Uploaded: {datetime.fromisoformat(d['uploaded_at']).date()})": d["doc_id"] for d in docs}
            sel = st.selectbox("Select document", options=list(opt.keys()))
            doc_id = opt.get(sel)
    question = st.text_area("Ask a question", value=st.session_state.question)
    if st.button("Send"):
        if not question.strip():
            st.warning("Please enter a question")
        else:
            payload = {"question": question, "mode": "single" if mode=="Single PDF" else "kb", "doc_id": doc_id, "top_k": 5}
            r = requests.post(f"{API_BASE}/query", json=payload)
            if r.status_code == 200:
                resp = r.json()
                print(resp)
                # Add assistant message to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": resp["answer"],
                })
                st.subheader("Answer")
                st.write(resp["answer"])
                
            else:
                st.error(r.text)

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go to Upload", key="nav_upload_chat"):
            st.session_state.tab = "Upload & Index"
            st.rerun()
    with col2:
        if st.button("Go to Documents", key="nav_docs_chat"):
            st.session_state.tab = "Documents"
            st.rerun()
