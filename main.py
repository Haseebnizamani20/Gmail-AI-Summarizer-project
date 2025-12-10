# main.py - Gmail Inbox Summarizer (Streamlit + LangChain + Ollama)

import os
import base64
from bs4 import BeautifulSoup
from typing import List, Dict
import streamlit as st

# Gmail
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

# LangChain (classic) + Ollama
from langchain_ollama import OllamaLLM
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import LLMChain

# ---------------------------
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"
# ---------------------------


def get_gmail_service():
    """Authenticate and return Gmail service object."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    service = build("gmail", "v1", credentials=creds)
    return service


def _decode_part(data: str) -> str:
    if not data:
        return ""
    decoded = base64.urlsafe_b64decode(data.encode("utf-8"))
    return decoded.decode("utf-8", errors="ignore")


def extract_body_from_payload(payload: Dict) -> str:
    body = payload.get("body", {}).get("data")
    if body:
        return _decode_part(body)

    parts = payload.get("parts", []) or []
    text_parts, html_parts = [], []

    for part in parts:
        mime = part.get("mimeType", "")
        data = part.get("body", {}).get("data")

        if data:
            if mime == "text/plain":
                text_parts.append(_decode_part(data))
            elif mime == "text/html":
                html_parts.append(_decode_part(data))

        for sub in part.get("parts", []) or []:
            s_mime = sub.get("mimeType", "")
            s_data = sub.get("body", {}).get("data")
            if s_data:
                if s_mime == "text/plain":
                    text_parts.append(_decode_part(s_data))
                elif s_mime == "text/html":
                    html_parts.append(_decode_part(s_data))

    if text_parts:
        return "\n\n".join(text_parts)
    if html_parts:
        return BeautifulSoup(html_parts[0], "html.parser").get_text()

    return "(No readable body found)"


def fetch_unread_emails(service, max_results: int = 10) -> List[Dict]:
    resp = service.users().messages().list(userId="me", labelIds=["UNREAD"], maxResults=max_results).execute()
    msgs = resp.get("messages", [])
    results = []

    for m in msgs:
        mdata = service.users().messages().get(userId="me", id=m["id"], format="full").execute()
        payload = mdata.get("payload", {})

        headers = payload.get("headers", [])
        subject = "(No Subject)"
        sender = "(Unknown Sender)"

        for h in headers:
            name = h.get("name", "").lower()
            if name == "subject":
                subject = h.get("value", subject)
            elif name == "from":
                sender = h.get("value", sender)

        body = extract_body_from_payload(payload)
        snippet = mdata.get("snippet", "")

        results.append({
            "id": m["id"],
            "subject": subject,
            "sender": sender,
            "body": body,
            "snippet": snippet
        })

    return results


# ===== LLM Chains =====
llm = OllamaLLM(model="gemma:2b")

summary_prompt = PromptTemplate(
    input_variables=["email"],
    template="Summarize this email in 2 short lines:\n\n{email}"
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

classify_prompt = PromptTemplate(
    input_variables=["email"],
    template=(
        "Classify this email into one single category:\n"
        "[important, task, invoice, meeting, promotional, personal]\n\n"
        "Email: {email}\nCategory:"
    )
)
classify_chain = LLMChain(llm=llm, prompt=classify_prompt)

extract_prompt = PromptTemplate(
    input_variables=["email"],
    template=(
        "From the email below, extract:\n"
        "1) date/deadline (if any)\n"
        "2) amount (if any)\n"
        "3) main action item (one short sentence)\n\n"
        "Email: {email}\n\n"
        "Return as a short JSON-like answer (single line)."
    )
)
extract_chain = LLMChain(llm=llm, prompt=extract_prompt)


def analyze_email(email: Dict) -> Dict:
    text = f"Subject: {email['subject']}\nFrom: {email['sender']}\n\n{email['body']}"

    summary = summary_chain.run(text)
    category = classify_chain.run(text)
    extracted = extract_chain.run(text)

    return {
        "subject": email["subject"],
        "sender": email["sender"],
        "summary": summary.strip(),
        "category": category.strip(),
        "extracted": extracted.strip()
    }


# ===========================
# STREAMLIT UI
# ===========================

def main():
    st.title("📧 Gmail AI Inbox Summarizer")
    st.write("Automatically summarize, classify, and extract key info using **Ollama + LangChain**")

    if st.button("🔐 Connect to Gmail"):
        with st.spinner("Authenticating..."):
            service = get_gmail_service()
        st.success("Connected successfully!")

        with st.spinner("Fetching unread emails..."):
            emails = fetch_unread_emails(service, max_results=10)

        st.info(f"Found {len(emails)} unread emails")

        for e in emails:
            with st.expander(f"📩 {e['subject']} — {e['sender']}"):
                st.subheader("Raw Body")
                st.text(e["body"])

                with st.spinner("Running AI analysis..."):
                    result = analyze_email(e)

                st.subheader("AI Summary")
                st.write(result["summary"])

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Category")
                    st.write(result["category"])
                with col2:
                    st.subheader("Extracted Info")
                    st.write(result["extracted"])


if __name__ == "__main__":
    main()
