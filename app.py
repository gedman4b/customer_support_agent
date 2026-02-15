from __future__ import annotations

import os
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Iterable



@dataclass(frozen=True)
class QAItem:
    question: str
    answer: str


KNOWLEDGE_BASE: tuple[QAItem, ...] = (
    QAItem(
        question="What does the eligibility verification agent (EVA) do?",
        answer=(
            "EVA automates the process of verifying a patient’s eligibility and benefits "
            "information in real-time, eliminating manual data entry errors and reducing "
            "claim rejections."
        ),
    ),
    QAItem(
        question="What does the claims processing agent (CAM) do?",
        answer=(
            "CAM streamlines the submission and management of claims, improving "
            "accuracy, reducing manual intervention, and accelerating reimbursements."
        ),
    ),
    QAItem(
        question="How does the payment posting agent (PHIL) work?",
        answer=(
            "PHIL automates the posting of payments to patient accounts, ensuring fast, "
            "accurate reconciliation of payments and reducing administrative burden."
        ),
    ),
    QAItem(
        question="Tell me about Thoughtful AI's Agents.",
        answer=(
            "Thoughtful AI provides a suite of AI-powered automation agents designed to "
            "streamline healthcare processes. These include Eligibility Verification "
            "(EVA), Claims Processing (CAM), and Payment Posting (PHIL), among others."
        ),
    ),
    QAItem(
        question="What are the benefits of using Thoughtful AI's agents?",
        answer=(
            "Using Thoughtful AI's Agents can significantly reduce administrative costs, "
            "improve operational efficiency, and reduce errors in critical processes "
            "like claims management and payment posting."
        ),
    ),
)

MATCH_THRESHOLD = 0.40
ACRONYM_TOKENS = {"eva", "cam", "phil"}
ACRONYM_BONUS = 0.2
ACRONYM_BONUS_MIN_OVERLAP = 0.25


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9']+", normalize(text)) if token}


def similarity_score(a: str, b: str) -> float:
    a_norm = normalize(a)
    b_norm = normalize(b)

    sequence = SequenceMatcher(None, a_norm, b_norm).ratio()
    a_tokens = tokenize(a_norm)
    b_tokens = tokenize(b_norm)

    overlap = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
    shared_tokens = a_tokens & b_tokens
    shared_acronym = bool(ACRONYM_TOKENS & shared_tokens)
    shared_non_acronym = bool(shared_tokens - ACRONYM_TOKENS)
    acronym_bonus = (
        ACRONYM_BONUS
        if shared_acronym and shared_non_acronym and overlap >= ACRONYM_BONUS_MIN_OVERLAP
        else 0.0
    )

    return (0.35 * sequence) + (0.65 * overlap) + acronym_bonus


def find_best_match(user_question: str, items: Iterable[QAItem] = KNOWLEDGE_BASE) -> tuple[QAItem | None, float]:
    best_item: QAItem | None = None
    best_score = 0.0

    for item in items:
        score = similarity_score(user_question, item.question)
        if score > best_score:
            best_score = score
            best_item = item

    if best_score < MATCH_THRESHOLD:
        return None, best_score

    return best_item, best_score


def llm_fallback(user_question: str) -> str:
    """Fallback to an LLM-like response.

    If OPENAI_API_KEY is available and the openai package is installed, this function
    makes a lightweight call to a chat model. Otherwise, it returns a safe generic
    assistant response so the app still works in local/offline environments.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a concise, friendly customer support assistant for "
                            "general questions."
                        ),
                    },
                    {"role": "user", "content": user_question},
                ],
                temperature=0.2,
            )
            return completion.choices[0].message.content or "I'm here to help."
        except Exception:
            pass

    return (
        "I don't have a predefined Thoughtful AI answer for that yet, but I can still help. "
        "Could you share a little more detail so I can provide a useful next step?"
    )


def answer_question(user_question: str) -> tuple[str, bool, float]:
    match, score = find_best_match(user_question)
    if match:
        return match.answer, True, score
    return llm_fallback(user_question), False, score


def render_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="Thoughtful AI Support Agent", page_icon="🤖", layout="centered")
    st.title("🤖 Thoughtful AI Support Agent")
    st.caption("Ask about Thoughtful AI's healthcare automation agents.")

    with st.expander("Sample questions"):
        for item in KNOWLEDGE_BASE:
            st.markdown(f"- {item.question}")

    user_input = st.text_input("How can I help today?", placeholder="e.g., What does EVA do?")

    if st.button("Get answer", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a question.")
            return

        response, used_kb, score = answer_question(user_input)
        st.subheader("Answer")
        st.write(response)

        if used_kb:
            st.success(f"Answered using Thoughtful AI knowledge base (match score: {score:.2f}).")
        else:
            st.info("No close hardcoded match found; used fallback assistant response.")


if __name__ == "__main__":
    render_app()
