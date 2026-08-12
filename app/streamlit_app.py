"""
streamlit_app.py — Simple chat UI for the FetchWise agent.

Shows the conversation plus, per turn, which path the agent took
(retrieval / tool / refusal) and the evidence behind the answer: the
retrieved doc excerpts and/or the tool call + result. This is meant to make
the agent's decision-making visible, not just its final text — useful for
demoing and for spot-checking behavior alongside eval.py's automated scores.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Make src/ importable when Streamlit runs this file directly.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from agent import FetchWiseAgent  # noqa: E402

PATH_LABELS = {
    "retrieval": ("📄", "Answered from knowledge base"),
    "tool": ("🔧", "Called check_order_status"),
    "refusal": ("🚫", "Refused / no answer available"),
}


@st.cache_resource(show_spinner="Loading FetchWise agent (embedding model + FAISS index)...")
def get_agent() -> FetchWiseAgent:
    return FetchWiseAgent()


def render_path_badge(path: str) -> None:
    icon, label = PATH_LABELS.get(path, ("❓", path))
    st.caption(f"{icon} **Path:** {label}")


def render_evidence(resp) -> None:
    """Show what actually grounded the answer, so the path badge isn't just
    a claim — the retrieved excerpts / tool call are inspectable too."""
    with st.expander("Show retrieved excerpts / tool call"):
        if resp.tool_call:
            st.markdown(f"**Tool call:** `check_order_status(order_id=\"{resp.tool_call.arguments.get('order_id')}\")`")
            st.json(resp.tool_call.result)

        if resp.retrieved_chunks:
            st.markdown("**Retrieved chunks (top-k):**")
            for chunk in resp.retrieved_chunks:
                used = "✅ used as context" if chunk.score >= 0.35 else "— below relevance threshold, not used"
                st.markdown(f"- `{chunk.doc_id}` (score {chunk.score:.3f}) {used}")
        elif not resp.tool_call:
            st.caption("No excerpts retrieved.")


def main() -> None:
    st.set_page_config(page_title="FetchWise", page_icon="🐾")
    st.title("🐾 FetchWise")
    st.caption("Support assistant for Fetchly — ask about orders, shipping, returns, subscriptions, and more.")

    try:
        agent = get_agent()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {"role", "content", "resp" (assistant only)}

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_path_badge(msg["resp"].path)
                render_evidence(msg["resp"])

    query = st.chat_input("Ask a question about your order, our policies, etc.")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = agent.answer(query)
            st.markdown(resp.answer)
            render_path_badge(resp.path)
            render_evidence(resp)

        st.session_state.messages.append({"role": "assistant", "content": resp.answer, "resp": resp})

    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            "FetchWise combines retrieval-augmented generation over a small "
            "FAQ/policy knowledge base with one tool call (mock order lookup) "
            "and a guardrail that refuses clearly off-topic questions.\n\n"
            "Every answer below is tagged with the path the agent actually took, "
            "and you can expand it to see the retrieved excerpts or tool call "
            "that grounded the response."
        )
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
