"""
Quick demo UI. Run with:
    streamlit run ui/streamlit_app.py
"""
import streamlit as st

from agent.graph import agent

st.set_page_config(page_title="GraphRAG Research Assistant", page_icon="📚")
st.title("📚 GraphRAG Research Assistant")
st.caption("Ask about the ingested papers — content questions use RAG, relational questions use the knowledge graph.")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask a question about the papers...")

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Routing query and retrieving..."):
            result = agent.invoke({"question": question})
        st.write(result.get("answer", "No answer produced."))
        st.caption(f"Route: {result.get('query_type')} | Grounded: {result.get('is_grounded')}")
        if result.get("citations"):
            st.caption("Sources: " + ", ".join(result["citations"]))

    st.session_state.history.append({"role": "assistant", "content": result.get("answer", "")})
