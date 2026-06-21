import streamlit as st

from school_rag.config import Settings
from school_rag.factory import build_workflow


st.set_page_config(page_title="Egyptian Private Universities RAG")
st.title("Egyptian Private Universities Assistant")
st.caption("Answers are grounded in the indexed UniversitiesEgypt dataset.")


@st.cache_resource
def workflow():
    return build_workflow(Settings.from_env())


question = st.chat_input("Ask about Egyptian private universities…")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            result = workflow().run(question)
            st.write(result["answer"])
            st.caption(f"Rewritten query: {result['rewritten_query']}")
            with st.expander("Retrieved evidence"):
                for rank, chunk in enumerate(result.get("retrieved_chunks", []), 1):
                    st.markdown(
                        f"**{rank}. {chunk['score']:.4f} {chunk['score_type']} — "
                        f"{chunk['title']} / {chunk['section']}**"
                    )
                    st.write(chunk["text"])
                    st.link_button("Open source", chunk["source"], key=f"source-{rank}")
        except Exception as exc:
            st.error(str(exc))
