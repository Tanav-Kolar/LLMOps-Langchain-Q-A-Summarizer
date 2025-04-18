import streamlit as st
from main import qa_chain, log_feedback, init_db

init_db()
st.title("LangChain Document Q&A")

query = st.text_input("Ask a question about the document")
if query:
    response = qa_chain.run(query)
    st.write("### Answer:", response)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Helpful"):
            log_feedback(query, response, "up")
            st.success("Thanks for your feedback!")
    with col2:
        if st.button("👎 Not Helpful"):
            log_feedback(query, response, "down")
            st.warning("Thanks! We'll use this to improve.")
