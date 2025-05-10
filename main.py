import streamlit as st
from utils.loader import add_documents, initialize_documents
from utils.query import query_chromadb, generate_answer, reset_collection

# For consitution initialization IMPORTANT: Make sure to have the file "const.txt" in the data folder.
# initialize_documents()

st.set_page_config(page_title="Kazakhstan Constitution Assistant")
st.title("📘 AI Assistant: Constitution of Kazakhstan 🇰🇿")

use_constitution = st.checkbox("📘 Use Constitution", value=True)

st.markdown("Upload the Constitution or other documents:")

uploaded_files = st.file_uploader("📎 Upload files (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    add_documents(uploaded_files)
    st.success("✅ Documents successfully added to the knowledge base!")

st.markdown("---")
st.markdown("Ask a question based on the uploaded documents:")

user_query = st.text_input("❓ Enter your question")

if st.button("🗑️ Clear Collection"):
    reset_collection()
    st.success("✅ Collection cleared and reset. The database is now empty.")

if user_query:
    docs = query_chromadb(user_query, use_constitution=use_constitution)
    context = " ".join([d for sublist in docs for d in sublist]) if docs else "No context found"
    answer = generate_answer(context, user_query)

    if context.strip() and context != "No context found":
        st.markdown("**📎 Context:**")
        st.info(context[:1000] + "..." if len(context) > 1000 else context)

    st.markdown("**🤖 Answer:**")
    st.success(answer)
