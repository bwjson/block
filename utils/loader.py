import os
import chromadb
import PyPDF2
import docx2txt
import chromadb.errors
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .query import helper_create_collection, user_collection, constitution_collection

DATA_PATH = os.path.join("data", "const.txt")

def load_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def split_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def initialize_documents():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    existing_docs = constitution_collection.get()["documents"]
    if existing_docs:
        print("Collection already initialized. Skipping.")
        return

    print("Loading and splitting constitution...")
    full_text = load_text(DATA_PATH)
    chunks = split_text(full_text)

    ids = [f"init_{i}" for i in range(len(chunks))]
    constitution_collection.add(documents=chunks, ids=ids)

    print(f"Added {len(chunks)} chunks to the collection.")

def load_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def load_docx(file):
    text = docx2txt.process(file)
    return text

def add_documents(docs):
    global user_collection

    try:
        _ = user_collection.get()
    except chromadb.errors.NotFoundError:
        print("Collection not found. Creating a new one.")
        user_collection = helper_create_collection("user_docs_collection")

    chunks = []
    ids = []
    for i, doc in enumerate(docs):
        file_type = doc.type
        content = ""

        if file_type == "text/plain":
            content = doc.read().decode("utf-8")
        elif file_type == "application/pdf":
            content = load_pdf(doc)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            content = load_docx(doc)

        parts = [content[i:i+1000] for i in range(0, len(content), 1000)]
        for j, part in enumerate(parts):
            chunks.append(part)
            ids.append(f"user_{i}_{j}")

    if chunks and ids:
        user_collection.add(documents=chunks, ids=ids)


