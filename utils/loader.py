import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from .query import collection


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

    existing_docs = collection.get()["documents"]
    if existing_docs:
        print("🟢 Collection already initialized. Skipping.")
        return

    print("📄 Loading and splitting constitution...")
    full_text = load_text(DATA_PATH)
    chunks = split_text(full_text)

    ids = [f"init_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)

    print(f"✅ Added {len(chunks)} chunks to the collection.")

def add_documents(docs, collection=collection):
    chunks = []
    ids = []
    for i, doc in enumerate(docs):
        content = doc.read().decode("utf-8")
        parts = [content[i:i+1000] for i in range(0, len(content), 1000)]
        for j, part in enumerate(parts):
            chunks.append(part)
            ids.append(f"doc_{i}_{j}")
    if chunks and ids:
        collection.add(documents=chunks, ids=ids)
    else:
        raise ValueError("No valid documents found to add!")

