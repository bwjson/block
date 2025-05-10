import os
import chromadb
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from config import OLLAMA_BASE_URL

llm_model = "llama3"

chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))

class ChromaDBEmbeddingFunction:
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)

embedding = ChromaDBEmbeddingFunction(
    OllamaEmbeddings(model=llm_model, base_url=OLLAMA_BASE_URL)
)

constitution_collection = chroma_client.get_or_create_collection(
    name="constitution_collection",
    embedding_function=embedding
)

user_collection = chroma_client.get_or_create_collection(
    name="user_docs_collection",
    embedding_function=embedding
)

def query_chromadb(query_text, use_constitution=True, n_results=3):
    target_collection = constitution_collection if use_constitution else user_collection
    try:
        docs = target_collection.get()["documents"]
        if not docs:
            return []  
        results = target_collection.query(query_texts=[query_text], n_results=n_results)
        return results["documents"]
    except Exception as e:
        print("Error during query:", e)
        return []

def generate_answer(context, question):
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    llm = OllamaLLM(model=llm_model)
    return llm.invoke(prompt)

def reset_collection():
    chroma_client.delete_collection("user_docs_collection")
    global user_collection
    user_collection = chroma_client.get_or_create_collection(
        name="user_docs_collection",
        embedding_function=embedding
    )

def helper_create_collection(collection_name):
    try:
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding
        )
        return collection
    except Exception as e:
        print(f"❌ Error creating collection {collection_name}: {e}")
        return None
