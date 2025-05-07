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

collection = chroma_client.get_or_create_collection(
    name="constitution_collection",
    embedding_function=embedding
)

def query_chromadb(query_text, n_results=3):
    results = collection.query(query_texts=[query_text], n_results=n_results)
    return results["documents"]

def generate_answer(context, question):
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    llm = OllamaLLM(model=llm_model)
    return llm.invoke(prompt)

def reset_collection():
    chroma_client.delete_collection("constitution_collection")
    global collection
    collection = chroma_client.get_or_create_collection(
        name="constitution_collection",
        embedding_function=embedding
    )
