
import os
import pickle

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


# File paths
DATA_PATH = "data/rag_dataset.csv"
FAISS_STORE_PATH = "data/faiss_store"
BM25_PATH = "data/bm25_retriever.pkl"


# 1. Load the product dataset
def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path).fillna("")


# 2. Convert product records into LangChain Documents
def create_langchain_documents(df):
    documents = []

    for _, row in df.iterrows():
        text = (
            f"Product Name: {row['product_name']}. "
            f"Brand: {row['brand']}. "
            f"Category: {row['category']}. "
            f"Material: {row['material']}. "
            f"Color: {row['color']}. "
            f"Gender: {row['gender']}. "
            f"Season: {row['season']}. "
            f"Size: {row['size_available']}. "
            f"Price: ${row['price_usd']}. "
            f"Description: {row['description']}. "
            f"Care Instructions: {row['care_instructions']}."
        )

        doc = Document(
            page_content=text,
            metadata={
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "brand": row["brand"],
                "category": row["category"],
                "gender": row["gender"],
                "season": row["season"],
                "size_available": row["size_available"],
                "price": row["price_usd"],
            },
        )

        documents.append(doc)

    return documents


# 3. Load the free Hugging Face embedding model
def load_embedding_model():
    print("Loading free local embedding model...")

    return HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        ),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# 4. Build and save the FAISS index locally
def build_and_save_faiss(documents, embeddings):
    print("Building FAISS vector index...")

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    vector_store.save_local(FAISS_STORE_PATH)

    return vector_store


# 5. Build and save the BM25 keyword retriever
def build_and_save_bm25(documents):
    print("Building BM25 keyword retriever...")

    bm25_retriever = BM25Retriever.from_documents(
        documents
    )

    bm25_retriever.k = 30

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)

    return bm25_retriever


# 6. Main indexing pipeline
def main():
    print("\nStarting local RAG indexing...\n")

    os.makedirs("data", exist_ok=True)

    print("Loading product dataset...")
    df = load_dataset(DATA_PATH)

    if df.empty:
        raise ValueError("Product dataset is empty.")

    print(f"Loaded {len(df)} products.")

    print("Creating product documents...")
    documents = create_langchain_documents(df)

    print("Loading Hugging Face embeddings...")
    embeddings = load_embedding_model()

    build_and_save_faiss(documents, embeddings)

    build_and_save_bm25(documents)

    print("\nRAG indexes created successfully!")
    print(f"Total documents: {len(documents)}")
    print(f"FAISS index: {FAISS_STORE_PATH}")
    print(f"BM25 index: {BM25_PATH}")
    print("\nNo Google Cloud credentials required.")


if __name__ == "__main__":
    main()
