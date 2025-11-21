import chromadb
import os
import re
from glob import glob

CHROMA_PATH = "./chroma_db_data"
client = chromadb.PersistentClient(path=CHROMA_PATH)

user_query = input("Enter your query: ")

COLLECTION_NAME = "qwen_rag_knowledge"
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
)

results = collection.query(
    query_texts=[user_query],
    n_results=2,
    include=['documents', 'distances'] # Specify what data to return
)

print(results)