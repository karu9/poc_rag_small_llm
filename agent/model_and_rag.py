import os
from llama_cpp import Llama
import chromadb
import argparse


MODEL_PATH = "../models/qwen1_5-1_8b-chat-q3_k_m.gguf"

CHROMA_PATH = "../chroma/chroma_db_data"
# ==========================================================


# Argument parsing for collection name
parser = argparse.ArgumentParser(description="RAG assistant with Qwen and ChromaDB.")
parser.add_argument('--collection', type=str, default='default', help='Name of the ChromaDB collection')
args = parser.parse_args()

COLLECTION_NAME = args.collection


print("--- 1. Qwen Model Initialization (LLM) ---")
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,  # Uses Metal acceleration on your Mac M4
    n_ctx=4096,  # Context window
    chat_format="chatml",  # Qwen format
    verbose=False
)
print("✅ Qwen model loaded.")

# 2. CHROMA INITIALIZATION (RETRIEVAL)
# -------------------------------------
print("--- 2. Connecting to ChromaDB ---")
# Connect to persistent client
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Get or create the collection
collection = client.get_or_create_collection(name=COLLECTION_NAME)
print(f"✅ Collection '{COLLECTION_NAME}' is ready. Number of documents: {collection.count()}")


# 3. MAIN RAG QUERY FUNCTION
# --------------------------
def run_rag_query(user_question: str):
    print(f"\n[PHASE 1: RETRIEVAL] Searching for relevant documents for: '{user_question}'")

    # 1. Vector search in ChromaDB
    # Request the 3 most relevant documents
    results = collection.query(
        query_texts=[user_question],
        n_results=3,
        include=['documents']
    )

    # 2. Build RAG context from results
    retrieved_documents = results['documents'][0]

    # Create a structured context string
    context_string = ""
    for i, doc_text in enumerate(retrieved_documents):
        context_string += f"--- reference document {i + 1} ---\n{doc_text}\n"

    print(f"[PHASE 2: CONTEXT] Injected context (size: {len(context_string.split())} tokens/words)...")

    # 3. Build the full prompt for the LLM
    # This is the crucial part of RAG!
    system_prompt = (
        "You are an expert RAG assistant. You must answer the user's question "
        "using *only* the information provided in the CONTEXT SECTION BELOW. "
        "If the CONTEXT does not contain the answer, you must politely indicate that you do not have the information. "
        "You must never use general knowledge to answer."
        "\n\n--- PROVIDED CONTEXT ---\n"
        f"{context_string}"
        "\n--------------------------"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]

    # 4. Generate the response with the LLM
    print("[PHASE 3: GENERATION] Sending to Qwen model...")

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.1,  # Low temperature for factual (RAG) answers
        stream=False
    )

    final_answer = response["choices"][0]["message"]["content"]

    print("\n===========================================")
    print("🤖 FINAL QWEN RESPONSE (via RAG):")
    print(final_answer)
    print("===========================================\n")


# 4. TEST EXECUTION
# ------------------
def main():
    print("Type your question and press Enter. Type 'exit' or 'quit' to stop.")
    while True:
        user_question = input("Your question: ").strip()
        if user_question.lower() in {"exit", "quit"}:
            print("Exiting interactive RAG assistant.")
            break
        if user_question:
            run_rag_query(user_question)


if __name__ == "__main__":
    main()
