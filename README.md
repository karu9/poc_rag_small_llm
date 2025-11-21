
# Small llm with RAG and Function calling POC

## Step 1: initial steps

Install llama-cpp-python
[https://medium.com/@akdemir_bahadir/how-to-build-and-install-llama-cpp-python-on-apple-silicon-without-losing-your-mind-96d186f86d73](https://medium.com/@akdemir_bahadir/how-to-build-and-install-llama-cpp-python-on-apple-silicon-without-losing-your-mind-96d186f86d73)

install chromadb

```jsx
pip install chromadb pydantic
```

download a model on huggin face such as https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-GGUF

Don’t take more than 4Bytes model for testing.

## Step 2: create dataset for RAG

create a python file to index chroma, that will look at .md files in source_documents to index them:

```jsx
import chromadb
import os
import re
from glob import glob

CHROMA_PATH = "./chroma_db_data"
client = chromadb.PersistentClient(path=CHROMA_PATH)
print(f"ChromaDB client initialized, saving data to: {CHROMA_PATH}")

COLLECTION_NAME = "qwen_rag_knowledge"
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
)

print(f"Collection '{COLLECTION_NAME}' is ready.")

def extract_markdown_paragraphs(md_text):
    lines = md_text.splitlines()
    paragraphs = []
    title_stack = []
    content_buffer = []
    prev_level = 0
    in_code_block = False
    for line in lines:
        # Detect start/end of code block
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            content_buffer.append(line)  # Always include code block markers
            continue
        title_match = re.match(r'^(#+)\s+(.*)', line)
        if title_match and not in_code_block:
            # If there is buffered content, flush it as a paragraph
            if content_buffer:
                # Remove inline code from content, but keep code blocks
                clean_content = [re.sub(r'`[^`]+`', '', l) if not l.strip().startswith('```') else l for l in content_buffer]
                paragraph = ' - '.join(title_stack) + ' - ' + '\n'.join([l for l in clean_content if l.strip()])
                if paragraph.strip():
                    paragraphs.append(paragraph.strip())
                content_buffer = []
            level = len(title_match.group(1))
            title = title_match.group(2).strip()
            # Adjust title stack for current level
            title_stack = title_stack[:level-1] + [title]
            prev_level = level
        else:
            if line.strip() != '' or in_code_block:
                # Remove inline code from line, but keep code blocks
                if not in_code_block and not line.strip().startswith('```'):
                    clean_line = re.sub(r'`[^`]+`', '', line)
                else:
                    clean_line = line
                content_buffer.append(clean_line)
    # Flush last buffer
    if content_buffer and title_stack:
        clean_content = [re.sub(r'`[^`]+`', '', l) if not l.strip().startswith('```') else l for l in content_buffer]
        paragraph = ' - '.join(title_stack) + ' - ' + '\n'.join([l for l in clean_content if l.strip()])
        if paragraph.strip():
            paragraphs.append(paragraph.strip())
    return paragraphs
    
 def get_documents_from_markdown(source_dir):
    md_files = glob(os.path.join(source_dir, '*.md'))
    all_paragraphs = []
    for md_file in md_files:
        with open(md_file, 'r', encoding='utf-8') as f:
            md_text = f.read()
        paragraphs = extract_markdown_paragraphs(md_text)
        all_paragraphs.extend(paragraphs)
    return all_paragraphs

# Replace static documents_to_add with processed markdown paragraphs
documents_to_add = get_documents_from_markdown('./source_documents')
ids_for_documents = [f"qwen_doc_{i}" for i in range(len(documents_to_add))]

try:
    # Use the .add() method to insert the data
    collection.add(
        ids=ids_for_documents,
        documents=documents_to_add,
    )
    print(f"\nSuccessfully added {collection.count()} documents to the collection.")

except Exception as e:
    print(f"An error occurred during insertion: {e}")

retrieved_doc = collection.get(ids=["qwen_doc_0"])
print("\n--- Verification Sample ---")
print(f"Document ID 'qwen_doc_0' content: {retrieved_doc['documents'][0]}")
```

 

if you want to test the indexing, you can use this to query it :

```jsx
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
```

## Step 3: Test the Model

you can simply test the model with this python file: it gets the model and launches it asking recursively for prompt

```jsx
from llama_cpp import Llama

# --- MODEL PATH CONFIGURATION ---
# >>> CHANGE THIS PATH <<<
MODEL_PATH = "../models/qwen1_5-1_8b-chat-q3_k_m.gguf"
# --------------------------------

# 1. Load the LLaMA model
print(f"Loading model from: {MODEL_PATH}")
# NOTE: n_gpu_layers=-1 utilizes the Metal GPU on your Mac M4 for maximum speed.
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,  # Crucial for Mac M-series GPU acceleration
    n_ctx=4096,
    chat_format="chatml",  # Qwen uses the ChatML format
    verbose=True  # Set to True to see llama.cpp loading logs
)
print("✅ Model loaded successfully with Metal (n_gpu_layers=-1).")

# 2. Run a simple chat query
def run_simple_query(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    print(f"\n--- Running Query: {prompt} ---")

    # Use create_chat_completion for the official Qwen chat format
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=128,
        temperature=0.7,
        stream=False  # Use stream=False for simple synchronous testing
    )

    # Extract and print the model's response
    print("\nModel Response:")
    print(response["choices"][0]["message"]["content"])
    print("---------------------------------------")

def main():
    print("Type your prompt and press Enter. Type 'exit' or 'quit' to stop.")
    while True:
        prompt = input("Prompt: ").strip()
        if prompt.lower() in {"exit", "quit"}:
            print("Exiting chat loop.")
            break
        if prompt:
            run_simple_query(prompt)

if __name__ == "__main__":
    main()
```

[Exploration of RAG for llms](Small%20llm%20with%20RAG%20and%20Function%20calling%20POC/Exploration%20of%20RAG%20for%20llms%202b014774af6b8029aaa1de2946645893.md)