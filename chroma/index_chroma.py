import chromadb
import os
import re
from glob import glob
import argparse

CHROMA_PATH = "./chroma_db_data"
client = chromadb.PersistentClient(path=CHROMA_PATH)
print(f"ChromaDB client initialized, saving data to: {CHROMA_PATH}")

# Argument parsing for collection name
parser = argparse.ArgumentParser(description="Index markdown files into ChromaDB.")
parser.add_argument('--collection', type=str, default='qwen_rag_knowledge', help='Name of the ChromaDB collection')
args = parser.parse_args()

COLLECTION_NAME = args.collection
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
    all_documents = []
    for md_file in md_files:
        with open(md_file, 'r', encoding='utf-8') as f:
            md_text = f.read()
        all_documents.append(md_text)
    return all_documents

source_dir = COLLECTION_NAME
md_files = glob(os.path.join(source_dir, '*.md'))
documents_to_add = get_documents_from_markdown(source_dir)
ids_for_documents = [f"qwen_doc_{i}" for i in range(len(md_files))]

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