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
