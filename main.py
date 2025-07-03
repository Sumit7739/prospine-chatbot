# main_cli.py (CLI + Ollama + Qwen2.5:3b)
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from nltk.tokenize import sent_tokenize
import nltk
import datetime
import os
import requests
import sys # Import sys for exiting
import json # <--- Import json

# --- Initialization ---

print("Initializing Prospine AI Assistant (CLI)...")

# Download NLTK punkt if not already
try:
    print("Checking NLTK 'punkt' resource...")
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' not found. Downloading...")
    nltk.download("punkt")
    print("NLTK 'punkt' downloaded.")
except Exception as e:
    print(f"Error checking/downloading NLTK 'punkt': {e}")
    # Decide if you want to exit or continue without it
    # sys.exit(1) # Uncomment to exit if download fails

# Load embedding model
try:
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embedding model loaded.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    print("Please ensure the model is installed or accessible.")
    sys.exit(1)

# Load and chunk knowledge base
try:
    print("Loading and processing knowledge base (prospine_data.txt)...")
    with open("prospine_data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    sentences = sent_tokenize(text)
    chunk_size = 5
    chunks = [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

    print(f"Knowledge base split into {len(chunks)} chunks.")

    print("Generating embeddings for chunks...")
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    print("Embeddings generated.")

except FileNotFoundError:
    print("Error: prospine_data.txt not found in the current directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error processing knowledge base: {e}")
    sys.exit(1)


# Initialize Chroma vector DB
try:
    print("Initializing vector database (ChromaDB)...")
    # Using persistent storage might be better for CLI so you don't rebuild every time
    data_path = "chroma_data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    chroma_client = chromadb.PersistentClient(path=data_path, settings=Settings(anonymized_telemetry=False))

    # Using in-memory for simplicity, like the original script
    chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

    # Check if collection exists, delete if it does to rebuild, or reuse
    collection_name = "prospine"
    try:
        chroma_client.delete_collection(name=collection_name)
        print(f"Existing collection '{collection_name}' deleted.")
    except Exception: # Replace with specific ChromaDB exception if known
        print(f"No existing collection '{collection_name}' found, creating new.")
        pass # Collection doesn't exist, which is fine

    collection = chroma_client.create_collection(collection_name)

    print(f"Adding {len(chunks)} chunks to the vector database...")
    for i, chunk in enumerate(chunks):
        # Ensure embedding is a list of floats if necessary (depends on ChromaDB version)
        embedding_list = embeddings[i].tolist()
        collection.add(documents=[chunk], embeddings=[embedding_list], ids=[str(i)])

    print("Vector database initialized and populated.")

except Exception as e:
    print(f"Error initializing or populating ChromaDB: {e}")
    sys.exit(1)

# --- Core Functions ---

def get_relevant_chunks(query):
    """Retrieves relevant chunks from ChromaDB based on the query."""
    try:
        query_embedding = embedder.encode([query])[0].tolist() # Ensure list format
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        if results and results.get('documents') and results['documents'][0]:
             return results['documents'][0]
        else:
            print("Warning: No relevant documents found in vector DB.")
            return [] # Return empty list if no documents found
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return [] # Return empty list on error

def generate_answer(question, context):
    """Generates an answer using Ollama based on the question and context, streaming the output."""
    prompt = f"""
You are the official Prospine AI Assistant. 
You are a helpful, respectful, and honest AI assistant.
You are representing prospine as you are prospine official ai bot, so say 'we','us','our' in your sentences whenever you are talking about prospine,
Answer the user's question clearly, warmly, and professionally using the context provided below. If the context doesn't contain the answer, say you don't have that information in your knowledge base.

[Context]
{context}

[User Question]
{question}

[Answer]
"""
    full_response = ""
    try:
        # Use a context manager for the request
        with requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            json={
                "model": "deepseek-r1:1.5b", # Ensure this model is pulled in Ollama
                "prompt": prompt,
                "stream": True # <--- Set stream to True
            },
            timeout=60, # Add a timeout
            stream=True # <--- Enable streaming in requests
        ) as response:
            response.raise_for_status() # Check for initial HTTP errors

            # Process the stream line by line
            for line in response.iter_lines():
                if line:
                    try:
                        # Decode bytes to string and parse JSON
                        data = json.loads(line.decode('utf-8'))
                        # Extract the response part (token)
                        token = data.get("response", "")
                        # Print the token immediately without a newline, flushing the buffer
                        print(token, end='', flush=True)
                        # Append token to the full response for logging
                        full_response += token

                        # Optional: Check if generation is done (though iter_lines handles stream end)
                        # if data.get("done", False):
                        #     break
                    except json.JSONDecodeError:
                        print(f"\nWarning: Could not decode JSON line: {line}")
                    except Exception as e:
                        print(f"\nError processing stream chunk: {e}")
                        # Decide how to handle partial errors, maybe break or continue

            print() # Print a final newline after the stream is finished
            return full_response.strip() # Return the complete response for logging

    except requests.exceptions.ConnectionError:
        error_message = "⚠️ Ollama connection error: Could not connect to Ollama server at http://localhost:11434. Is Ollama running?"
        print(error_message) # Print error directly as we can't stream it
        return error_message # Return for logging
    except requests.exceptions.Timeout:
        error_message = "⚠️ Ollama error: Request timed out."
        print(error_message)
        return error_message
    except requests.exceptions.RequestException as e:
        error_message = f"⚠️ Ollama request error: {str(e)}"
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"⚠️ An unexpected error occurred during Ollama generation: {str(e)}"
        print(error_message)
        return error_message


def log_conversation(user_message, bot_answer):
    """Appends the conversation turn to the history file."""
    try:
        with open("conversation_history.txt", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\nYou: {user_message}\nBot: {bot_answer}\n\n")
    except Exception as e:
        print(f"Warning: Could not write to conversation_history.txt: {e}")

# --- Main CLI Loop ---

if __name__ == "__main__":
    print("\n--- Prospine AI Assistant Ready ---")
    print("Type your questions below. Type 'quit' or 'exit' to end.")

    try:
        while True:
            user_message = input("You: ")
            if user_message.lower() in ["quit", "exit"]:
                print("Bot: Goodbye!")
                break

            if not user_message.strip():
                # Don't print Bot: prefix if input is empty
                print("Please enter a question.")
                continue

            # 1. Retrieve relevant context
            # print("Bot: Searching knowledge base...") # Keep this informational message
            relevant_chunks = get_relevant_chunks(user_message)
            if not relevant_chunks:
                 context = "No relevant information found in the knowledge base."
                 # Optionally, you could skip generation or inform the LLM differently
                 # For now, we'll pass this info to the LLM
            else:
                context = "\n---\n".join(relevant_chunks) # Use a separator for clarity

            # 2. Generate answer (will stream output)
            print("Bot: Thinking...") # Keep this informational message
            print("Bot: ", end='', flush=True) # <--- Print prefix *before* calling generate_answer
                                               #      Use end='' and flush=True so the stream starts on the same line
            answer = generate_answer(user_message, context) # This function now prints the stream

            # 3. Log the complete answer (logging happens after streaming is done)
            #    No need to print the answer here as it was streamed by the function
            if answer and not answer.startswith("⚠️"): # Log only if generation was successful
                log_conversation(user_message, answer)

    except KeyboardInterrupt:
        print("\nBot: Exiting due to user interruption.")
    except Exception as e:
        print(f"\nAn unexpected error occurred in the main loop: {e}")
    finally:
        print("Shutting down.")
        # Optional: Clean up ChromaDB if needed, especially if using persistent storage
        # chroma_client.clear() # Or specific cleanup methods

