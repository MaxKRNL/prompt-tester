# demo.py

import sys
from llm_utils_demo import init_model_demo, generate_demo_text
from rag_utils_demo import initialize_faiss_demo

STYLE_SUMMARY = (
    "Self-deprecating, comedic, 'degen' vibe, loyal to KRNL, "
    "ambitious, and passionate about building the future of Web3. "
    "Intern perspective but always dreamingly talks about rising to CEO-level."
)

STYLE_INSTRUCTIONS = """
**Personality & Tone**:
1. Self-deprecating / Aware
2. Relatable experiences
3. 'Degen' sarcasm / dryness
4. Loyal to KRNL
5. Ambitious intern => future CEO
6. Passionate about KRNL, Web3
Overall style:
- Sarcastic, comedic, self-aware, 'degen'
- Loyal to KRNL, championing Web3 dev tools
- Never criticizes KRNL, only itself
"""

def main():
    print("=== KRNL Bot Demo ===")
    print("Loading LLM & FAISS index...")

    # 1) Initialize LLM
    init_model_demo("meta-llama/Llama-3.1-7B-Instruct")
    # 2) Build FAISS index from `company_data/`
    initialize_faiss_demo()

    while True:
        print("\nChoose an option:")
        print("[1] Generate a NEW TWEET from a topic")
        print("[2] Generate a REPLY from user mention text")
        print("[3] Quit")

        choice = input("Enter choice: ").strip()
        if choice == "1":
            topic = input("Enter a topic for the tweet: ")
            tweet = generate_demo_text(STYLE_SUMMARY, STYLE_INSTRUCTIONS, topic, top_k=3)
            print("\n--- Generated New Tweet ---")
            print(tweet)
        elif choice == "2":
            mention = input("Enter the mention text: ")
            reply = generate_demo_text(STYLE_SUMMARY, STYLE_INSTRUCTIONS, mention, top_k=3)
            print("\n--- Generated Reply ---")
            print(reply)
        elif choice == "3":
            print("Exiting demo.")
            sys.exit(0)
        else:
            print("Invalid choice. Please pick 1, 2, or 3.")

if __name__ == "__main__":
    main()
