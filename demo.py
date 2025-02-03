# demo.py

import sys
from llm_utils_demo import init_model_demo, generate_demo_text
from rag_utils_demo import load_precomputed_index

STYLE_SUMMARY = (
    "Self-deprecating, comedic, 'degen' vibe, loyal to KRNL, "
    "ambitious, and passionate about building the future of Web3. "
    "Intern perspective dreaming of CEO-level."
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
    # 1) Load the LLM pipeline
    init_model_demo("meta-llama/Llama-3.1-8B-Instruct")

    # 2) Load the precomputed FAISS index
    load_precomputed_index()

    while True:
        print("\nChoose an option:")
        print("[1] Generate a new tweet from a topic")
        print("[2] Generate a reply from mention text")
        print("[3] Quit")
        choice = input("Enter choice: ").strip()

        if choice == "1":
            topic = input("Enter topic: ")
            text = generate_demo_text(
                STYLE_SUMMARY, STYLE_INSTRUCTIONS, topic, top_k=3
            )
            print("\n--- New Tweet ---\n", text)
        elif choice == "2":
            mention = input("Enter mention text: ")
            reply = generate_demo_text(
                STYLE_SUMMARY, STYLE_INSTRUCTIONS, mention, top_k=3
            )
            print("\n--- Reply ---\n", reply)
        elif choice == "3":
            print("Exiting demo.")
            sys.exit(0)
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
