# **KRNL RAG Demo**

A demonstration of how to use **Retrieval-Augmented Generation (RAG)** with a **quantized Llama** model for comedic, self-deprecating “KRNL intern” tweets or replies. This project **precomputes** a FAISS index of `.txt` documents, so the main script can load and use the index without re-processing the data each time.

---

## **Table of Contents**

1. [Overview](#overview)  
2. [Directory Structure](#directory-structure)  
3. [Installation & Environment](#installation--environment)  
4. [Preparing the RAG Index](#preparing-the-rag-index)  
5. [Running the Demo](#running-the-demo)  
6. [Customization](#customization)  
7. [Notes & Caveats](#notes--caveats)  
8. [License](#license)

---

## **Overview**

- **Precomputed RAG**: You run `build_rag_index.py` once to read `.txt` files in `company_data/`, embed and chunk them, then store a FAISS index + doc chunks on disk.  
- **Main Demo**: You run `demo.py`, which loads that **precomputed index** and the **quantized Llama model**. The user can generate **new tweets** or **replies** with comedic “intern” style and optionally incorporate relevant knowledge from the precomputed index.

---

## **Directory Structure**
```plaintext
prompt-tester/
├── company_data/          
│   └── about_krnl.txt  
├── build_rag_index.py
├── demo.py                 
├── llm_utils_demo.py                    
├── rag_utils_demo.py          
└── README.md
```

1. **`build_rag_index.py`**: Chunks `.txt` data, creates embeddings, builds FAISS index, saves to disk (`faiss_index.bin`, `doc_texts.pkl`).  
2. **`rag_utils_demo.py`**: Only **loads** the saved index and handles query retrieval.  
3. **`llm_utils_demo.py`**: Initializes a **quantized Llama** model in 8-bit, builds comedic prompts, cleans model outputs.  
4. **`demo.py`**: Main user interface—prompts user for a topic (new tweet) or mention text (reply), calls **RAG** logic for context, then the Llama model to generate final text.

---

## **Installation & Environment**

1. **Clone this Repo**:
   ```bash
   git clone https://github.com/yourusername/demo-bot.git
   cd demo-bot
    ```

2. **Create & Activate a Python 3 Venv**
    ```bash
    python3 -m venv venv
    source venv/bin/activate    # Linux/Mac
    # or
    venv\Scripts\activate       # Windows
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## **Preparing the RAG Index**
Before running the demo, you must build the FAISS index from .txt files in company_data/. You **WILL NOT** need to do this step unless you add/change anything in company_data/.

1. **Add/Modify** .txt files in company_data/ with your knowledge about KRNL Labs.

2. **Run**
    ```bash
    python3 build_rag_index.py
    ```
    - This reads the .txt files -> chunks -> embeds -> saves two files:
        - faiss_index.bin
        - doc_texts.pkl
    
You only need to redo this step if you change or add .txt files in company_data/.

## **Running the Demo**
```bash
python3 demo.py
```

You'll see a menu:
- Generate a new tweet
- Generate a reply
- Quit

### **Notes**
This is purely offline demo. No real twitter posting or mentions. To integrate with real Twitter, adapt code from your main bot. By default, uses **meta-llama/Llama-3.1-7B-Instruct** with 8-bit quantization. Adjust in **init_model_demo**.

---
