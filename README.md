# KRNL RAG Demo

A small demo showcasing:
- **Quantized Llama** model loading (8-bit).
- **FAISS** indexing of `company_data/` for retrieval-augmented generation (RAG).
- **Comedic, self-deprecating** "KRNL intern" persona with optional Web3 references.
- Command-line interface to either generate a **new tweet** from a topic or a **reply** from mention text.

## **Structure**
```plaintext
prompt-tester/
├── demo.py                 
├── llm_utils_demo.py                    
├── rag_utils_demo.py          
├── company_data/          
│   └── about_krnl.txt  
└── README.md
```

## **Setup**

1. **Start venv**
To avoid dependency conflict, create and activate a **Python3** virtual environment before installing dependencies:
```bash
python3 -m venv venv
```

Activate it (Linux/Mac)
```bash
source venv/bin/activate
```

Or Windows
```bash
venv\Scripts\activate
```
Once you activate the environment, all pip installations will go into your venv folder rather than your system-wide python.

2. **Clone this repo**
```bash
git clone https://github.com/MaxKRNL/prompt-tester.git
cd prompt-tester
```

3. **Install** dependencies:
    ```bash
    pip install -r requirements.txt
    ```

A GPU environment is recommended for smooth 8-bit usage. 

4. **Run**
 ```bash
python3 demo.py
```

5. The console will let you pick generating a new tweet or a reply.  

6. The code automatically **retrieves** context from the `.txt` in `company_data/` to demonstrate RAG.  

7. The **final** comedic tweet or reply is printed to your terminal.

8. **(Optional)** Adjust chunk size or embedding model in rag_utils_demo.py.

### **Notes**
This is purely offline demo. No real twitter posting or mentions. To integrate with real Twitter, adapt code from your main bot. By default, uses **meta-llama/Llama-3.1-7B-Instruct** with 8-bit quantization. Adjust in **init_model_demo**.

---
