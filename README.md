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

1. **Install** dependencies:
    ```bash
    pip install -r requirements.txt
    ```

A GPU environment is recommended for smooth 8-bit usage. 

2. Add your textual knowledge base in company_data/. 

3. **(Optional)** Adjust chunk size or embedding model in rag_utils_demo.py.

### **Notes**
This is purely offline demo. No real twitter posting or mentions. To integrate with real Twitter, adapt code from your main bot. By default, uses **meta-llama/Llama-3.1-7B-Instruct** with 8-bit quantization. Adjust in **init_model_demo**.


---

## **How to Run**

1. Place all files/folders as shown above.  
2. **Install** dependencies via: 
```bash
pip install -r requirements.txt
```
3. **Run**:
```bash
python3 demo.py
```
4. The console will let you pick generating a new tweet or a reply.  
5. The code automatically **retrieves** context from the `.txt` in `company_data/` to demonstrate RAG.  
6. The **final** comedic tweet or reply is printed to your terminal.

This meets your requirement to **keep RAG and knowledge_base** in the **demo** while using the **full prompt** and **cleanup tools**. Have fun showcasing the bot’s comedic style and **KRNL** knowledge!
