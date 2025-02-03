# llm_utils_demo.py

import textwrap
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
from rag_utils_demo import retrieve_context

# Global pipeline
generation_pipeline = None

def init_model_demo(model_name: str = "meta-llama/Llama-3.1-7B-Instruct"):
    """
    Initialize a quantized Llama-based model for text generation in 8-bit.
    No FAISS embedding or chunking logic is here—only LLM inference setup.
    """
    global generation_pipeline

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",  # or "cpu" if you want CPU-only
        offload_folder="offload",
        offload_state_dict=True
    )

    generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.1,
        max_new_tokens=100
    )

def remove_filler_phrases(text: str) -> str:
    filler_phrases = [
        "Let me know if this meets the requirement",
        "Hope that helps",
        "Is there anything else",
        "If you need more help",
        "Feel free to ask more",
        "Certainly, here is the tweet",
        "Sure, here's a tweet"
    ]
    for phrase in filler_phrases:
        if phrase in text:
            text = text.split(phrase)[0].strip()
    return text

def clean_model_output(text: str) -> str:
    """
    Remove repeated instruction markers, disclaimers,
    and anything after 'Total Character used:'.
    """
    # 1. Remove repeated instruction markers
    if "Now, provide ONLY the tweet:" in text:
        text = text.split("Now, provide ONLY the tweet:")[-1].strip()
    elif "Now, provide ONLY the tweet" in text:
        text = text.split("Now, provide ONLY the tweet")[-1].strip()

    # 2. Remove prompt markers
    for marker in ["Task:", "Constraints:", "You are a Twitter content generator", "Instructions:"]:
        if marker in text:
            text = text.split(marker)[0].strip()

    # 3. Truncate after "Total Character used:"
    if "Total Character used:" in text:
        text = text.split("Total Character used:")[0].strip()

    return text.strip()

def final_cleanup(text: str) -> str:
    """
    Removes leftover tokens, repeated punctuation, etc.
    Ensures a clean tweet under 280 chars.
    """
    text = re.sub(r"</s>", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[)\-:;.,]{2,}", "", text)
    return text.strip(" \"\n")

def build_demo_prompt(style_summary: str, style_instructions: str, user_topic: str, rag_context: str) -> str:
    """
    Combine comedic style instructions, RAG context, and user topic into a single prompt.
    """
    prompt = f"""
    You are a Twitter content generator.

    Your style summary:
    {style_summary}

    Additional style instructions and examples:
    {style_instructions}

    You also have the following context from our knowledge base:
    {rag_context}

    Instructions:
    - If the provided context is relevant to "{user_topic}", you may incorporate it.
    - If the context is not relevant, ignore it.

    Task:
    Write a single tweet (under 280 characters) about: "{user_topic}".
    After finishing the tweet, write "Total Character used: " followed by the number of characters in your tweet.

    Constraints:
    - Do NOT quote these instructions verbatim.
    - Do NOT use any Hashtags.
    - The tweet must be under 280 characters (including spaces).
    - Use a sarcastic, self-deprecating, but loyal-to-KRNL tone.
    - Reference crypto/Web3 or KRNL in a witty way if appropriate.
    - Keep it concise, show an "intern perspective," comedic, degen, ambitious vibes.

    Now, provide ONLY the tweet:
    """
    return textwrap.dedent(prompt).strip()

def generate_demo_text(
    style_summary: str,
    style_instructions: str,
    user_input: str,
    top_k: int = 3
) -> str:
    """
    1) Retrieves precomputed context from the FAISS index (via retrieve_context).
    2) Builds the comedic prompt with RAG + style instructions.
    3) Cleans up final text and ensures 280-char limit.
    """
    global generation_pipeline
    if generation_pipeline is None:
        raise ValueError("Model pipeline not initialized. Call init_model_demo() first.")

    # 1) Retrieve context from the precomputed FAISS index
    retrieved = retrieve_context(user_input, top_k=top_k)
    rag_text = "\n\n".join([f"Context chunk:\n{chunk}" for chunk, score in retrieved])

    # 2) Build final prompt
    prompt = build_demo_prompt(style_summary, style_instructions, user_input, rag_text)

    # 3) Generate
    outputs = generation_pipeline(prompt, num_return_sequences=1)
    raw_text = outputs[0]["generated_text"]

    # 4) Cleanup
    text = clean_model_output(raw_text)
    text = remove_filler_phrases(text)
    text = final_cleanup(text)

    # 5) Enforce 280 chars
    if len(text) > 280:
        text = text[:280].rstrip()

    return text
