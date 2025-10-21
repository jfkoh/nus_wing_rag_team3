import os
import json
import argparse
from typing import List, Dict, Any


def load_chunks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                yield json.loads(ln)


def batched(iterable, batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def get_device(requested: str) -> str:
    """Determine the best device to use for torch."""
    if requested == "auto":
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # Check for Apple Silicon GPU
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    # If user requests a specific device, honor it
    return requested

def check_hf_connection(model_name: str, cache_dir: str):
    """Check if we can connect to Hugging Face Hub and download a file."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        # This case is already handled by the try/except in ask()
        return

    try:
        # Try to download a tiny, non-essential file to test connection
        hf_hub_download(repo_id=model_name, filename="config.json", cache_dir=cache_dir)
    except Exception as e:
        print(f"---! Connection Test Failed !---")
        print(f"Failed to download a test file from Hugging Face model '{model_name}'.")
        print("This often indicates a network issue, such as a firewall, proxy, or DNS problem.")
        print("Please check your network connection and proxy settings.")
        print(f"Original error: {e}")
        # Re-raise to stop execution
        raise e


def build(args):
    # import chromadb
    # from sentence_transformers import SentenceTransformer
    import chromadb
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "Failed to import sentence-transformers / transformers. "
            "This usually means 'huggingface-hub' / 'transformers' versions are incompatible. "
            "Run: python -m pip install -U huggingface-hub transformers sentence-transformers"
        ) from e

    os.makedirs(args.persist, exist_ok=True)

    # Ensure model cache stays in workspace if provided
    model_kwargs = {}
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        model_kwargs["cache_folder"] = args.cache_dir

    device = get_device(args.device)
    print(f"Loading model: {args.model} on device: {device}")
    model = SentenceTransformer(args.model, device=device, **model_kwargs)

    print(f"Opening Chroma at: {args.persist}")
    client = chromadb.PersistentClient(path=args.persist)
    coll = client.get_or_create_collection(name=args.collection, metadata={"hnsw:space": "cosine"})

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    total = 0
    for batch in batched(load_chunks(args.chunks), args.batch):
        ids.clear(); docs.clear(); metas.clear()
        for obj in batch:
            cid = obj.get("id")
            text = obj.get("text", "")
            if not cid or not text:
                continue
            ids.append(cid)
            docs.append(text)
            metas.append({k: obj.get(k) for k in (
                "doc_id", "domain", "section", "page_start", "page_end", "chunk_idx", "source_path"
            )})

        if not ids:
            continue

        emb = model.encode(docs, show_progress_bar=True, normalize_embeddings=True)
        coll.upsert(ids=ids, documents=docs, embeddings=emb.tolist(), metadatas=metas)
        total += len(ids)
        print(f"Upserted {total} chunks...")

    print(f"Done. Total upserted: {total}")


def query(args):
    # import chromadb
    # from sentence_transformers import SentenceTransformer
    import chromadb
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "Failed to import sentence-transformers / transformers. "
            "Run: python -m pip install -U huggingface-hub transformers sentence-transformers"
        ) from e

    model_kwargs = {}
    if args.cache_dir:
        model_kwargs["cache_folder"] = args.cache_dir
    
    device = get_device(args.device)
    model = SentenceTransformer(args.model, device=device, **model_kwargs)

    client = chromadb.PersistentClient(path=args.persist)
    coll = client.get_or_create_collection(name=args.collection)

    q_emb = model.encode([args.q], normalize_embeddings=True)
    res = coll.query(query_embeddings=q_emb.tolist(), n_results=args.top_k, include=["distances", "documents", "metadatas"])

    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    for i, cid in enumerate(ids):
        dist = dists[i]
        doc = docs[i]
        meta = metas[i]
        # print(f"{i+1}. {cid}  sim={1.0 - dist:.4f}")
        # print(f"   doc={meta.get('doc_id')}  section={meta.get('section')}  domain={meta.get('domain')}")
        # print(f"   preview: {doc[:200].replace('\n',' ')}")
        preview = doc[:args.preview_len].replace("\n", " ")
        print(f"{i+1}. {cid}  sim={1.0 - dist:.4f}")
        print(f"   doc={meta.get('doc_id')}  section={meta.get('section')}  domain={meta.get('domain')}")
        print(f"   preview: {preview}")


def ask(args):
    """Perform full RAG pipeline: Retrieve from Chroma, then Generate with SLM."""
    import torch
    import chromadb
    try:
        from sentence_transformers import SentenceTransformer
        from transformers import pipeline, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "Failed to import sentence-transformers / transformers. "
            "This usually means 'huggingface-hub' / 'transformers' versions are incompatible. "
            "Run: python -m pip install -U torch huggingface-hub transformers sentence-transformers"
        ) from e

    # --- 0. Pre-flight Check ---
    check_hf_connection(args.llm, args.cache_dir)

    # --- 1. Retrieval Step (same as in query) ---
    model_kwargs = {}
    if args.cache_dir:
        model_kwargs["cache_folder"] = args.cache_dir
    
    device = get_device(args.device)
    retriever_model = SentenceTransformer(args.model, device=device, **model_kwargs)

    client = chromadb.PersistentClient(path=args.persist)
    coll = client.get_or_create_collection(name=args.collection)

    print(f"Retrieving top-{args.top_k} chunks for question: '{args.q}'")
    q_emb = retriever_model.encode([args.q], normalize_embeddings=True)
    res = coll.query(query_embeddings=q_emb.tolist(), n_results=args.top_k, include=["documents"])
    
    context_docs = res.get("documents", [[]])[0]
    if not context_docs:
        print("Could not find any relevant documents in the database.")
        return

    # --- 2. Augmentation Step ---
    # Load the tokenizer for the generative model to ensure the prompt fits
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(args.llm, cache_dir=args.cache_dir)
        # Use the model's configured max length, or a default like 1024
        model_max_length = llm_tokenizer.model_max_length or 1024
    except Exception:
        print(f"Warning: Could not load tokenizer for '{args.llm}'. Using a default max length of 1024.")
        model_max_length = 1024

    # Define a template for the prompt, leaving a placeholder for the context
    prompt_template_str = f"""
Use the following pieces of context to answer the question at the end.
If you don't know the answer from the context, just say that you don't know. Do not make up an answer.
Provide a direct answer and summarize it in a single paragraph.

Context:
---
{{context}}
---
Question: {args.q}

Helpful Answer:
"""
    # Calculate the number of tokens used by the template itself
    template_tokens = len(llm_tokenizer.encode(prompt_template_str.format(context="")))
    available_for_context = model_max_length - template_tokens - args.max_tokens

    # Join and truncate context to fit
    context_str = "\n---\n".join(context_docs)
    context_tokens = llm_tokenizer.encode(context_str, max_length=available_for_context, truncation=True)
    truncated_context = llm_tokenizer.decode(context_tokens, skip_special_tokens=True)

    prompt_template = prompt_template_str.format(context=truncated_context)

    # --- 3. Generation Step ---
    pipeline_kwargs = {
        "model": args.llm,
        "device": device,
        "model_kwargs": {"cache_dir": args.cache_dir}
    }
    
    # Add hardware-specific optimizations for a massive performance boost
    if device == "cuda":
        try:
            import bitsandbytes
            pipeline_kwargs["model_kwargs"]["load_in_4bit"] = True
            print("CUDA device detected. Enabled 4-bit quantization for the SLM.")
        except ImportError as e:
            print("---")
            print("Warning: 'bitsandbytes' is not installed, so 4-bit quantization is disabled.")
            print("         For faster inference on NVIDIA GPUs, run 'pip install -r requirements.txt'.")
            print(f"         Original error: {e}")
            print("---")
    elif device == "mps":
        pipeline_kwargs["torch_dtype"] = torch.float16
        print("MPS device detected. Using float16 for faster inference.")

    generation_args = {
        "max_new_tokens": args.max_tokens,
        "do_sample": not args.deterministic,
    }
    if args.deterministic:
        print(f"Generating answer with SLM: {args.llm} (Deterministic)")
    else:
        generation_args["temperature"] = args.temperature
        generation_args["top_p"] = args.top_p
        print(f"Generating answer with SLM: {args.llm} (Temp: {args.temperature}, Top_P: {args.top_p})")

    print("Loading text generation pipeline... (this may take a few minutes)")
    generator = pipeline("text-generation", **pipeline_kwargs)
    if generator.tokenizer.eos_token_id:
        generation_args['eos_token_id'] = generator.tokenizer.eos_token_id
    print("Pipeline loaded. Generating text...")
    
    generated_text = generator(prompt_template, **generation_args)
    print("Text generation complete.")
    print("\n--- Generated Answer ---\n")
    print(generated_text[0]['generated_text'].split("Helpful Answer:")[1].strip())


def main():
    ap = argparse.ArgumentParser(description="Ingest chunks into ChromaDB and query")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_b = sub.add_parser("build", help="Build vector store from chunks.jsonl")
    ap_b.add_argument("--chunks", default="processed/chunks.jsonl")
    ap_b.add_argument("--persist", default="chroma")
    ap_b.add_argument("--collection", default="papers")
    ap_b.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap_b.add_argument("--cache-dir", default=".cache")
    ap_b.add_argument("--device", default="auto", help="Device for sentence-transformer model (e.g., 'cpu', 'cuda', 'mps', 'auto')")
    ap_b.add_argument("--batch", type=int, default=128)

    ap_q = sub.add_parser("query", help="Query the vector store")
    ap_q.add_argument("--persist", default="chroma")
    ap_q.add_argument("--collection", default="papers")
    ap_q.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap_q.add_argument("--device", default="auto", help="Device for sentence-transformer model (e.g., 'cpu', 'cuda', 'mps', 'auto')")
    ap_q.add_argument("--cache-dir", default=".cache")
    ap_q.add_argument("--q", required=True)
    ap_q.add_argument("--top-k", type=int, default=5)
    ap_q.add_argument("--preview-len", type=int, default=200, help="Length of the preview text to display")

    ap_a = sub.add_parser("ask", help="Ask a question using the full RAG pipeline (Retrieve + Generate)")
    ap_a.add_argument("--persist", default="chroma")
    ap_a.add_argument("--collection", default="papers")
    ap_a.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Model for retrieval embeddings")
    ap_a.add_argument("--llm", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="SLM for answer generation")
    ap_a.add_argument("--device", default="auto", help="Device for models (e.g., 'cpu', 'cuda', 'mps', 'auto')")
    ap_a.add_argument("--cache-dir", default=".cache")
    ap_a.add_argument("--q", required=True)
    ap_a.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve for context")
    ap_a.add_argument("--max-tokens", type=int, default=256, help="Max new tokens for the SLM to generate")
    ap_a.add_argument("--temperature", type=float, default=0.3, help="Generation temperature (0-1)")
    ap_a.add_argument("--top-p", type=float, default=0.95, help="Top-p (nucleus) sampling (0-1)")
    ap_a.add_argument("--deterministic", action="store_true", help="Use deterministic generation (no sampling)")

    args = ap.parse_args()
    if args.cmd == "build":
        build(args)
    elif args.cmd == "query":
        query(args)
    elif args.cmd == "ask":
        ask(args)

if __name__ == "__main__":
    main()
