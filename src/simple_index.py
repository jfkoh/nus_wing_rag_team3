import os
import re
import json
import argparse
import hashlib
from typing import List, Tuple

import numpy as np


def tokenize(text: str) -> List[str]:
    text = text.lower()
    # Split on non-alphanumeric; keep numbers and letters
    toks = re.split(r"[^a-z0-9]+", text)
    return [t for t in toks if len(t) >= 2]


def hash_token(token: str, dim: int) -> int:
    # Stable hash via md5
    h = hashlib.md5(token.encode("utf-8")).digest()
    # Use first 8 bytes as unsigned integer
    idx = int.from_bytes(h[:8], byteorder="big", signed=False) % dim
    return idx


def embed(text: str, dim: int = 768) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in tokenize(text):
        i = hash_token(tok, dim)
        vec[i] += 1.0
    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def build_index(chunks_path: str, out_dir: str, dim: int = 768) -> None:
    os.makedirs(out_dir, exist_ok=True)
    vectors: List[np.ndarray] = []
    ids: List[str] = []
    metas = []

    n = 0
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("id")
            text = obj.get("text", "")
            if not cid or not text:
                continue
            v = embed(text, dim=dim)
            vectors.append(v)
            ids.append(cid)
            metas.append({
                "id": cid,
                "doc_id": obj.get("doc_id"),
                "section": obj.get("section"),
                "domain": obj.get("domain"),
                "source_path": obj.get("source_path"),
                "preview": text[:200].replace("\n", " ")
            })
            n += 1
            if n % 200 == 0:
                print(f"Embedded {n} chunks...")

    M = np.vstack(vectors) if vectors else np.zeros((0, dim), dtype=np.float32)
    np.save(os.path.join(out_dir, "vectors.npy"), M)
    with open(os.path.join(out_dir, "ids.txt"), "w", encoding="utf-8") as w:
        for cid in ids:
            w.write(cid + "\n")
    with open(os.path.join(out_dir, "meta.jsonl"), "w", encoding="utf-8") as w:
        for m in metas:
            w.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "info.json"), "w", encoding="utf-8") as w:
        json.dump({"dim": dim, "count": len(ids)}, w, indent=2)
    print(f"Index built: {len(ids)} vectors, dim={dim}")


def load_index(out_dir: str) -> Tuple[np.ndarray, List[str], dict]:
    M = np.load(os.path.join(out_dir, "vectors.npy"))
    with open(os.path.join(out_dir, "ids.txt"), "r", encoding="utf-8") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    meta = {}
    with open(os.path.join(out_dir, "meta.jsonl"), "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            m = json.loads(ln)
            meta[m["id"]] = m
    return M, ids, meta


def query_index(out_dir: str, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
    M, ids, _ = load_index(out_dir)
    if M.size == 0:
        return []
    q = embed(query, dim=M.shape[1])
    sims = M @ q  # cosine if M rows normalized
    top_idx = np.argsort(-sims)[:top_k]
    return [(ids[i], float(sims[i])) for i in top_idx]


def print_results(out_dir: str, results: List[Tuple[str, float]]) -> None:
    _, _, meta = load_index(out_dir)
    for rank, (cid, score) in enumerate(results, start=1):
        m = meta.get(cid, {})
        print(f"{rank}. {cid}  score={score:.4f}")
        print(f"   doc={m.get('doc_id')}  section={m.get('section')}  domain={m.get('domain')}")
        print(f"   preview: {m.get('preview','')}")


def main():
    ap = argparse.ArgumentParser(description="Simple hashed-embedding index builder/query")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_build = sub.add_parser("build", help="Build index from chunks.jsonl")
    ap_build.add_argument("--chunks", default="nus_wing_rag_team3/processed/chunks.jsonl")
    ap_build.add_argument("--out", default="nus_wing_rag_team3/vector")
    ap_build.add_argument("--dim", type=int, default=768)

    ap_query = sub.add_parser("query", help="Query the built index")
    ap_query.add_argument("--out", default="nus_wing_rag_team3/vector")
    ap_query.add_argument("--top-k", type=int, default=5)
    ap_query.add_argument("--q", required=True, help="Query text")

    args = ap.parse_args()
    if args.cmd == "build":
        build_index(args.chunks, args.out, dim=args.dim)
    elif args.cmd == "query":
        res = query_index(args.out, args.q, top_k=args.top_k)
        print_results(args.out, res)


if __name__ == "__main__":
    main()

