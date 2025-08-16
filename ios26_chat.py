# ios26_chatbot.py
# --------------------------------------------
# Retrieval-augmented chatbot for iOS 26 Q&A (Gemini + FAISS)
# --------------------------------------------
# Usage:
#   # 1) Build index (downloads & indexes sources)
#   python ios26_chatbot.py --build-index
#
#   # 2) Ask a question (CLI)
#   python ios26_chatbot.py --ask "What's new in iOS 26 beta 6?"
#
#   # 3) Serve API
#   uvicorn ios26_chatbot:app --host 0.0.0.0 --port 8000
#
# Env:
#   export GOOGLE_API_KEY="YOUR_KEY"
# --------------------------------------------

import os
import re
import json
import math
import time
import argparse
import hashlib
import datetime as dt
from typing import List, Dict, Any, Tuple
import uvicorn

import requests
from bs4 import BeautifulSoup

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import faiss

import google.generativeai as genai
from pydantic import BaseModel
from fastapi import FastAPI

# ------------- Config -------------
EMBED_MODEL = "text-embedding-004"
GEN_MODEL = "gemini-1.5-flash"
INDEX_DIR = "ios26_index"
DOCS_PATH = os.path.join(INDEX_DIR, "docs.jsonl")
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
TFIDF_PATH = os.path.join(INDEX_DIR, "tfidf.json")
CHUNK_SIZE = 1800     # ~ characters per chunk
CHUNK_OVERLAP = 300
TOP_K = 8
RECENCY_DECAY_DAYS = 45  # downweight older docs slightly

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

SEED_SOURCES = [
    # Official Apple
    "https://www.apple.com/newsroom/2025/06/apple-elevates-the-iphone-experience-with-ios-26/",
    "https://www.apple.com/os/ios/",
    "https://developer.apple.com/news/releases/",

    # Reputable coverage
    "https://www.macrumors.com/2025/08/11/ios-26-beta-6-features/",
    "https://www.macrumors.com/2025/08/14/apple-releases-ios-26-public-beta-3/",
    "https://9to5mac.com/2025/08/14/apple-releases-new-ios-26-beta-6-build-for-developers/",
    "https://9to5mac.com/2025/08/14/ios-26-ipados-26-public-beta-3/",
    "https://www.tomsguide.com/phones/iphones/ios-26-guide",
    "https://www.tomsguide.com/vehicle-tech/carplay-in-ios-26-5-biggest-upgrades-coming-to-your-car",
]

# ------------- Utils -------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def now_iso() -> str:
    return dt.datetime.utcnow().isoformat()

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def guess_date_from_page(soup: BeautifulSoup) -> str:
    # Try some common date patterns in newsroom/macrumors/9to5mac articles
    for tag in soup.find_all(["time", "meta", "span"]):
        content = " ".join([tag.get("datetime") or "", tag.get("content") or "", tag.text or ""]).strip()
        m = re.search(r"(20[12]\d[-/][01]\d[-/][0-3]\d)", content)
        if m:
            return m.group(1)
    # fallback: today
    return dt.date.today().isoformat()

def fetch_url(url: str, timeout=20) -> Tuple[str, str, str]:
    """Return (title, text, date_iso)"""
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "iOS26Bot/1.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Title
    title = soup.title.text.strip() if soup.title else url

    # Main text (very simple heuristic; good enough for curated sources)
    for selector in ["article", "main", "section", "div#content", "div.article", "div#main"]:
        node = soup.select_one(selector)
        if node:
            text = clean_text(node.get_text(" "))
            if len(text) > 500:
                return title, text, guess_date_from_page(soup)

    # fallback: whole page text
    text = clean_text(soup.get_text(" "))
    return title, text, guess_date_from_page(soup)

def chunk_text(s: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    chunks = []
    i = 0
    while i < len(s):
        chunk = s[i:i+size]
        chunks.append(chunk)
        if i + size >= len(s):
            break
        i += size - overlap
    return chunks

# def softmax(x: np.ndarray) -> np.ndarray:
#     e = np.exp(x - np.max(x))
#     return e / e.sum()

def time_decay_weight(date_iso: str, today: dt.date = None) -> float:
    today = today or dt.date.today()
    try:
        d = dt.date.fromisoformat(date_iso[:10])
        days = (today - d).days
        if days <= 0:
            return 1.0
        # e.g., halve every RECENCY_DECAY_DAYS
        return math.pow(0.5, days / RECENCY_DECAY_DAYS)
    except Exception:
        return 1.0

# ------------- Embedding / Index -------------
def configure_gemini():
    api_key = ''
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY env var.")
    genai.configure(api_key=api_key)

def embed_texts(texts: List[str]) -> np.ndarray:
    configure_gemini()
    # Batch for speed
    embs = []
    model = genai.embed_content  # wrapper call
    for i in range(0, len(texts), 32):
        batch = texts[i:i+32]
        res = genai.embed_content(model=EMBED_MODEL, content=batch)
        # res is {"embedding": [...]} if single, but for list we use embed_content for each item.
        # We'll map manually:
        if isinstance(batch, list):
            # The SDK returns a dict per call; iterate items
            # To ensure consistent batching, call per item:
            embs.extend([genai.embed_content(model=EMBED_MODEL, content=t)["embedding"] for t in batch])
        else:
            embs.append(res["embedding"])
    return np.array(embs, dtype="float32")

def build_index():
    ensure_dir(INDEX_DIR)
    docs = []
    print(f"[{now_iso()}] Fetching sources...")
    for url in SEED_SOURCES:
        try:
            title, text, date_iso = fetch_url(url)
            if len(text) < 600:
                print(f"  Skipping (too short): {url}")
                continue
            for c in chunk_text(text):
                doc = {
                    "id": hashlib.md5((url + c[:64]).encode()).hexdigest(),
                    "url": url,
                    "title": title,
                    "date": date_iso,
                    "text": c,
                    "source": re.sub(r"^https?://(www\.)?", "", url).split("/")[0],
                }
                docs.append(doc)
            print(f"  OK: {url} => {title} ({date_iso})")
        except Exception as e:
            print(f"  ERROR: {url} => {e}")

    if not docs:
        raise RuntimeError("No documents fetched. Check connectivity or sources.")

    # Save docs
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Embeddings (dense)
    print(f"[{now_iso()}] Embedding {len(docs)} chunks...")
    dense_embs = embed_texts([d["text"] for d in docs])
    # Normalize for cosine sim
    norms = np.linalg.norm(dense_embs, axis=1, keepdims=True) + 1e-9
    dense_embs = dense_embs / norms

    # FAISS index (cosine => IndexFlatIP)
    dim = dense_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(dense_embs)
    faiss.write_index(index, FAISS_PATH)

    # TF-IDF (sparse) for hybrid
    print(f"[{now_iso()}] Building TF-IDF...")
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    tfidf_mat = tfidf.fit_transform([d["text"] for d in docs])

    tfidf_dump = {
        "vocab": tfidf.vocabulary_,
        "idf": tfidf.idf_.tolist(),
        "docs": [d["id"] for d in docs],
    }
    with open(TFIDF_PATH, "w", encoding="utf-8") as f:
        json.dump(tfidf_dump, f, cls=NumpyEncoder)

    # Save a parallel npz for TF-IDF matrix (to avoid heavy JSON)
    np.savez_compressed(os.path.join(INDEX_DIR, "tfidf_mat.npz"), data=tfidf_mat.data, indices=tfidf_mat.indices, indptr=tfidf_mat.indptr, shape=tfidf_mat.shape)

    print(f"[{now_iso()}] Index built. Docs: {len(docs)}")

def load_index():
    # Load docs
    docs = []
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    # Load FAISS
    index = faiss.read_index(FAISS_PATH)

    # Load TF-IDF
    with open(TFIDF_PATH, "r", encoding="utf-8") as f:
        tfidf_dump = json.load(f)
    npz = np.load(os.path.join(INDEX_DIR, "tfidf_mat.npz"))
    from scipy.sparse import csr_matrix
    tfidf_mat = csr_matrix((npz["data"], npz["indices"], npz["indptr"]), shape=tuple(npz["shape"]))

    # Rebuild vectorizer
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    tfidf.vocabulary_ = tfidf_dump["vocab"]
    tfidf.idf_ = np.array(tfidf_dump["idf"])
    tfidf._tfidf._idf_diag = None  # will be computed lazily

    return docs, index, tfidf, tfidf_mat

def hybrid_search(query: str, docs, index, tfidf, tfidf_mat, top_k=TOP_K):
    # Dense
    q_emb = embed_texts([query])[0].reshape(1, -1)
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    sims, idxs = index.search(q_emb.astype("float32"), top_k*4)  # overfetch
    dense_scores = sims[0]
    dense_idxs = idxs[0]

    # Sparse
    q_vec = tfidf.transform([query])
    sparse_scores = (q_vec @ tfidf_mat.T).toarray()[0]

    # Combine (normalize to 0..1)
    d_max = (dense_scores.max() or 1.0)
    d_norm = dense_scores / d_max
    s_max = (sparse_scores.max() or 1.0)
    s_norm = sparse_scores / s_max

    # Build candidate set
    cand = {}
    for rank, di in enumerate(dense_idxs):
        cand[di] = 0.6 * d_norm[rank]  # weight dense higher
    # add sparse
    for di, sc in enumerate(s_norm):
        if sc > 0:
            cand[di] = cand.get(di, 0.0) + 0.4 * sc

    # Apply time decay using doc date
    today = dt.date.today()
    for di in list(cand.keys()):
        doc = docs[di]
        cand[di] *= time_decay_weight(doc.get("date", ""), today)

    # Rank
    ranked = sorted(cand.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = [{"doc": docs[di], "score": float(score)} for di, score in ranked]
    return results

# ------------- Generation -------------
def generate_answer(question: str, contexts: List[Dict[str, Any]]) -> str:
    configure_gemini()
    model = genai.GenerativeModel(GEN_MODEL)

    # Build context block with explicit sources
    ctx_blocks = []
    for i, c in enumerate(contexts, 1):
        doc = c["doc"]
        ctx_blocks.append(
            f"[{i}] Title: {doc['title']}\nURL: {doc['url']}\nDate: {doc.get('date','')}\nSource: {doc.get('source','')}\n---\n{doc['text']}\n"
        )
    ctx = "\n\n".join(ctx_blocks)

    system_rules = (
        "You are a factual iOS 26 assistant. Answer ONLY using the provided context.\n"
        "- If the answer is not in context, say you don't have that info yet.\n"
        "- Clearly label **Official (Apple)** vs **Third-party reporting** when relevant.\n"
        "- Prefer the most recent sources.\n"
        "- Provide a brief answer (3–8 bullets or 1–2 short paragraphs) and then list sources as [1], [2]...\n"
        "- Do not invent dates or features. If a feature is in beta, say so.\n"
    )

    prompt = (
        f"{system_rules}\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Answer:"
    )

    resp = model.generate_content(prompt)
    return resp.text.strip()

# ------------- CLI / API -------------
def ask_question(q: str) -> Tuple[str, List[Dict[str,Any]]]:
    docs, index, tfidf, tfidf_mat = load_index()
    hits = hybrid_search(q, docs, index, tfidf, tfidf_mat, TOP_K)
    answer = generate_answer(q, hits)
    return answer, hits

# FastAPI
app = FastAPI(title="iOS 26 Q&A Bot (RAG + Gemini)")

if not os.path.exists(INDEX_DIR):
    print("[INFO] Index not found. Building index from scratch...")
    build_index()

class AskBody(BaseModel):
    question: str

@app.post("/ask")
def ask_api(body: AskBody):

    answer, hits = ask_question(body.question)
    # Return sources explicitly for UI
    sources = [
        {"rank": i+1, "title": h["doc"]["title"], "url": h["doc"]["url"], "date": h["doc"].get("date",""), "score": h["score"]}
        for i, h in enumerate(hits)
    ]
    return {"answer": answer, "sources": sources}

# ------------- Entrypoint -------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # p = argparse.ArgumentParser()
    # p.add_argument("--build_index", action="store_true", help="Fetch sources and build FAISS + TF-IDF index")
    # p.add_argument("--ask", type=str, default=None, help="Ask a single question on the CLI")
    # args = p.parse_args()

    # if args.build_index:
    #     build_index()
    # elif args.ask:
    #     ans, srcs = ask_question(args.ask)
    #     print("\n=== Answer ===\n")
    #     print(ans)
    #     print("\n=== Top Sources ===")
    #     for i, h in enumerate(srcs, 1):
    #         d = h["doc"]
    #         print(f"[{i}] {d['title']}  ({d.get('date','')})  {d['url']}")
    # else:
    #     print("Nothing to do. Use --build-index or --ask \"...\"")
