# AR Smart-Shopping / Wardrobe Assistant

A mobile-friendly camera app that points at clothing, objects, or packaged food and
overlays real-time ratings, sustainability scores, and allergen warnings — powered by
a Vision-Language Model + a RAG pipeline + Qwen2.5-1.5B-Instruct.

## Pipeline

```
Camera / Image
      ↓
   Streamlit            (frontend/app.py)
      ↓
   FastAPI              (backend/main.py)
      ↓
Vision-Language Model   (backend/vision.py — Qwen2-VL-2B-Instruct, Hugging Face)
      ↓
Product / Clothing description
      ↓
Embedding Model         (all-MiniLM-L6-v2)
      ↓
ChromaDB                (backend/rag.py)
      ↓
RAG
      ↓
Qwen LLM                (backend/llm.py — Qwen2.5-1.5B-Instruct)
      ↓
FastAPI JSON response
      ↓
Streamlit
```

The vision-language model looks directly at the camera frame and writes a short
natural-language description of the most prominent item (what it is, its category,
color/material/packaging). That description — not a fixed label from an object
detector — is what gets embedded and used to retrieve the matching product record
from ChromaDB, and it's also passed to the Qwen LLM so the final overlay text is
grounded in what the vision model actually saw.

## Project structure

```
ar-shopping-assistant/
├── backend/
│   ├── main.py         # FastAPI app — /describe and /analyze endpoints
│   ├── vision.py         # Vision-language model wrapper (Qwen2-VL)
│   ├── rag.py            # Embeddings + ChromaDB + retrieval
│   └── llm.py             # Qwen2.5-1.5B-Instruct wrapper
├── frontend/
│   └── app.py           # Streamlit camera UI
├── data/
│   └── products.json    # Sample product/attribute database
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.10+ and ideally a GPU for reasonable inference speed (both the
vision-language model and the LLM will run on CPU, just slower).

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

On first run, these auto-download (needs internet access):
- `Qwen/Qwen2-VL-2B-Instruct` (~4.5 GB) — the vision-language model that describes the frame
- `Qwen/Qwen2.5-1.5B-Instruct` (~3 GB) — the RAG-answer LLM
- `all-MiniLM-L6-v2` — the embedding model, from Hugging Face

## Running it

**Terminal 1 — start the backend:**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 — start the frontend:**
```bash
cd frontend
streamlit run app.py
```

Open the Streamlit URL it prints (usually `http://localhost:8501`), allow camera
access, take a photo of an item, and the overlay (rating, sustainability score,
allergens, review summary) will appear below the captured image, along with the
vision model's own description of what it saw.

To use on your **phone**, run both on your laptop, connect your phone to the same
Wi-Fi, and open `http://<your-laptop-IP>:8501` in your phone's browser
(update `BACKEND_URL` in `frontend/app.py` to your laptop's IP too, since
`localhost` on the phone won't reach the backend).

## Extending the product database

`data/products.json` is a stand-in for a real product catalog. Each entry needs:
- `keywords`: search keywords folded into the embedded text to help retrieval
  match the vision model's description (these no longer need to be a fixed
  label set, since there's no detector class list to match against)
- rating, review summary, sustainability score/notes, allergens, price range

Add more entries (or swap in a real product API) and delete the `chroma_store/`
folder so it re-embeds on next run.

## Swapping models

- **Smaller/faster VLM**: `Qwen2-VL-2B-Instruct` is already fairly light; for lower
  VRAM you could try a smaller captioning-only model (e.g. BLIP), trading off
  description quality and structured detail.
- **Bigger VLM**: swap `MODEL_NAME` in `vision.py` for a larger Qwen2-VL variant
  (7B, 72B) if you have the VRAM and want richer descriptions.
- **Bigger LLM**: swap `MODEL_NAME` in `llm.py` for a larger Qwen2.5 variant if you
  have the VRAM.
- **Cloud vision API instead of a local VLM**: replace `vision.py`'s internals
  with a call to Google Gemini Pro Vision or OpenAI GPT-4o — useful if you want
  to skip local GPU requirements entirely.
