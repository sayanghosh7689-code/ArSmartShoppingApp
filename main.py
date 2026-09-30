


import io

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# Import modules from the backend package
from .vision import describe_image
from .rag import retrieve
from .llm import generate_overlay


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AR Smart-Shopping Assistant API"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# RESPONSE MODEL
# ============================================================

class AnalyzeResponse(BaseModel):
    item_description: str
    matched_product: dict | None
    overlay_text: str


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ============================================================
# VISION ONLY
# ============================================================

@app.post("/describe")
async def describe(
    file: UploadFile = File(...)
):
    """
    Run only the vision-language model step
    and return its raw description.
    """

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    item_description = describe_image(image)

    return {
        "item_description": item_description
    }


# ============================================================
# FULL ANALYSIS PIPELINE
# ============================================================

@app.post(
    "/analyze",
    response_model=AnalyzeResponse
)
async def analyze(
    file: UploadFile = File(...),
    user_query: str = Form("")
):
    """
    Full pipeline:

    Image
       ↓
    Vision Model
       ↓
    Product Retrieval / RAG
       ↓
    Qwen LLM
       ↓
    Streamlit
    """

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")


    # --------------------------------------------------------
    # VISION MODEL
    # --------------------------------------------------------

    item_description = describe_image(
        image
    )


    # --------------------------------------------------------
    # BUILD RETRIEVAL QUERY
    # --------------------------------------------------------

    retrieval_query = item_description

    if user_query:

        retrieval_query = (
            f"{item_description}. "
            f"{user_query}"
        )


    # --------------------------------------------------------
    # RAG / PRODUCT RETRIEVAL
    # --------------------------------------------------------

    matches = retrieve(
        retrieval_query,
        top_k=1
    )

    matched_product = (
        matches[0]
        if matches
        else None
    )


    # --------------------------------------------------------
    # QWEN / LLM OVERLAY
    # --------------------------------------------------------

    overlay_text = generate_overlay(
        item_description=item_description,
        context_products=matches,
        user_query=user_query,
    )


    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return AnalyzeResponse(
        item_description=item_description,
        matched_product=matched_product,
        overlay_text=overlay_text,
    )


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

