import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


_tokenizer = None
_model = None


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = (
    "You are an AR shopping-overlay assistant. "
    "Given a vision model's description of what's on camera "
    "and retrieved product context, produce a SHORT overlay-style answer.\n\n"

    "Format your answer as:\n"
    "1. One-line verdict\n"
    "2. Rating\n"
    "3. Sustainability score\n"
    "4. Allergen warnings\n"
    "5. One-sentence review summary\n\n"

    "Say 'None' if there are no allergens. "
    "Keep the answer compact because it will be displayed "
    "as a camera overlay.\n\n"

    "IMPORTANT: Only use facts present in the retrieved product context. "
    "Never invent ratings, sustainability scores, allergens, prices, "
    "or product information."
)


# ============================================================
# LOAD MODEL
# ============================================================

def _load():

    global _tokenizer
    global _model

    if _model is None:

        print(
            f"Loading Qwen model: {MODEL_NAME}"
        )

        # ----------------------------------------------------
        # TOKENIZER
        # ----------------------------------------------------

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )


        # ----------------------------------------------------
        # DEVICE
        # ----------------------------------------------------

        if torch.cuda.is_available():

            print("🚀 Using GPU")

            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16
            )

            _model = _model.to("cuda")

        else:

            print("💻 Using CPU")

            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32
            )

            _model = _model.to("cpu")


        # Evaluation mode
        _model.eval()

        print("✅ Qwen model loaded successfully")


    return _tokenizer, _model


# ============================================================
# GENERATE OVERLAY
# ============================================================

def generate_overlay(
    item_description: str,
    context_products: list[dict],
    user_query: str = ""
) -> str:

    """
    Generate the final AR overlay text.

    Pipeline:

    Vision description
          ↓
    Retrieved product
          ↓
    Qwen
          ↓
    AR overlay
    """


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    tokenizer, model = _load()


    # --------------------------------------------------------
    # PRODUCT CONTEXT
    # --------------------------------------------------------

    if context_products:

        context_str = json.dumps(
            context_products,
            indent=2,
            ensure_ascii=False
        )

    else:

        context_str = "No matching product was found."


    # --------------------------------------------------------
    # USER PROMPT
    # --------------------------------------------------------

    user_prompt = (
        f"Vision model's description of the camera frame:\n"
        f"{item_description}\n\n"

        f"Retrieved product context:\n"
        f"{context_str}\n\n"
    )


    if user_query:

        user_prompt += (
            f"User question:\n"
            f"{user_query}\n\n"
        )


    user_prompt += (
        "Write the AR overlay text now."
    )


    # --------------------------------------------------------
    # CHAT MESSAGES
    # --------------------------------------------------------

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },

        {
            "role": "user",
            "content": user_prompt
        }

    ]


    # --------------------------------------------------------
    # CHAT TEMPLATE
    # --------------------------------------------------------

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    inputs = tokenizer(
        [text],
        return_tensors="pt"
    )


    # --------------------------------------------------------
    # MOVE INPUT TO MODEL DEVICE
    # --------------------------------------------------------

    device = next(
        model.parameters()
    ).device


    inputs = {k: v.to(device) for k, v in inputs.items()}


    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    with torch.no_grad():

        generated = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )


    # --------------------------------------------------------
    # REMOVE PROMPT TOKENS + DECODE
    # --------------------------------------------------------

    prompt_len = inputs["input_ids"].shape[1]

    overlay_text = tokenizer.decode(
        generated[0][prompt_len:],
        skip_special_tokens=True,
    ).strip()


    # --------------------------------------------------------
    # FALLBACK (never return None)
    # --------------------------------------------------------

    if not overlay_text:

        if context_products:

            p = context_products[0]

            allergens = p.get("allergens", [])

            overlay_text = (
                f"{p.get('name', 'Item')}\n"
                f"Rating: {p.get('rating', 'N/A')}/5\n"
                f"Sustainability: {p.get('sustainability_score', 'N/A')}/100\n"
                f"Allergens: {', '.join(map(str, allergens)) if allergens else 'None'}\n"
                f"{p.get('review_summary', '')}"
            )

        else:

            overlay_text = "No matching product found."


    print(f"💬 Overlay: {overlay_text}")

    return overlay_text