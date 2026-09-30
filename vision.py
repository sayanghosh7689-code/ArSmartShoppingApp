# backend/vision.py

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


# ============================================================
# MODEL CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

_processor = None
_model = None


# ============================================================
# PROMPT
# ============================================================

DESCRIBE_PROMPT = (
    "Look at this photo and identify the single most prominent item. "
    "Respond with ONE short plain-text description, maximum 20 words. "
    "Include what the item is, its category "
    "(clothing, footwear, accessory, food, or other), "
    "and any obvious color, material, or packaging detail. "
    "Do not add commentary."
)


# ============================================================
# LOAD MODEL
# ============================================================

def _load():
    """
    Load the Qwen2-VL model only once.
    The model is reused for subsequent requests.
    """

    global _processor
    global _model

    if _model is not None:
        return _processor, _model

    print("🔄 Loading Qwen2-VL model...")
    print(f"📦 Model: {MODEL_NAME}")

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    _processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        print("🚀 Using NVIDIA GPU")

        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    else:

        print("💻 Using CPU")

        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
        )

        _model = _model.to("cpu")

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    _model.eval()

    print("✅ Qwen2-VL loaded successfully")

    return _processor, _model


# ============================================================
# IMAGE DESCRIPTION
# ============================================================

def describe_image(image: Image.Image) -> str:
    """
    Analyze one camera image using Qwen2-VL.

    Returns a short description such as:

    "Black cotton T-shirt, clothing, round neck"

    This description is then sent to the RAG system.
    """

    processor, model = _load()

    # --------------------------------------------------------
    # Make sure image is RGB
    # --------------------------------------------------------

    image = image.convert("RGB")


    # --------------------------------------------------------
    # Create Qwen vision message
    # --------------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": DESCRIBE_PROMPT,
                },
            ],
        }
    ]


    # --------------------------------------------------------
    # Convert message to model prompt
    # --------------------------------------------------------

    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


    # --------------------------------------------------------
    # Process image/video information
    # --------------------------------------------------------

    image_inputs, video_inputs = process_vision_info(
        messages
    )


    # --------------------------------------------------------
    # Convert everything to tensors
    # --------------------------------------------------------

    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


    # --------------------------------------------------------
    # Move tensors to model device
    # --------------------------------------------------------

    device = next(
        model.parameters()
    ).device

    inputs = {
        key: value.to(device)
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }


    # --------------------------------------------------------
    # Generate description
    # --------------------------------------------------------

    with torch.no_grad():

        generated = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )


    # --------------------------------------------------------
    # Remove prompt tokens
    # --------------------------------------------------------

    input_ids = inputs["input_ids"]

    trimmed = [
        output_ids[len(input_ids[i]):]
        for i, output_ids in enumerate(generated)
    ]


    # --------------------------------------------------------
    # Decode response
    # --------------------------------------------------------

    description = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()


    # --------------------------------------------------------
    # Safety fallback
    # --------------------------------------------------------

    if not description:

        description = "Unknown shopping item"


    print(
        f"👁️ Vision description: {description}"
    )

    return description