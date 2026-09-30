"""
app.py (Streamlit frontend)
----------------------------
Mobile-friendly camera UI. User takes a snapshot -> frame is sent to the
FastAPI backend -> the full Vision-Language-Model + RAG + Qwen pipeline runs ->
results are shown below the captured image.

Pipeline stage:  Camera / Image -> Streamlit -> FastAPI -> ... -> Streamlit
"""

import io

import requests
import streamlit as st
from PIL import Image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AR Smart-Shopping Assistant",
    page_icon="🛍️",
    layout="centered",
)


# ============================================================
# BACKEND CONFIG
# ============================================================

try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except Exception:
    BACKEND_URL = "http://127.0.0.1:8765"

# First request loads two large models, so allow plenty of time.
REQUEST_TIMEOUT = 600


# ============================================================
# HEADER
# ============================================================

st.title("🛍️ AR Smart-Shopping Assistant")

st.caption(
    "Point your camera at clothing, objects, or packaged food "
    "to get product information, ratings, sustainability scores, "
    "and allergen warnings."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System")

    st.write(f"**Backend:** `{BACKEND_URL}`")

    st.divider()

    st.subheader("📷 Camera")

    st.write(
        "If the camera does not appear, allow camera permission "
        "in your browser and refresh the page."
    )

    if st.button("🔌 Check Backend"):

        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=5)

            if response.status_code == 200:
                st.success("✅ Backend is running")
                try:
                    st.json(response.json())
                except Exception:
                    pass
            else:
                st.error(f"Backend returned status {response.status_code}")

        except requests.exceptions.RequestException:
            st.error(
                "❌ Backend is not reachable.\n\n"
                f"Make sure FastAPI is running on {BACKEND_URL}"
            )


# ============================================================
# USER QUESTION
# ============================================================

st.subheader("💬 Ask About the Product")

user_query = st.text_input(
    "Optional: Ask something specific about the item",
    placeholder="Example: Does this contain nuts?",
)


# ============================================================
# CAMERA
# ============================================================

st.subheader("📷 Scan Product")

st.info(
    "Click the camera button below and allow camera access "
    "when your browser asks for permission."
)

img_file = st.camera_input("Take a picture of the product")


# ============================================================
# PROCESS IMAGE
# ============================================================

if img_file is not None:

    # --------------------------------------------------------
    # OPEN + DISPLAY IMAGE
    # --------------------------------------------------------

    image = Image.open(img_file).convert("RGB")

    st.image(
        image,
        caption="📸 Captured Product",
        width="stretch",
    )

    # --------------------------------------------------------
    # CONVERT IMAGE TO JPEG
    # --------------------------------------------------------

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)

    # --------------------------------------------------------
    # SEND IMAGE TO FASTAPI
    # --------------------------------------------------------

    result = None

    with st.spinner(
        "🔍 Analyzing product with Vision Model + RAG + Qwen... "
        "(the first run can take a few minutes while models load)"
    ):

        try:
            response = requests.post(
                f"{BACKEND_URL}/analyze",
                files={"file": ("product.jpg", buffer, "image/jpeg")},
                data={"user_query": user_query},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            result = response.json()

        except requests.exceptions.ConnectionError:
            st.error(
                f"❌ Could not connect to the backend at {BACKEND_URL}. "
                "Is uvicorn running?"
            )

        except requests.exceptions.Timeout:
            st.error(
                "⏱️ The backend took too long to respond. "
                "Try again; models are cached after the first run."
            )

        except requests.exceptions.HTTPError:
            st.error(
                f"❌ Backend error ({response.status_code}). "
                "Check the uvicorn terminal for the traceback."
            )

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request failed: {e}")

    # --------------------------------------------------------
    # SHOW RESULTS
    # --------------------------------------------------------

    if result is not None:

        st.success("✅ Analysis completed!")

        # ---- Vision result ----

        st.subheader("👁️ Detected Item")
        st.write(result.get("item_description", "Unknown item"))

        # ---- AR overlay ----

        st.subheader("📋 AR Overlay")
        overlay_text = result.get(
            "overlay_text",
            "No additional information available.",
        )
        st.markdown(f"> {overlay_text}".replace("\n", "\n> "))

        # ---- Matched product ----

        product = result.get("matched_product")

        if product:

            st.subheader("🛒 Product Information")

            rating = product.get("rating", "N/A")
            sustainability = product.get("sustainability_score", "N/A")
            allergens = product.get("allergens", [])

            if isinstance(allergens, list):
                allergen_text = (
                    ", ".join(str(x) for x in allergens)
                    if allergens
                    else "None"
                )
            else:
                allergen_text = str(allergens)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("⭐ Rating", f"{rating} / 5")

            with col2:
                st.metric("🌱 Sustainability", f"{sustainability} / 100")

            with col3:
                st.metric("⚠️ Allergens", allergen_text)

            with st.expander("📦 Full Product Details"):
                st.json(product)

        else:
            st.warning(
                "🔎 No matching product was found in the product database."
            )


# ============================================================
# CAMERA NOT USED YET
# ============================================================

else:

    st.info("📷 Waiting for camera input...")

    st.markdown(
        """
        ### If the camera does not open

        1. Click **Allow** when Chrome asks for camera access.
        2. Click the 🔒 icon beside the website address.
        3. Open **Site settings**.
        4. Set **Camera → Allow**.
        5. Refresh the page using **Ctrl + R**.

        On Windows also check:

        **Settings → Privacy & security → Camera**

        Make sure:

        - Camera access = ON
        - Let apps access your camera = ON
        - Let desktop apps access your camera = ON
        """
    )