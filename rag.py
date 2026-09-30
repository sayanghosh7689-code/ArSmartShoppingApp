
import json
import os

import chromadb
from chromadb.utils import embedding_functions


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "data",
        "products.json"
    )
)

CHROMA_DIR = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "chroma_store"
    )
)


# ============================================================
# CHROMA CONFIG
# ============================================================

COLLECTION_NAME = "products"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_client = None
_collection = None


# ============================================================
# PRODUCT → TEXT
# ============================================================

def _product_to_text(product: dict) -> str:
    """
    Convert a product record into a single text string
    for semantic embedding.
    """

    allergens = product.get(
        "allergens",
        []
    )

    keywords = product.get(
        "keywords",
        []
    )

    # Convert allergens safely
    if allergens:
        allergen_text = ", ".join(
            map(str, allergens)
        )
    else:
        allergen_text = "none listed"

    # Convert keywords safely
    if keywords:
        keyword_text = ", ".join(
            map(str, keywords)
        )
    else:
        keyword_text = "none"

    return (
        f"Product: {product.get('name', '')}. "
        f"Category: {product.get('category', '')}. "

        f"Rating: "
        f"{product.get('rating', 'N/A')}/5 "
        f"from {product.get('num_reviews', 0)} reviews. "

        f"Review summary: "
        f"{product.get('review_summary', '')}. "

        f"Sustainability score: "
        f"{product.get('sustainability_score', 'N/A')}/100. "

        f"Sustainability notes: "
        f"{product.get('sustainability_notes', '')}. "

        f"Allergens: "
        f"{allergen_text}. "

        f"Price range: "
        f"{product.get('price_range', 'N/A')}. "

        f"Related search keywords: "
        f"{keyword_text}."
    )


# ============================================================
# CHROMADB COLLECTION
# ============================================================

def get_collection():
    """
    Initialize ChromaDB and populate the collection
    from products.json if the collection is empty.
    """

    global _client
    global _collection

    # --------------------------------------------------------
    # Already initialized
    # --------------------------------------------------------

    if _collection is not None:
        return _collection

    print("🔄 Initializing ChromaDB...")


    # --------------------------------------------------------
    # Create persistent ChromaDB client
    # --------------------------------------------------------

    _client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )


    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    print(
        f"🔄 Loading embedding model: "
        f"{EMBED_MODEL_NAME}"
    )

    embed_fn = (
        embedding_functions
        .SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL_NAME
        )
    )


    # --------------------------------------------------------
    # Get or create collection
    # --------------------------------------------------------

    _collection = (
        _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn
        )
    )


    # --------------------------------------------------------
    # Populate collection if empty
    # --------------------------------------------------------

    if _collection.count() == 0:

        print(
            f"📂 Loading products from:\n"
            f"{DATA_PATH}"
        )


        # ----------------------------------------------------
        # Check products.json
        # ----------------------------------------------------

        if not os.path.exists(DATA_PATH):

            raise FileNotFoundError(
                f"products.json was not found:\n"
                f"{DATA_PATH}"
            )


        # ----------------------------------------------------
        # Load JSON
        # ----------------------------------------------------

        with open(
            DATA_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            products = json.load(f)


        # ----------------------------------------------------
        # Validate JSON
        # ----------------------------------------------------

        if not isinstance(products, list):

            raise ValueError(
                "products.json must contain "
                "a list of products."
            )


        if not products:

            raise ValueError(
                "products.json is empty."
            )


        # ----------------------------------------------------
        # Prepare ChromaDB records
        # ----------------------------------------------------

        ids = []
        documents = []
        metadatas = []


        for index, product in enumerate(products):

            if not isinstance(product, dict):

                continue


            # Product ID
            product_id = str(
                product.get(
                    "id",
                    f"product_{index}"
                )
            )


            # Make sure ID is unique
            if product_id in ids:

                product_id = (
                    f"{product_id}_{index}"
                )


            ids.append(product_id)


            # Embedding document
            documents.append(
                _product_to_text(product)
            )


            # Store complete product
            metadatas.append(
                {
                    "name": str(
                        product.get(
                            "name",
                            "Unknown Product"
                        )
                    ),

                    "raw": json.dumps(
                        product,
                        ensure_ascii=False
                    )
                }
            )


        # ----------------------------------------------------
        # Add products to ChromaDB
        # ----------------------------------------------------

        if ids:

            _collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

            print(
                f"✅ Added {len(ids)} products "
                f"to ChromaDB."
            )

        else:

            raise ValueError(
                "No valid products were found "
                "in products.json."
            )

    else:

        print(
            f"✅ ChromaDB already contains "
            f"{_collection.count()} products."
        )


    return _collection


# ============================================================
# RETRIEVE PRODUCTS
# ============================================================

def retrieve(
    query: str,
    top_k: int = 3
) -> list[dict]:
    """
    Retrieve the most relevant products
    using semantic similarity.
    """

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not query or not query.strip():

        return []


    # --------------------------------------------------------
    # Get collection
    # --------------------------------------------------------

    collection = get_collection()


    # --------------------------------------------------------
    # Check collection
    # --------------------------------------------------------

    total_products = collection.count()

    if total_products == 0:

        return []


    # Don't request more products than exist
    top_k = min(
        max(1, top_k),
        total_products
    )


    # --------------------------------------------------------
    # Semantic search
    # --------------------------------------------------------

    print(
        f"🔎 Searching products for: {query}"
    )

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )


    # --------------------------------------------------------
    # Extract results
    # --------------------------------------------------------

    products = []


    if (
        results
        and results.get("metadatas")
        and results["metadatas"][0]
    ):

        for metadata in results["metadatas"][0]:

            if (
                metadata
                and "raw" in metadata
            ):

                try:

                    product = json.loads(
                        metadata["raw"]
                    )

                    products.append(
                        product
                    )

                except json.JSONDecodeError:

                    print(
                        "⚠️ Could not decode "
                        "product metadata."
                    )


    print(
        f"✅ Retrieved {len(products)} product(s)"
    )

    return products

