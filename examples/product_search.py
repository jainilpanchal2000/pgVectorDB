"""
Product Search Example — pgVectorDB v0.0.3 Multimodal Search
=============================================================

Demonstrates multi-embedding search across structured product data:
- TextSpace: product description (semantic)
- NumberSpace: price (numeric, prefer lower)
- NumberSpace: rating (numeric, prefer higher)
- CategorySpace: category (one-hot)

The key insight: no re-ranking needed when you embed all signals at index-time
and combine them at query-time with dynamic weights.

Requirements:
    pip install pgvectordb langchain-openai asyncpg asyncio
    PostgreSQL with pgvector extension enabled

Usage:
    python examples/product_search.py
"""

import asyncio
import os
from langchain_openai import OpenAIEmbeddings

# Try local import first, then installed package
try:
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from pgvectordb import pgVectorDB, TextSpace, NumberSpace, CategorySpace
    from pgvectordb.rerankers import CrossEncoderReranker
except ImportError:
    from pgvectordb import pgVectorDB, TextSpace, NumberSpace, CategorySpace
    from pgvectordb.rerankers import CrossEncoderReranker

from langchain_core.documents import Document


# ==================== Sample Data ====================

SAMPLE_PRODUCTS = [
    Document(
        page_content="Sony WH-1000XM5 Wireless Headphones with Industry-Leading Noise Canceling",
        metadata={
            "price": 349.99,
            "rating": 4.8,
            "category": "electronics",
            "brand": "Sony",
        },
    ),
    Document(
        page_content="Bose QuietComfort 45 Bluetooth Wireless Noise Cancelling Headphones",
        metadata={
            "price": 279.00,
            "rating": 4.7,
            "category": "electronics",
            "brand": "Bose",
        },
    ),
    Document(
        page_content="Apple AirPods Pro 2nd Generation with USB-C Active Noise Cancellation",
        metadata={
            "price": 229.00,
            "rating": 4.6,
            "category": "electronics",
            "brand": "Apple",
        },
    ),
    Document(
        page_content="Anker Soundcore Q45 Adaptive Active Noise Cancelling Headphones",
        metadata={
            "price": 59.99,
            "rating": 4.3,
            "category": "electronics",
            "brand": "Anker",
        },
    ),
    Document(
        page_content="Levi's Men's 511 Slim Fit Jeans — Classic 5-pocket styling",
        metadata={
            "price": 49.99,
            "rating": 4.5,
            "category": "fashion",
            "brand": "Levi's",
        },
    ),
    Document(
        page_content="Nike Air Max 270 Running Shoes — Max Air cushioning for all-day comfort",
        metadata={
            "price": 150.00,
            "rating": 4.4,
            "category": "fashion",
            "brand": "Nike",
        },
    ),
    Document(
        page_content="KitchenAid Artisan Tilt-Head Stand Mixer — 5 Qt Stainless Steel Bowl",
        metadata={
            "price": 449.99,
            "rating": 4.9,
            "category": "home",
            "brand": "KitchenAid",
        },
    ),
    Document(
        page_content="Instant Pot Duo Plus 9-in-1 Electric Pressure Cooker 6 Quart",
        metadata={
            "price": 99.99,
            "rating": 4.7,
            "category": "home",
            "brand": "Instant Pot",
        },
    ),
    Document(
        page_content="Dyson V15 Detect Absolute Cordless Vacuum — Laser dust detection",
        metadata={"price": 749.99, "rating": 4.8, "category": "home", "brand": "Dyson"},
    ),
    Document(
        page_content="Jabra Evolve2 85 Professional Wireless Headset — ANC for open offices",
        metadata={
            "price": 499.00,
            "rating": 4.6,
            "category": "electronics",
            "brand": "Jabra",
        },
    ),
]


async def run_product_search():
    """Run the full multimodal product search demo."""
    # ==================== Configuration ====================
    conn_str = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/pgvectordb_demo",
    )

    # Use OpenAI embeddings — swap for any LangChain-compatible model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ.get("OPENAI_API_KEY", "your-key-here"),
    )

    # ==================== Initialize pgVectorDB ====================
    rag = pgVectorDB(
        collection_name="products",
        embedding_model=embeddings,
        connection_string=conn_str,
    )
    await rag.initialize()
    print("✓ pgVectorDB initialized")

    # ==================== Define Vector Spaces ====================
    spaces = [
        # Text space for product descriptions — auto-detects embedding dimensions
        TextSpace(
            name="description",
            field="content",  # maps to doc.page_content
        ),
        # Price: prefer lower prices (mode="minimum")
        NumberSpace(
            name="price",
            field="price",  # maps to doc.metadata["price"]
            min_value=0.0,
            max_value=1000.0,
            mode="minimum",  # Distance increases with price
        ),
        # Rating: prefer higher ratings (mode="maximum")
        NumberSpace(
            name="rating",
            field="rating",
            min_value=0.0,
            max_value=5.0,
            mode="maximum",  # Distance decreases with rating
        ),
        # Category: one-hot encoding
        CategorySpace(
            name="category",
            field="category",
            categories=["electronics", "fashion", "home"],
        ),
    ]

    rag.register_spaces(spaces)
    print(f"✓ Registered {len(spaces)} spaces")

    # ==================== Index Documents ====================
    print(f"\nIndexing {len(SAMPLE_PRODUCTS)} products with multi-embeddings...")
    ids = await rag.add_documents_multimodal(SAMPLE_PRODUCTS, show_progress=True)
    print(f"✓ Indexed {len(ids)} products")

    # Build HNSW indexes for each space
    indexes = await rag.build_multimodal_index()
    print(f"✓ Built indexes: {list(indexes.keys())}")

    # ==================== Search Examples ====================
    print("\n" + "=" * 60)
    print("SEARCH EXAMPLE 1: Noise-cancelling headphones, budget-conscious")
    print("=" * 60)

    results = await rag.multimodal_search(
        query_params={
            "description": "noise cancelling wireless headphones",
            "price": 100.0,  # Looking for ~$100
            "rating": 4.5,  # High rating preferred
            "category": "electronics",
        },
        weights={
            "description": 0.5,  # Most important: what it is
            "price": 0.3,  # Second: affordability
            "rating": 0.15,  # Third: quality
            "category": 0.05,  # Weak category prior
        },
        k=5,
    )

    for i, r in enumerate(results, 1):
        price = r.metadata.get("price", "?")
        rating = r.metadata.get("rating", "?")
        cat = r.metadata.get("category", "?")
        print(f"{i}. [{r.score:.3f}] ${price} ⭐{rating} ({cat})")
        print(f"   {r.content[:80]}")

    print("\n" + "=" * 60)
    print("SEARCH EXAMPLE 2: Premium kitchen appliance, highly rated")
    print("=" * 60)

    results = await rag.multimodal_search(
        query_params={
            "description": "premium kitchen appliance cooking",
            "price": 500.0,  # Premium budget
            "rating": 4.8,  # Near-perfect rating
            "category": "home",
        },
        weights={
            "description": 0.4,
            "price": 0.2,
            "rating": 0.3,  # Emphasize quality
            "category": 0.1,
        },
        k=3,
    )

    for i, r in enumerate(results, 1):
        price = r.metadata.get("price", "?")
        rating = r.metadata.get("rating", "?")
        print(f"{i}. [{r.score:.3f}] ${price} ⭐{rating}")
        print(f"   {r.content[:80]}")

    print("\n" + "=" * 60)
    print("SEARCH EXAMPLE 3: Same query, different weights (text-only)")
    print("=" * 60)
    print("Weight: description=1.0 (pure semantic, ignore price/category)")

    results = await rag.multimodal_search(
        query_params={
            "description": "noise cancelling wireless headphones",
        },
        weights={"description": 1.0},
        k=5,
    )

    for i, r in enumerate(results, 1):
        price = r.metadata.get("price", "?")
        print(f"{i}. [{r.score:.3f}] ${price} — {r.content[:70]}")

    # ==================== Reranker Demo ====================
    print("\n" + "=" * 60)
    print("SEARCH EXAMPLE 4: Multimodal + CrossEncoder Reranking")
    print("=" * 60)

    try:
        reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranked = await rag.rerank_search(
            query="affordable noise cancelling headphones good value",
            reranker=reranker,
            k=10,  # Fetch 10 candidates
            rerank_top_k=3,  # Return best 3
            search_method="multimodal",
            query_params={
                "description": "affordable noise cancelling headphones good value",
                "price": 100.0,
                "category": "electronics",
            },
            weights={"description": 0.6, "price": 0.3, "category": 0.1},
        )

        for i, r in enumerate(reranked, 1):
            price = r.metadata.get("price", "?")
            print(f"{i}. [rerank={r.score:.3f}] ${price}")
            print(f"   {r.content[:80]}")
    except ImportError:
        print("⚠ sentence-transformers not installed. Skipping reranker demo.")
        print("  Install with: pip install sentence-transformers")

    print("\n✅ Product search demo complete!")


if __name__ == "__main__":
    asyncio.run(run_product_search())
