"""
Real Estate NLQ (Natural Language Query) Agent — pgVectorDB v0.0.3
===================================================================

Inspired by Superlinked's Real Estate NLQ article, this example demonstrates
how to replace rigid SQL filters with flexible, weighted multimodal search.

Traditional approach (rigid):
    WHERE price < 500000 AND bedrooms >= 2 AND city = 'NYC'
    → Misses great near-matches, requires exact criteria

Multimodal approach (flexible):
    query_params={
        "description": "spacious modern apartment with city views",
        "price": 400000,       # Soft preference, not hard filter
        "bedrooms": 2,
        "city": "NYC",
    },
    weights={"description": 0.4, "price": 0.3, "bedrooms": 0.2, "city": 0.1}
    → Finds the BEST match even if no listing exactly fits all criteria

The magic: weights can be adjusted at query-time without re-indexing.

Requirements:
    pip install pgvectordb langchain-openai asyncpg asyncio
    PostgreSQL with pgvector extension enabled

Usage:
    python examples/real_estate_nlq.py
"""

import asyncio
import os

try:
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from pgvectordb import pgVectorDB, TextSpace, NumberSpace, CategorySpace
    # from pgvectordb.rerankers import CohereReranker, create_reranker
except ImportError:
    from pgvectordb import pgVectorDB, TextSpace, NumberSpace, CategorySpace

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


# ==================== Sample Real Estate Data ====================

REAL_ESTATE_LISTINGS = [
    Document(
        page_content="Luxury 2BR apartment in Manhattan Financial District with panoramic city views, "
        "modern kitchen, hardwood floors, doorman building",
        metadata={
            "price": 850000,
            "bedrooms": 2,
            "bathrooms": 2.0,
            "sqft": 1050,
            "city": "NYC",
            "neighborhood": "Financial District",
        },
    ),
    Document(
        page_content="Cozy 1BR studio in Brooklyn Park Slope, exposed brick, private backyard, "
        "steps from Prospect Park",
        metadata={
            "price": 420000,
            "bedrooms": 1,
            "bathrooms": 1.0,
            "sqft": 650,
            "city": "NYC",
            "neighborhood": "Park Slope",
        },
    ),
    Document(
        page_content="Spacious 3BR family home in Hoboken with roof deck, garage parking, "
        "Hudson River views, renovated kitchen",
        metadata={
            "price": 950000,
            "bedrooms": 3,
            "bathrooms": 2.5,
            "sqft": 1800,
            "city": "Hoboken",
            "neighborhood": "Uptown Hoboken",
        },
    ),
    Document(
        page_content="Modern 2BR condo in Jersey City Heights, floor-to-ceiling windows, "
        "Manhattan views, gym, rooftop pool",
        metadata={
            "price": 580000,
            "bedrooms": 2,
            "bathrooms": 2.0,
            "sqft": 1100,
            "city": "Jersey City",
            "neighborhood": "Heights",
        },
    ),
    Document(
        page_content="Charming pre-war 2BR in Upper West Side Manhattan, high ceilings, "
        "close to Central Park, classic doorman building",
        metadata={
            "price": 1200000,
            "bedrooms": 2,
            "bathrooms": 1.0,
            "sqft": 950,
            "city": "NYC",
            "neighborhood": "Upper West Side",
        },
    ),
    Document(
        page_content="Brand new 1BR in Astoria Queens, modern finishes, private terrace, "
        "15 min to midtown Manhattan",
        metadata={
            "price": 380000,
            "bedrooms": 1,
            "bathrooms": 1.0,
            "sqft": 720,
            "city": "Queens",
            "neighborhood": "Astoria",
        },
    ),
    Document(
        page_content="Renovated 3BR townhouse in Williamsburg Brooklyn, two-car garage, "
        "private garden, home office, walking distance to L train",
        metadata={
            "price": 1450000,
            "bedrooms": 3,
            "bathrooms": 3.0,
            "sqft": 2200,
            "city": "NYC",
            "neighborhood": "Williamsburg",
        },
    ),
    Document(
        page_content="Affordable 2BR in Bronx Mott Haven, newly renovated, stainless appliances, "
        "near BX crossings, easy 4/5 train to Manhattan",
        metadata={
            "price": 285000,
            "bedrooms": 2,
            "bathrooms": 1.0,
            "sqft": 850,
            "city": "NYC",
            "neighborhood": "Mott Haven",
        },
    ),
]


def print_listing(idx: int, r) -> None:
    """Pretty-print a single search result."""
    meta = r.metadata
    price = meta.get("price", "?")
    beds = meta.get("bedrooms", "?")
    baths = meta.get("bathrooms", "?")
    sqft = meta.get("sqft", "?")
    city = meta.get("city", "?")
    hood = meta.get("neighborhood", "?")
    print(
        f"\n  {idx}. [score={r.score:.3f}] ${price:,} | {beds}BR/{baths}ba | {sqft} sqft"
    )
    print(f"     📍 {hood}, {city}")
    print(f"     {r.content[:100]}...")


async def run_real_estate_nlq():
    """Run the real estate NLQ multimodal search demo."""
    # ==================== Setup ====================
    conn_str = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/pgvectordb_demo",
    )
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ.get("OPENAI_API_KEY", "your-key-here"),
    )

    pgvdb = pgVectorDB(
        collection_name="real_estate",
        embedding_model=embeddings,
        connection_string=conn_str,
    )
    await pgvdb.initialize()
    print("✓ pgVectorDB initialized")

    # ==================== Vector Spaces ====================
    spaces = [
        # Semantic description: what the property feels like
        TextSpace(name="description", field="content"),
        # Price: prefer lower (mode=minimum) — want under budget
        NumberSpace(
            name="price",
            field="price",
            min_value=200_000,
            max_value=2_000_000,
            mode="minimum",
        ),
        # Bedrooms: prefer closer to target (mode=similar)
        NumberSpace(
            name="bedrooms",
            field="bedrooms",
            min_value=0,
            max_value=6,
            mode="similar",
        ),
        # Bathrooms: prefer higher (mode=maximum)
        NumberSpace(
            name="bathrooms",
            field="bathrooms",
            min_value=0,
            max_value=5,
            mode="maximum",
        ),
        # City preference as categorical
        CategorySpace(
            name="city",
            field="city",
            categories=["NYC", "Hoboken", "Jersey City", "Queens", "Bronx"],
        ),
    ]

    pgvdb.register_spaces(spaces)
    print(
        f"✓ Registered {len(spaces)} spaces: description, price, bedrooms, bathrooms, city"
    )

    # ==================== Ingest Listings ====================
    print(f"\nIndexing {len(REAL_ESTATE_LISTINGS)} listings...")
    ids = await pgvdb.add_documents_multimodal(REAL_ESTATE_LISTINGS)
    await pgvdb.build_multimodal_index()
    print(f"✓ Indexed {len(ids)} listings with per-space HNSW indexes")

    # ==================== NLQ Query 1: Young professional ====================
    print("\n" + "=" * 65)
    print("NLQ QUERY 1: 'Modern 1-2BR, budget ~$450K, NYC, good transit'")
    print("=" * 65)
    print("(Weights: description=0.4, price=0.35, bedrooms=0.15, city=0.1)")

    results = await pgvdb.multimodal_search(
        query_params={
            "description": "modern apartment good transit access bright",
            "price": 450_000,
            "bedrooms": 1,
            "city": "NYC",
        },
        weights={
            "description": 0.40,
            "price": 0.35,
            "bedrooms": 0.15,
            "city": 0.10,
        },
        k=5,
    )
    for i, r in enumerate(results, 1):
        print_listing(i, r)

    # ==================== NLQ Query 2: Growing family ====================
    print("\n" + "=" * 65)
    print("NLQ QUERY 2: 'Family home 3BR, outdoor space, any city'")
    print("=" * 65)
    print("(Weights: description=0.3, price=0.2, bedrooms=0.35, bathrooms=0.15)")

    results = await pgvdb.multimodal_search(
        query_params={
            "description": "family home outdoor space garden parking",
            "price": 1_000_000,
            "bedrooms": 3,
            "bathrooms": 2,
        },
        weights={
            "description": 0.30,
            "price": 0.20,
            "bedrooms": 0.35,  # Emphasize room count
            "bathrooms": 0.15,
        },
        k=4,
    )
    for i, r in enumerate(results, 1):
        print_listing(i, r)

    # ==================== NLQ Query 3: Luxury investor ====================
    print("\n" + "=" * 65)
    print("NLQ QUERY 3: 'Luxury property with views, NYC, price no concern'")
    print("=" * 65)
    print("(Weights: description=0.7, bathrooms=0.2, city=0.1 — price ignored)")

    results = await pgvdb.multimodal_search(
        query_params={
            "description": "luxury panoramic views high-end finishes premium building amenities",
            "bathrooms": 2,
            "city": "NYC",
        },
        weights={
            "description": 0.70,  # Almost pure semantic
            "bathrooms": 0.20,
            "city": 0.10,
        },
        k=3,
    )
    for i, r in enumerate(results, 1):
        print_listing(i, r)

    # ==================== Hybrid + Reranking ====================
    print("\n" + "=" * 65)
    print("NLQ QUERY 4: Multimodal Hybrid Search (vector + BM25 keywords)")
    print("=" * 65)

    results = await pgvdb.multimodal_hybrid_search(
        query_params={
            "description": "renovated Brooklyn apartment near park",
            "price": 500_000,
            "city": "NYC",
        },
        weights={"description": 0.6, "price": 0.3, "city": 0.1},
        keyword_weight=0.25,  # 25% BM25, 75% multimodal vector
        k=4,
    )
    print("(75% multimodal vector + 25% BM25 keyword fusion)")
    for i, r in enumerate(results, 1):
        print_listing(i, r)

    print("\n✅ Real Estate NLQ demo complete!")
    print("\nKey insight: Adjust weights at query-time to reflect user intent.")
    print("No re-indexing needed. No hard filters. Best match always surfaces.")


if __name__ == "__main__":
    asyncio.run(run_real_estate_nlq())
