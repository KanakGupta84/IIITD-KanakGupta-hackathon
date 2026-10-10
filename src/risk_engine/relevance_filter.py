import re
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/news/news_sample.csv")
OUTPUT_FILE = Path("data/news/news_relevant.csv")
REVIEW_FILE = Path("data/news/news_relevance_review.csv")


# ============================================================
# FINANCIAL / RISK KEYWORDS
# ============================================================

KEYWORD_CATEGORIES = {

    # --------------------------------------------------------
    # Monetary policy and central banking
    # --------------------------------------------------------
    "monetary_policy": [
        "rbi",
        "reserve bank of india",
        "repo rate",
        "reverse repo",
        "interest rate",
        "interest rates",
        "monetary policy",
        "inflation",
        "policy rate",
        "cash reserve ratio",
        "crr",
        "slr",
    ],

    # --------------------------------------------------------
    # Financial markets
    # --------------------------------------------------------
    "markets": [
        "nifty",
        "sensex",
        "stock market",
        "stock markets",
        "equity market",
        "equity markets",
        "equities",
        "stocks",
        "shares",
        "share price",
        "market index",
        "market indices",
        "investors",
        "investor sentiment",
        "trading",
        "market volatility",
        "volatility",
    ],

    # --------------------------------------------------------
    # Banking and financial institutions
    # --------------------------------------------------------
    "banking": [
        "bank",
        "banks",
        "banking",
        "banking sector",
        "lending",
        "loan",
        "loans",
        "borrower",
        "borrowers",
        "deposit",
        "deposits",
        "nbfc",
        "nfbcs",
        "non-banking financial company",
        "non-banking financial companies",
        "financial institution",
        "financial institutions",

        # Banking risk indicators
        "funding cost",
        "funding costs",
        "margin pressure",
        "net interest margin",
        "nim",
        "asset quality",
        "provisioning",
        "capital adequacy",
        "liquidity",
        "liquidity risk",
        "credit growth",
        "loan growth",
        "bad loans",
        "npa",
        "npas",
    ],

    # --------------------------------------------------------
    # Credit and debt
    # --------------------------------------------------------
    "credit": [
        "credit",
        "credit rating",
        "credit ratings",
        "credit risk",
        "default",
        "defaults",
        "defaulted",
        "debt",
        "bond",
        "bonds",
        "downgrade",
        "downgrades",
        "bankruptcy",
        "insolvency",
        "nclt",
        "restructuring",
        "restructuring plan",
        "debt restructuring",
    ],

    # --------------------------------------------------------
    # Corporate activity
    # --------------------------------------------------------
    "corporate": [
        "earnings",
        "quarterly results",
        "quarterly result",
        "revenue",
        "revenues",
        "profit",
        "profits",
        "loss",
        "losses",
        "ipo",
        "acquisition",
        "acquisitions",
        "merger",
        "mergers",
        "m&a",
        "valuation",
        "fundraising",
        "fund raising",
        "corporate",
        "company",
        "companies",
    ],

    # --------------------------------------------------------
    # Macroeconomics
    # --------------------------------------------------------
    "macro_economy": [
        "gdp",
        "economic growth",
        "economic outlook",
        "economy",
        "economic",
        "unemployment",
        "employment",
        "recession",
        "growth outlook",
        "fiscal deficit",
        "current account",
        "industrial production",
        "manufacturing",
        "consumer spending",
        "consumer confidence",
        "business confidence",
    ],

    # --------------------------------------------------------
    # Commodities and energy
    # --------------------------------------------------------
    "commodities": [
        "oil",
        "crude oil",
        "crude",
        "energy prices",
        "commodity",
        "commodities",
        "gold",
        "natural gas",
    ],

    # --------------------------------------------------------
    # Geopolitical risk
    # --------------------------------------------------------
    "geopolitical": [
        "tariff",
        "tariffs",
        "sanctions",
        "trade war",
        "trade restrictions",
        "geopolitical",
        "geopolitics",
        "war",
        "conflict",
        "middle east",
        "russia",
        "ukraine",
        "china",
        "united states",
    ],

    # --------------------------------------------------------
    # Regulation
    # --------------------------------------------------------
    "regulation": [
        "regulation",
        "regulatory",
        "regulator",
        "regulators",
        "sebi",
        "rbi",
        "government policy",
        "policy change",
        "compliance",
        "guidelines",
        "rules",
    ],
}


# ============================================================
# STRONG FINANCIAL TERMS
# ============================================================
#
# These receive additional weight when they appear in the
# actual article title/body.
#
# IMPORTANT:
# query_used is NOT considered evidence here.
# It only tells us why GDELT retrieved the article.
# ============================================================

STRONG_FINANCIAL_TERMS = [
    "rbi",
    "repo rate",
    "interest rate",
    "inflation",
    "nifty",
    "sensex",
    "stock market",
    "equity",
    "banking",
    "bank",
    "loan",
    "credit rating",
    "default",
    "debt",
    "bond",
    "bonds",
    "earnings",
    "ipo",
    "acquisition",
    "merger",
    "valuation",
    "gdp",
    "recession",
    "tariff",
    "sanctions",
    "oil",
    "crude",
    "sebi",

    # Additional banking-risk terms
    "nbfc",
    "funding costs",
    "funding cost",
    "margin pressure",
    "net interest margin",
    "asset quality",
    "liquidity",
    "liquidity risk",
    "npa",
    "npas",
    "bad loans",
    "capital adequacy",
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Convert a value into normalized lowercase text.
    """

    if pd.isna(value):
        return ""

    value = str(value).lower()

    # Normalize whitespace.
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def contains_keyword(text, keyword):
    """
    Check whether a keyword/phrase occurs in the text.

    Word boundaries help prevent accidental substring matches.
    """

    keyword = normalize_text(keyword)

    if not keyword:
        return False

    pattern = r"\b" + re.escape(keyword) + r"\b"

    return re.search(pattern, text) is not None


# ============================================================
# CATEGORY MATCHING
# ============================================================

def find_category_matches(text):
    """
    Identify financial-risk categories represented in a piece
    of text.

    Returns:
        {
            "banking": ["bank", "loan"],
            "credit": ["credit", "debt"]
        }
    """

    matched_categories = {}

    for category, keywords in KEYWORD_CATEGORIES.items():

        matches = []

        for keyword in keywords:

            if contains_keyword(text, keyword):
                matches.append(keyword)

        if matches:
            matched_categories[category] = matches

    return matched_categories


# ============================================================
# RELEVANCE SCORING
# ============================================================

def calculate_relevance(article):
    """
    Calculate financial relevance using ONLY the actual article
    title and article body.

    GDELT's query_used field is deliberately NOT used for
    relevance scoring.

    Scoring:

        Category appearing in title  -> +3
        Category appearing in body   -> +1

        Strong financial term in title -> +3
        Strong financial term in body  -> +1

    This is intentionally transparent and deterministic.
    """

    title = normalize_text(article.get("title"))
    text = normalize_text(article.get("text"))

    # --------------------------------------------------------
    # Find categories
    # --------------------------------------------------------

    title_matches = find_category_matches(title)
    body_matches = find_category_matches(text)

    # --------------------------------------------------------
    # Category score
    # --------------------------------------------------------

    score = 0.0

    score += len(title_matches) * 3.0
    score += len(body_matches) * 1.0

    # --------------------------------------------------------
    # Strong terms
    # --------------------------------------------------------

    strong_title_terms = [
        term
        for term in STRONG_FINANCIAL_TERMS
        if contains_keyword(title, term)
    ]

    strong_body_terms = [
        term
        for term in STRONG_FINANCIAL_TERMS
        if contains_keyword(text, term)
    ]

    score += len(strong_title_terms) * 3.0
    score += len(strong_body_terms) * 1.0

    # --------------------------------------------------------
    # Combined categories
    # --------------------------------------------------------

    all_categories = sorted(
        set(title_matches)
        | set(body_matches)
    )

    # --------------------------------------------------------
    # Human-readable explanation
    # --------------------------------------------------------

    reasons = []

    if title_matches:
        reasons.append(
            "title:"
            + ",".join(sorted(title_matches.keys()))
        )

    if body_matches:
        reasons.append(
            "body:"
            + ",".join(sorted(body_matches.keys()))
        )

    if strong_title_terms:
        reasons.append(
            "strong_title:"
            + ",".join(sorted(strong_title_terms))
        )

    if strong_body_terms:
        reasons.append(
            "strong_body:"
            + ",".join(sorted(strong_body_terms))
        )

    return {
        "relevance_score": round(score, 2),
        "matched_categories": ", ".join(all_categories),
        "relevance_reason": " | ".join(reasons),
    }


# ============================================================
# RELEVANCE DECISION
# ============================================================

def classify_relevance(score, matched_categories):
    """
    Convert the numerical score into a conservative decision.

    KEEP
        Strong financial evidence.

    REVIEW
        Some financial evidence, but not enough for automatic
        inclusion.

    REJECT
        No meaningful financial evidence.
    """

    categories = [
        category.strip()
        for category in str(matched_categories).split(",")
        if category.strip()
    ]

    # No financial categories in title/body.
    if not categories:
        return "reject"

    # Strong evidence.
    if score >= 5:
        return "keep"

    # Some evidence but not enough for automatic inclusion.
    return "review"


# ============================================================
# MAIN FILTER
# ============================================================

def filter_news():

    print("=" * 70)
    print("FINANCIAL NEWS RELEVANCE FILTER")
    print("=" * 70)

    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        print("\nERROR:")
        print(f"Input file does not exist: {INPUT_FILE}")

        return

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(f"\nArticles loaded: {len(df)}")

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "article_id",
        "source_type",
        "source",
        "published_at",
        "retrieved_at",
        "language",
        "title",
        "text",
        "url",
        "query_used",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print("\nERROR:")
        print("Missing required columns:")

        for column in missing_columns:
            print(f"  - {column}")

        return

    # --------------------------------------------------------
    # Score every article
    # --------------------------------------------------------

    results = []

    for _, row in df.iterrows():

        result = calculate_relevance(row)

        relevance = classify_relevance(
            result["relevance_score"],
            result["matched_categories"],
        )

        result["relevance"] = relevance

        results.append(result)

    result_df = pd.DataFrame(results)

    df = pd.concat(
        [
            df.reset_index(drop=True),
            result_df,
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("ARTICLE RELEVANCE RESULTS")
    print("-" * 70)

    for _, row in df.iterrows():

        print(f"\n{row['article_id']}")
        print(f"Title : {row['title']}")
        print(f"Score : {row['relevance_score']}")
        print(f"Status: {row['relevance']}")

        print(
            "Categories: "
            f"{row['matched_categories'] or 'None'}"
        )

        print(
            "Reason: "
            f"{row['relevance_reason'] or 'No financial evidence'}"
        )

    # --------------------------------------------------------
    # Save review dataset
    # --------------------------------------------------------

    REVIEW_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        REVIEW_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Keep only automatically accepted articles
    # --------------------------------------------------------

    relevant_df = df[
        df["relevance"] == "keep"
    ].copy()

    # --------------------------------------------------------
    # Preserve original news schema
    # --------------------------------------------------------

    raw_columns = [
        "article_id",
        "source_type",
        "source",
        "published_at",
        "retrieved_at",
        "language",
        "title",
        "text",
        "url",
        "query_used",
    ]

    relevant_df = relevant_df[raw_columns]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    relevant_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    keep_count = len(
        df[df["relevance"] == "keep"]
    )

    review_count = len(
        df[df["relevance"] == "review"]
    )

    reject_count = len(
        df[df["relevance"] == "reject"]
    )

    print("\n" + "=" * 70)
    print("FILTER COMPLETE")
    print("=" * 70)

    print(f"Total articles : {len(df)}")
    print(f"Keep           : {keep_count}")
    print(f"Review         : {review_count}")
    print(f"Reject         : {reject_count}")

    print("\nFiltered dataset:")
    print(OUTPUT_FILE)

    print("\nReview dataset:")
    print(REVIEW_FILE)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    filter_news()