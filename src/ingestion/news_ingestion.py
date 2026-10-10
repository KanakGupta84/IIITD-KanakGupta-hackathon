import random
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import pandas as pd
import requests
import trafilatura


# ============================================================
# CONFIGURATION
# ============================================================

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

OUTPUT_FILE = "data/news/news_sample.csv"

TARGET_ARTICLES = 10

MAX_RECORDS_PER_QUERY = 250

MAX_RETRIES = 4

# Delay between successful GDELT requests.
MIN_REQUEST_DELAY = 3
MAX_REQUEST_DELAY = 6

# Initial wait after a 429.
INITIAL_BACKOFF = 15

# Maximum number of articles for which we try to
# download the full article body in one run.
MAX_EXTRACTIONS = 100


# ============================================================
# FINANCIAL SEARCH QUERIES
# ============================================================

FINANCIAL_QUERIES = [

    (
        'sourcecountry:india '
        'sourcelang:english '
        '(RBI OR "repo rate" OR "interest rate" OR inflation)'
    ),

    (
        'sourcecountry:india '
        'sourcelang:english '
        '(banking OR banks OR "banking sector" OR "credit growth")'
    ),

    (
        'sourcecountry:india '
        'sourcelang:english '
        '(stocks OR equities OR "stock market" OR Nifty OR Sensex)'
    ),

    (
        'sourcecountry:india '
        'sourcelang:english '
        '("credit rating" OR downgrade OR default OR debt OR bonds)'
    ),

    (
        'sourcecountry:india '
        'sourcelang:english '
        '("corporate earnings" OR "quarterly results" OR revenue OR profit)'
    ),

    (
        'sourcecountry:india '
        'sourcelang:english '
        '(oil OR crude OR commodities OR "energy prices")'
    ),

    (
        'sourcecountry:india '
        'sourcelang:english '
        '(tariffs OR sanctions OR trade OR geopolitical OR geopolitics)'
    ),

    (
        'sourcecountry:india '
        'sourcelang:english '
        '("economic growth" OR GDP OR unemployment OR "economic outlook")'
    ),
]


# ============================================================
# HELPERS
# ============================================================

def current_utc_timestamp():
    """Return the current UTC timestamp in ISO format."""

    return datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def parse_gdelt_date(seendate):
    """
    Convert GDELT's date format:

        20261008T053000Z

    into:

        2026-10-08T05:30:00Z
    """

    if not seendate:
        return None

    try:
        dt = datetime.strptime(
            seendate,
            "%Y%m%dT%H%M%SZ"
        )

        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    except (ValueError, TypeError):
        return None


def get_source_from_url(url):
    """Extract a readable publisher/domain from a URL."""

    if not url:
        return None

    try:
        domain = urlparse(url).netloc.lower()

        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    except Exception:
        return None


# ============================================================
# GDELT REQUEST
# ============================================================

def fetch_gdelt_articles(
    query,
    max_records=MAX_RECORDS_PER_QUERY,
    max_retries=MAX_RETRIES
):
    """
    Fetch articles from GDELT.

    Handles:
        - HTTP 429 rate limiting
        - exponential backoff
        - HTTP errors
        - malformed JSON

    Returns:
        list of raw GDELT article dictionaries
    """

    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": max_records,
        "timespan": "30d",
        "format": "json",
        "sort": "datedesc",
    }

    for attempt in range(max_retries + 1):

        try:

            response = requests.get(
                GDELT_URL,
                params=params,
                timeout=30
            )

        except requests.RequestException as error:

            print(
                f"Request failed: {error}"
            )

            if attempt == max_retries:
                return []

            wait_time = INITIAL_BACKOFF * (2 ** attempt)

            print(
                f"Waiting {wait_time} seconds before retry..."
            )

            time.sleep(wait_time)

            continue

        print(
            f"    Attempt {attempt + 1}: "
            f"HTTP {response.status_code}"
        )

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if response.status_code == 200:

            try:

                data = response.json()

            except ValueError:

                print(
                    "    GDELT returned invalid JSON."
                )

                return []

            return data.get("articles", [])

        # ----------------------------------------------------
        # RATE LIMIT
        # ----------------------------------------------------

        if response.status_code == 429:

            if attempt == max_retries:

                print(
                    "    Maximum retries reached."
                )

                return []

            wait_time = INITIAL_BACKOFF * (2 ** attempt)

            print(
                f"    Rate limit reached. "
                f"Waiting {wait_time} seconds..."
            )

            time.sleep(wait_time)

            continue

        # ----------------------------------------------------
        # OTHER HTTP ERROR
        # ----------------------------------------------------

        print(
            f"    GDELT returned HTTP "
            f"{response.status_code}"
        )

        return []

    return []


# ============================================================
# ARTICLE NORMALIZATION
# ============================================================

def normalize_gdelt_article(article, query):
    """
    Convert a raw GDELT article into our standardized
    internal representation.
    """

    url = article.get("url")

    return {
        "article_id": None,
        "source_type": "news",
        "source": get_source_from_url(url),
        "published_at": parse_gdelt_date(
            article.get("seendate")
        ),
        "retrieved_at": current_utc_timestamp(),
        "language": article.get("language"),
        "title": article.get("title"),
        "text": None,
        "url": url,
        "query_used": query,
    }


# ============================================================
# URL DEDUPLICATION
# ============================================================

def deduplicate_articles(articles):
    """
    Remove duplicate articles using their URL.
    """

    if not articles:
        return []

    df = pd.DataFrame(articles)

    # Remove rows with no URL.
    df = df[
        df["url"].notna()
        & (df["url"].str.strip() != "")
    ]

    # Remove duplicate URLs.
    df = df.drop_duplicates(
        subset=["url"],
        keep="first"
    )

    return df.to_dict("records")


# ============================================================
# ARTICLE TEXT EXTRACTION
# ============================================================

def extract_article_text(url):
    """
    Download and extract the main article text using
    Trafilatura.

    Returns:
        str or None
    """

    if not url:
        return None

    try:

        downloaded = trafilatura.fetch_url(url)

        if not downloaded:
            return None

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            favor_precision=True
        )

        if not text:
            return None

        return text.strip()

    except Exception as error:

        print(
            f"    Extraction failed: {error}"
        )

        return None


def enrich_articles_with_text(articles):
    """
    Attempt full-text extraction for collected articles.

    Articles that cannot be extracted are retained.
    Their text field remains None.
    """

    total = min(
        len(articles),
        MAX_EXTRACTIONS
    )

    print(
        f"\nExtracting article text "
        f"for up to {total} articles..."
    )

    for index in range(total):

        article = articles[index]

        print(
            f"    [{index + 1}/{total}] "
            f"{article.get('title', '')[:80]}"
        )

        article["text"] = extract_article_text(
            article["url"]
        )

        # Small delay between publisher requests.
        time.sleep(
            random.uniform(1.5, 3.0)
        )

    return articles


# ============================================================
# ARTICLE IDs
# ============================================================

def assign_article_ids(articles):
    """Assign stable sequential article IDs."""

    for index, article in enumerate(
        articles,
        start=1
    ):

        article["article_id"] = (
            f"news_{index:04d}"
        )

    return articles


# ============================================================
# SAVE
# ============================================================

def save_articles(articles, output_file):
    """Save normalized articles to CSV."""

    if not articles:

        print(
            "No articles collected. "
            "Nothing to save."
        )

        return

    df = pd.DataFrame(articles)

    columns = [
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

    df = df[columns]

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nSaved {len(df)} articles to:"
    )

    print(output_file)

    print(
        f"Articles containing extracted text: "
        f"{df['text'].notna().sum()}"
    )


# ============================================================
# MAIN COLLECTION PIPELINE
# ============================================================

def collect_financial_news():

    print("=" * 60)
    print("FINANCIAL NEWS COLLECTION")
    print("=" * 60)

    print(
        f"Target articles: {TARGET_ARTICLES}"
    )

    print(
        f"Queries: {len(FINANCIAL_QUERIES)}"
    )

    all_articles = []

    # --------------------------------------------------------
    # QUERY LOOP
    # --------------------------------------------------------

    for query_number, query in enumerate(
        FINANCIAL_QUERIES,
        start=1
    ):

        # Stop once we have enough raw articles.
        if len(all_articles) >= TARGET_ARTICLES:
            break

        print("\n" + "-" * 60)

        print(
            f"Query {query_number}/"
            f"{len(FINANCIAL_QUERIES)}"
        )

        print(query)

        articles = fetch_gdelt_articles(
            query=query,
            max_records=MAX_RECORDS_PER_QUERY
        )

        print(
            f"    Returned: {len(articles)}"
        )

        for article in articles:

            normalized = normalize_gdelt_article(
                article,
                query
            )

            all_articles.append(
                normalized
            )

        # ----------------------------------------------------
        # DELAY BETWEEN GDELT REQUESTS
        # ----------------------------------------------------

        if query_number < len(
            FINANCIAL_QUERIES
        ):

            delay = random.uniform(
                MIN_REQUEST_DELAY,
                MAX_REQUEST_DELAY
            )

            print(
                f"    Waiting {delay:.1f} "
                f"seconds before next query..."
            )

            time.sleep(delay)

    # --------------------------------------------------------
    # DEDUPLICATION
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DEDUPLICATING")
    print("=" * 60)

    print(
        f"Before deduplication: "
        f"{len(all_articles)}"
    )

    all_articles = deduplicate_articles(
        all_articles
    )

    print(
        f"After URL deduplication: "
        f"{len(all_articles)}"
    )

    # --------------------------------------------------------
    # TARGET LIMIT
    # --------------------------------------------------------

    all_articles = all_articles[
        :TARGET_ARTICLES
    ]

    print(
        f"Articles selected: "
        f"{len(all_articles)}"
    )

    # --------------------------------------------------------
    # ARTICLE TEXT
    # --------------------------------------------------------

    all_articles = enrich_articles_with_text(
        all_articles
    )

    # --------------------------------------------------------
    # ASSIGN IDS
    # --------------------------------------------------------

    all_articles = assign_article_ids(
        all_articles
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_articles(
        all_articles,
        OUTPUT_FILE
    )

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    collect_financial_news()