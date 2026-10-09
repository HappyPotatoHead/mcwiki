from pathlib import Path

# chroma db
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
# quicker retrieval
WIKI_CACHE_PATH = DATA_DIR / "wiki_cache" / "pages.jsonl"
CHROMA_DIR = DATA_DIR / "chroma"


## WIKI FETCHING THINGS

# provided api by the wiki
# So this is not web scrapping
WIKI_API = "https://minecraft.wiki/api.php"
# user agent is needed for the wiki api
USER_AGENT = "minecraft-rag/0.1 (local RAG study project; runs offline after ingest)"

# so that I dont get kicked off
REQUEST_DELAY = 0.35
REQUEST_RETRIES = 4
ENUM_WORKERS = 4
FETCH_WORKERS = 6
# "wikitext" (50 pages/request) or "extracts" (1 page/request)
FETCH_MODE = "wikitext"
WIKI_BATCH_SIZE = 50

# Top-level categories
# Each expanded recursively
CATEGORIES = [
    "Villager mechanics",
    "Enchantments",
    "Animals",
    "Blocks",
    "Items",
    "Mobs",
    "Entities",
    "Biomes",
    "Dimensions",
    "Tools",
    "Weapons",
    "Armor",
    "Food",
    "Redstone",
    "Plants",
    "Mechanics",
    "Combat",
    "Storage",
    "Structures",
    "Utilities",
    "Trees",
    "Crafting",
]

# Top-level sections that are pure noise for QA retrieval.
DROP_SECTIONS = {
    "Videos",
    "Gallery",
    "References",
    "Navigation",
    "External links",
    "Notes",
    "See also",
    "Issues",
    "Data values",
    "Achievements",
    "Advancements",
    "History",
    "Trivia",
}

CHUNK_MAX_CHARS = 1700
CHUNK_MIN_CHARS = 200

# models meant to convert words to vectors
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# query for the embedding model to keep in mind when embedding
QUERY_INSTRUCTION = "Represent this sentence for retrieving relevant passages: "

# cross-encoder reranking
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

COLLECTION_NAME = "minecraft_wiki"
# cosine similarity
RETRIEVE_TOP_N = 50
# cross-encoder reranking
RERANK_TOP_K = 5

OLLAMA_MODEL = "llama3.1:8b"

DEVICE = "cuda"

"""
Cross-encoder reranking

two-stage information retrieval technique where a deep learning model evaluates a query and a shortlisted document together as a single input sequence to calculate an exact relevance score

"""
