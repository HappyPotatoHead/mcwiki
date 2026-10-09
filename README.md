# Minecraft Wiki RAG

A local command-line application that answers Minecraft questions using information retrieved from the [Minecraft Wiki](https://minecraft.wiki/).

Instead of relying entirely on an LLM's existing knowledge, the application retrieves relevant wiki content and provides it as context when generating an answer.

## How It Works

The application has two main workflows: **ingestion** and **querying**.

### 1. Ingestion

Before asking questions, wiki content must be collected and indexed.

1. **Fetch:** Retrieve pages from the Minecraft Wiki through its API.
2. **Process:** Clean the extracted text and split it into smaller chunks.
3. **Embed:** Convert each chunk into a numerical vector using an embedding model.
4. **Store:** Save the vectors, text chunks and associated metadata in ChromaDB.

This creates a searchable knowledge base for subsequent queries.

### 2. Querying

When a user searches for information or asks a question:

1. **Embed the query:** Convert the question into a vector using the same embedding model.
2. **Retrieve:** Search ChromaDB for relevant chunks using cosine similarity.
3. **Rerank:** Use a cross-encoder model to score and reorder the retrieved chunks according to their relevance to the question.
4. **Generate:** Pass the question and the highest-ranked chunks to a local LLM, which generates an answer grounded in the retrieved context.

The application also supports retrieval evaluation to compare ranking performance.

## Technology Stack

- **Python** for application logic
- **ChromaDB** for vector storage and retrieval
- **Sentence Transformers** for text embeddings and reranking
- **Ollama** for local LLM inference
- **Typer** for the command-line interface

## Usage

```bash
# Ingest wiki content
./mcwiki ingest

# Search for relevant wiki passages
./mcwiki search "How do villagers reproduce?"

# Ask a question and generate an answer
./mcwiki ask "How do I brew a potion of weakness?"

# Evaluate retrieval performance
./mcwiki eval

# Check index status
./mcwiki status
```

## Evaluation

> Deeper evaluation is needed

The retrieval pipeline is evaluated using a set of test questions with expected wiki pages. Metrics include Recall@1, Recall@5 and Mean Reciprocal Rank (MRR).

In the current evaluation, cross-encoder reranking improved Recall@1 from 61.1% to 72.2% and MRR from 0.751 to 0.819, while Recall@5 remained at 94.4%.

These results measure the ranking of expected wiki pages within the retrieved candidate set. They do not, by themselves, establish the factual accuracy of generated answers.

## Future Improvements

- Improve retrieval quality using evaluation results and failure analysis.
- Evaluate generated answers for relevance and factual grounding.
- Explore alternative embedding and reranking models.
