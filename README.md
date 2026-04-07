# NexusHub

NexusHub is a FastAPI-based backend for knowledge discovery and semantic search. It integrates with the OpenAlex API to provide access to academic papers, authors, and institutions, with features for embeddings, graph analysis, and metadata crawling.

## Features

- **Semantic Search**: Vector-based search using sentence transformers for intelligent query matching
- **Graph Analysis**: Network analysis of knowledge entities and relationships
- **Metadata Crawling**: Automated collection and processing of academic metadata
- **FastAPI Backend**: RESTful API with CORS support
- **SQLite Integration**: Efficient storage with vector search capabilities

## Project Structure

- `api.py`: Main FastAPI application with REST endpoints
- `discovery.py`: Discovery and search functionality
- `embeddings.py`: Text embedding generation and processing
- `fix_embeding.py`: Embedding fixes and utilities
- `graph_analysis.py`: Graph analysis and network processing
- `knowledge.py`: Knowledge base management
- `meta_crawl.py`: Metadata crawling from external sources
- `index.html`: Web interface
- `requirements.txt`: Python dependencies
- `run`: Execution script
- `try.py`: Test or experimental code
- `.gitignore`: Git ignore rules
- **SQLite Integration**: Efficient storage with vector search capabilities

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/TheNova6000/NexusHub.git
   cd NexusHub
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv nexus_env
   nexus_env\Scripts\activate  # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the development server:
```bash
uvicorn api:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

## API Endpoints

- `GET /`: Root endpoint
- `GET /search`: Search for papers, authors, or institutions
- `GET /graph`: Graph analysis endpoints
- `POST /embeddings`: Generate embeddings for text

## Configuration

Set environment variables as needed:
- `NEXUS_DB`: Path to the main database (default: `nexushub.db`)
- `KNOWLEDGE_DB`: Path to the knowledge database (default: `nexushub.db`)
- `OPENALEX_EMAIL`: Email for OpenAlex API requests

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.