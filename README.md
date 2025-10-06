# Climate Chatbot - PDF Question Answering System

A Generative AI-powered question-answering application that enables users to interact with PDF documents using natural language queries. Built with LangChain, Streamlit, and HuggingFace transformers, this chatbot leverages retrieval-augmented generation (RAG) to provide accurate, context-aware answers from PDF content.

## Features

- **PDF Document Ingestion**: Automatically processes and indexes PDF files for efficient retrieval
- **Semantic Search**: Uses sentence transformers for intelligent document retrieval
- **Local LLM Processing**: Runs entirely on CPU using the LaMini-T5-738M model
- **Vector Database**: ChromaDB-powered persistent storage for document embeddings
- **Interactive Web UI**: Clean Streamlit interface for easy interaction
- **Context-Aware Responses**: Provides concise answers with source context

## Architecture

The application uses a RAG (Retrieval-Augmented Generation) pipeline:

1. **Document Processing**: PDFs are split into chunks with configurable size and overlap
2. **Embedding Generation**: Text chunks are converted to vectors using `all-MiniLM-L6-v2`
3. **Vector Storage**: Embeddings are stored in ChromaDB for fast similarity search
4. **Retrieval**: User queries are matched against the vector database
5. **Generation**: Retrieved context is fed to LaMini-T5 for answer generation

## Prerequisites

- Python 3.8+
- 8GB+ RAM recommended for model inference
- CPU-based inference (no GPU required)
- Git LFS (for storing model files) - [Install Git LFS](https://git-lfs.github.com/)

## Installation

1. **Install Git LFS** (if not already installed)
   ```bash
   # On macOS
   brew install git-lfs

   # On Ubuntu/Debian
   sudo apt-get install git-lfs

   # On Windows (using chocolatey)
   choco install git-lfs

   # Initialize Git LFS
   git lfs install
   ```

2. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Chat_bot_Climate

   # Pull LFS files (model files)
   git lfs pull
   ```

3. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Add your PDF documents**

   Place your PDF files in the `docs/` directory. The default configuration uses:
   ```
   docs/IPCC_AR6_SYR_SPM.pdf
   ```

## Usage

### Running the Web Application

Start the Streamlit app:

```bash
streamlit run chatbot_app.py
```

The application will:
- Automatically ingest PDFs on first run (creates `db/` directory)
- Download the LaMini-T5-738M model if not cached
- Launch a web interface at `http://localhost:8501`

### Using the Application

1. Wait for the initial setup (vector store creation)
2. Enter your question in the text area
3. Click "Search" to get an answer
4. View the response and optional debug information

### Example Queries

- "What are the main findings of the IPCC report?"
- "Explain the impact of climate change on sea levels"
- "What mitigation strategies are recommended?"

## Project Structure

```
Chat_bot_Climate/
├── chatbot/
│   ├── __init__.py          # Package initialization
│   ├── config_chroma.py     # ChromaDB configuration
│   └── preprocess.py        # PDF ingestion and processing
├── docs/                    # PDF documents directory
├── db/                      # Vector database (generated)
├── chatbot_app.py          # Main Streamlit application
├── cli_demo.py             # Command-line interface demo
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Configuration

Key configuration parameters in [chatbot_app.py](chatbot_app.py):

```python
checkpoint = "LaMini-T5-738M"        # Language model
persist_directory = "db"              # Vector database location
chunk_size = 400                      # Document chunk size
chunk_overlap = 100                   # Overlap between chunks
embedding_model = "all-MiniLM-L6-v2" # Sentence transformer model
```

## Dependencies

Core libraries:
- **LangChain**: RAG pipeline orchestration
- **Streamlit**: Web interface
- **Transformers**: HuggingFace model integration
- **ChromaDB**: Vector database
- **Sentence-Transformers**: Text embeddings
- **PyTorch**: Model inference backend

See [requirements.txt](requirements.txt) for complete dependency list.

## Performance Notes

- **First Run**: Model download (~300MB) and PDF ingestion may take several minutes
- **Inference Speed**: CPU-based generation takes 5-15 seconds per query
- **Memory Usage**: Approximately 2-3GB RAM for model and embeddings
- **Caching**: Streamlit caches the model and vector store for faster subsequent runs

## Git LFS Setup for Model Storage

This repository uses Git LFS to handle large model files. The [.gitattributes](.gitattributes) file is configured to track:

- Model weights (`.bin`, `.safetensors`, `.pth`, `.pt`)
- PDF documents (`.pdf`)
- Database files (`.db`, `.sqlite`, `.parquet`)

### Pushing Model Files to GitHub

If you've downloaded the LaMini-T5-738M model locally and want to push it:

```bash
# Ensure Git LFS is tracking the files
git lfs track "*.bin"
git lfs track "*.safetensors"

# Add and commit the model files
git add .gitattributes
git add LaMini-T5-738M/
git commit -m "Add LaMini-T5-738M model via Git LFS"

# Push (LFS will handle large files)
git push origin main
```

**Note**: The `db/` directory (vector database) is gitignored as it's generated locally and can be large.

## Troubleshooting

### Common Issues

**Meta tensor errors on CPU**
- The application includes fallback logic for CPU-only environments
- Ensures models load with `device_map=None`

**ChromaDB version conflicts**
- Using ChromaDB 0.3.26 for compatibility
- Falls back to legacy `client_settings` if needed

**Out of memory**
- Reduce `chunk_size` in PDFIngestion
- Lower `max_length` in the text generation pipeline
- Close other applications to free RAM

**No answers returned**
- Verify PDFs are in the `docs/` directory
- Check the `db/` directory was created successfully
- Ensure questions are relevant to the document content

## Contributing

Contributions are welcome! Areas for improvement:
- Support for additional document formats (DOCX, TXT, HTML)
- GPU acceleration for faster inference
- Multi-document source attribution
- Chat history and conversation context
- Advanced retrieval strategies (hybrid search, re-ranking)

## License

See LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Model: [LaMini-T5-738M](https://huggingface.co/MBZUAI/LaMini-T5-738M)
- Embeddings: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Default dataset: IPCC AR6 Synthesis Report

## Contact

For questions or issues, please open an issue on the repository.