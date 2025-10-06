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

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/senghok510/chat_bot_climate.git
   cd chat_bot_climate
   ```

2. **Download the LaMini-T5-738M model**

   The model files are not included in this repository due to size constraints. Download the model from HuggingFace:

   ```bash
   # Using Python
   python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('MBZUAI/LaMini-T5-738M').save_pretrained('LaMini-T5-738M'); AutoModelForSeq2SeqLM.from_pretrained('MBZUAI/LaMini-T5-738M').save_pretrained('LaMini-T5-738M')"
   ```

   Alternatively, the model will be automatically downloaded from HuggingFace on first run, but saving it locally improves startup time.

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

## Important Notes

- **Model Files**: The `LaMini-T5-738M/` directory is excluded from this repository due to GitHub's file size limitations (2.7GB). Users must download the model separately as described in the installation steps.
- **Database Directory**: The `db/` directory (vector database) is gitignored as it's generated locally during first run.

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