
from __future__ import annotations
import os
from typing import Iterable, List, Optional
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from .config_chroma import get_chroma_client, CHROMA_SETTINGS


class PDFIngestion:
    """Encapsulates the end‑to‑end PDF -> Vector Store ingestion pipeline."""

    def __init__(
        self,
        *,
        docs_path: str = "docs",
        persist_directory: str = "db",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    client_settings=CHROMA_SETTINGS,  # kept for backward compat; not used in new client flow
        quiet: bool = False,
    ) -> None:
        self.docs_path = docs_path
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.client_settings = client_settings
        self.quiet = quiet
        self.pdf_paths: List[str] = []
        self.documents = None
        self.chunks = None
        self.db: Optional[Chroma] = None

    def find_pdf_paths(self) -> List[str]:
        pdfs: List[str] = []
        if not os.path.isdir(self.docs_path):
            return pdfs
        for root, _dirs, files in os.walk(self.docs_path):
            for name in files:
                if name.startswith('.'):
                    continue
                if name.lower().endswith('.pdf'):
                    pdfs.append(os.path.join(root, name))
        self.pdf_paths = pdfs
        return pdfs

    def load_documents(self, pdf_paths: Optional[Iterable[str]] = None):
        if pdf_paths is None:
            pdf_paths = self.pdf_paths or self.find_pdf_paths()
        pdf_list = list(pdf_paths)
        if not pdf_list:
            raise ValueError("No PDF files found to load.")
        docs = []
        for path in pdf_list:
            loader = PDFMinerLoader(path)
            docs.extend(loader.load())
        self.documents = docs
        return docs

    def documents_splitter(self, documents=None):
        if documents is None:
            if self.documents is None:
                self.load_documents()
            documents = self.documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        self.chunks = chunks
        return chunks

    def build_vector_store(self, texts) -> Chroma:
        if texts is None:
            if self.chunks is None:
                self.split_documents()
            texts = self.chunks
        embeddings = SentenceTransformerEmbeddings(model_name=self.embedding_model)
        
        try:
            client = get_chroma_client(self.persist_directory)
            db = Chroma.from_documents(
                texts,
                embeddings,
                persist_directory=self.persist_directory,
                client=client,
            )
        except Exception:
            # Fallback to legacy construction for older langchain/chromadb interplay
            db = Chroma.from_documents(
                texts,
                embeddings,
                persist_directory=self.persist_directory,
                client_settings=self.client_settings,
            )
        db.persist()
        self.db = db
        return db

    def ingest(self) -> Optional[Chroma]:
        pdfs = self.find_pdf_paths()
        if not pdfs:
            if not self.quiet:
                print("No PDF files found; skipping ingestion.")
            return None
        if not self.quiet:
            print(f"Found {len(pdfs)} PDF(s) -> loading...")
        self.load_documents(pdfs)
        if not self.quiet:
            print("Splitting into chunks...")
        self.split_documents()
        if not self.quiet:
            print("Building embeddings & vector store...")
        self.build_vector_store()
        if not self.quiet:
            print("Ingestion complete; vector store persisted.")
        return self.db










# def ingest(**kwargs) -> Optional[Chroma]:
#     return PDFIngestion(**kwargs).ingest()


# def find_pdf_paths(docs_path: str = "docs") -> List[str]:
#     return PDFIngestion(docs_path=docs_path, quiet=True).find_pdf_paths()


# def load_documents(pdf_paths: Iterable[str]):
#     return PDFIngestion(quiet=True).load_documents(pdf_paths)


# def split_documents(documents, *, chunk_size: int = 500, chunk_overlap: int = 100):
#     ing = PDFIngestion(chunk_size=chunk_size, chunk_overlap=chunk_overlap, quiet=True)
#     ing.documents = documents
#     return ing.split_documents()


# def create_vector_store(texts, **kwargs):
#     ing = PDFIngestion(**kwargs, quiet=True)
#     ing.chunks = texts
#     return ing.build_vector_store(texts)


__all__ = ["PDFIngestion", "ingest", "find_pdf_paths", "load_documents", "documents_splitter", "create_vector_store"]
