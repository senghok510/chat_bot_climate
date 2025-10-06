import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import os
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from chatbot.config_chroma import CHROMA_SETTINGS, get_chroma_client
from chatbot import ingest
from chatbot import PDFIngestion
st.set_page_config(layout="wide")

checkpoint = "LaMini-T5-738M"  
persist_directory = "db"
# /Users/senghok/Documents/Chat_bot_Climate
PATH_DOCS = os.path.join(os.path.dirname(__file__),"docs/IPCC_AR6_SYR_SPM.pdf" )
PATH_DB = os.path.join(os.path.dirname(__file__), "db")




def _load_model(cp: str):
    """Load model fully on CPU; fallback if meta tensor issue occurs."""
    try:
        return AutoModelForSeq2SeqLM.from_pretrained(
            cp,
            torch_dtype=torch.float32,
            device_map=None,  # ensure no partial meta mapping
        )
    except NotImplementedError:
        # Retry without specifying dtype (some older builds + cpu)
        return AutoModelForSeq2SeqLM.from_pretrained(cp, device_map=None)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = _load_model(checkpoint)


@st.cache_resource(show_spinner=False)
def data_ingestion():
    """Run ingestion if needed (idempotent). Returns True if store exists/created."""
    pdf = PDFIngestion(persist_directory= PATH_DB, embedding_model= "all-MiniLM-L6-v2", chunk_size = 400, chunk_overlap= 100 )
    db = pdf.ingest()
    return db is not None


@st.cache_resource(show_spinner=False)
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.7,
        top_p= 0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource(show_spinner=False)
def qa_chain():
    """Construct and cache the retrieval + QA chain."""
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    client = get_chroma_client(persist_directory)
    try:
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            client=client,
        )
    except Exception:
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS,
        )
    retriever = db.as_retriever()
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentences maximum and keep the answer concise. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)
    
    
    
    
def process_answer(question: str):
    chain = qa_chain()
    # Retrieval chain expects a dict with input key
    result = chain.invoke({"input": question})
    # create_retrieval_chain returns keys: input, context, answer
    answer = result.get("answer", "(No answer returned)")
    return answer, result
    


def main():
    st.title("Search your PDF ")
    with st.expander("About the App"):
        st.markdown(
            """This is a Generative Ai powered Question and Answering app that responds to questions about your PDF File."""
        )
    
    # Ensure vector store exists / is created
    with st.spinner("Preparing vector store (only first run)..."):
        ingested = data_ingestion()
    if not ingested:
        st.warning("No PDFs found in 'docs' directory. Add PDF files and reload.")
        return

    question = st.text_area(
        "Enter Your Question", 
        placeholder="Ask something about your PDFs...",
        key="question_input"
    )
    search_clicked = st.button("Search", key="search_button")
    if search_clicked:
        if question.strip():
            st.info(f"Your question: {question}")
            answer, metadata = process_answer(question.strip())
            st.subheader("Answer")
            st.write(answer)
            with st.expander("Debug / Raw Output"):
                st.json({k: (str(v)[:500] + '…' if isinstance(v, str) and len(v) > 500 else v) for k, v in metadata.items()})
        else:
            st.warning("Please enter a question before searching.")
    

if __name__ == "__main__":
    main()

