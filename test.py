from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
try:
    from importlib import resources 
except ModuleNotFoundError:
    import importlib_resources as resources



def main():
    load_dotenv()
    loader = PDFMinerLoader("docs/IPCC_AR6_SYR_SPM.pdf")
    pdfs = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
    texts = text_splitter.split_documents(pdfs) # list of Documnents
    embeddings = OpenAIEmbeddings()
    # create FAISS vector from documents
    vectorstore = FAISS.from_documents(texts, embeddings)
    # Retriever VectorStore class
    retriever = vectorstore.as_retriever(search_type = "similarity")
    
    llm = ChatOpenAI()
    
    system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]    
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    query = input("Enter your query: ")
    
    answer = chain.invoke({"input": query})
    print(answer.page_content)
    

    
    
    
 

    
  

if __name__ == "__main__":
    main()

