from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from evaluation.TRUE import TRUEEvaluator
import os

VECTOR_DB_PATH = "simple-rag/data/vectordb/faiss"

llm = Ollama(
    model="llama3",
    base_url="http://localhost:11434"
)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if os.path.exists(VECTOR_DB_PATH):
    print("Loading existing FAISS vector store...")
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("Creating new FAISS vector store...")

    pdf_folder = "simple-rag/data/pdfs"
    documents = []

    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())

    print(f"Loaded {len(documents)} pages")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)
    print("Vector store saved.")

llm = Ollama(model="llama3")


while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
        You are an assistant that answers questions using ONLY the context below.

        Context:
        {context}

        Question:
        {query}

        Answer:
    """

    response = llm.invoke(prompt)
    print("\nAnswer:\n", response)

    evaluator = TRUEEvaluator()
    score = evaluator.score(response, context)
    print(f"\nTRUE Score: {score:.2f}")

#using ONLY the context below