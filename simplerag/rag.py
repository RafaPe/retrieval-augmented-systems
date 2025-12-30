import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from evaluation.TRUE import TRUEEvaluator


VECTOR_DB_PATH = "simplerag/data/vectordb/faiss"
PDF_FOLDER = "simplerag/data/pdfs"


def run_simple_rag(
    gen_model: str = "llama3",
    judge_model: str = "phi3:mini",
    k: int = 3,
):
    """
    Runs an interactive RAG loop with FAISS persistence and TRUE evaluation.
    """

    # --- Generation model ---
    llm = Ollama(model=gen_model, base_url="http://localhost:11434")

    # --- Embeddings ---
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Load or create vector store ---
    index_file = os.path.join(VECTOR_DB_PATH, "index.faiss")

    if os.path.exists(index_file):
        print("Loading existing FAISS vector store...")
        vectorstore = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("Creating new FAISS vector store...")
        documents = []

        for file in os.listdir(PDF_FOLDER):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
                documents.extend(loader.load())

        print(f"Loaded {len(documents)} pages")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(documents)
        print(f"Created {len(chunks)} text chunks")

        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(VECTOR_DB_PATH)
        print("Vector store saved.")

    # --- Evaluator ---
    evaluator = TRUEEvaluator(model=judge_model)

    # --- Interactive loop ---
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        docs = vectorstore.similarity_search(query, k=k)
        context = "\n\n".join(doc.page_content for doc in docs)

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

        score = evaluator.score(response, context)
        print(f"\nTRUE Score: {score:.2f}")
