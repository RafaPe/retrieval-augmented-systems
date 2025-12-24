from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
import os

llm = Ollama(
    model="llama3",
    base_url="http://localhost:11434"
)

pdf_folder = "data/pdfs"
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


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)

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
