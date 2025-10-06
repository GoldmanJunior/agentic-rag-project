import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
doc_dir=os.path.join(current_dir, "loi")
db_dir=os.path.join(current_dir, "db")
persistent_directory=os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Documents directory: {doc_dir}")
print(f"Persistent directory: {persistent_directory}")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the documents directory exists
    if not os.path.exists(doc_dir):
        raise FileNotFoundError(
            f"The directory {doc_dir} does not exist. Please check the path."
        )

    # List all PDF files in the directory
    doc_files = [f for f in os.listdir(doc_dir) if f.endswith(".pdf")]

    # Read the text content from each file and store it with metadata
    documents = []
    for doc_file in doc_files:
        file_path = os.path.join(doc_dir, doc_file)
        loader = PyPDFLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            # Add metadata to each document indicating its source
            doc.metadata = {"source": doc_file}
            documents.append(doc)

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")   # Update to a valid embedding model if needed
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it
    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating and persisting vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
