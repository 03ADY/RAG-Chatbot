# backend/main.py

# Import necessary libraries
import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import shutil # For saving uploaded files

# LangChain components
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader, CSVLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings # Option for local embeddings (if no Google API Key)
from langchain_community.vectorstores import Chroma # Our vector database
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings # For Gemini LLM and Embeddings
from langchain.chains import ConversationalRetrievalChain # The core RAG chain
from langchain.memory import ConversationBufferMemory # For managing chat history
from langchain.text_splitter import RecursiveCharacterTextSplitter # For splitting documents

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file.")
    # In a real production app, you might exit or raise a more critical error here.
    # For demo, we'll allow it to proceed but LLM/Embeddings will fail.

DATA_DIR = "../data"
CHROMA_DB_DIR = "./chroma_db_persistent"

# --- FastAPI App Initialization ---
app = FastAPI(
    title="LangChain RAG Chatbot Backend",
    description="A FastAPI backend for a RAG chatbot using LangChain, Gemini Pro, and ChromaDB.",
    version="1.0.0",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://localhost:5500", # VS Code Live Server
    "http://127.00.1:5500",
    # Add your frontend's deployed URL here (e.g., "https://your-frontend-domain.com")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LangChain Components Initialization ---
llm: Optional[ChatGoogleGenerativeAI] = None
embeddings_model: Optional[GoogleGenerativeAIEmbeddings] = None
vectorstore: Optional[Chroma] = None
conversation_chain: Optional[ConversationalRetrievalChain] = None
memory: Optional[ConversationBufferMemory] = None

# Mapping for different document loaders
FILE_LOADER_MAP = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".csv": CSVLoader,
    # Add more as needed, e.g., ".html": UnstructuredHTMLLoader
}

def initialize_rag_components():
    """
    Initializes all LangChain components: LLM, Embeddings, Vector Store,
    and the Conversational Retrieval Chain.
    This function is called once on application startup.
    """
    global llm, embeddings_model, vectorstore, conversation_chain, memory

    logger.info("Initializing LangChain components...")

    try:
        # 1. Initialize LLM (Gemini Pro)
        if GOOGLE_API_KEY:
            llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.1)
            logger.info("Gemini Pro LLM initialized.")
        else:
            logger.warning("Google API Key not set. LLM will not be initialized.")
            llm = None

        # 2. Initialize Embeddings Model (Google Generative AI Embeddings)
        if GOOGLE_API_KEY:
            embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
            logger.info("GoogleGenerativeAIEmbeddings initialized.")
        else:
            logger.warning("Google API Key not set. Embeddings model will not be initialized.")
            embeddings_model = None

        if not embeddings_model:
            logger.error("Embeddings model is not initialized. Cannot build vector store.")
            return # Cannot proceed without embeddings

        # 3. Load Documents and Create/Load Vector Store
        documents = []
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR) # Create data directory if it doesn't exist
            logger.info(f"Created data directory: {DATA_DIR}")

        # Use DirectoryLoader to load documents of various types
        # This will iterate through DATA_DIR and use appropriate loaders
        loader_kwargs = {
            "loader_cls": FILE_LOADER_MAP.get(".txt"), # Default loader if not specified
            "loader_kwargs": {},
            "glob": "**/*", # Load all files recursively
            "silent_errors": True # Don't crash on unreadable files
        }
        
        # Manually iterate and load to use specific loaders for each extension
        loaded_count = 0
        for root, _, files in os.walk(DATA_DIR):
            for filename in files:
                file_path = os.path.join(root, filename)
                file_ext = os.path.splitext(filename)[1].lower()
                
                if file_ext in FILE_LOADER_MAP:
                    try:
                        loader_class = FILE_LOADER_MAP[file_ext]
                        loader = loader_class(file_path)
                        docs = loader.load()
                        documents.extend(docs)
                        loaded_count += 1
                        logger.info(f"Loaded document: {filename} ({len(docs)} pages/chunks)")
                    except Exception as e:
                        logger.warning(f"Error loading {filename} with {loader_class.__name__}: {e}")
                else:
                    logger.info(f"Skipping unsupported file type: {filename}")

        if not documents:
            logger.warning("No valid documents loaded. Creating a dummy document for initial setup.")
            dummy_content = "This is a dummy document. For the chatbot to provide specific answers, please add your actual files (e.g., .txt, .pdf, .docx, .csv) to the 'data' directory. After adding files, trigger a re-index or restart the backend server."
            documents.append({"page_content": dummy_content, "metadata": {"source": "dummy_doc.txt"}})
            # Note: For dummy, we don't save to disk to avoid clutter.

        # Split documents into smaller, overlapping chunks for better retrieval.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(splits)} chunks.")

        # Create or load the Chroma vector store.
        # If `CHROMA_DB_DIR` contains existing data, Chroma will load it.
        # Otherwise, it will create a new one from `splits`.
        if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
            logger.info(f"Loading existing vector store from '{CHROMA_DB_DIR}'.")
            vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings_model)
        else:
            logger.info(f"Creating new vector store and persisting to '{CHROMA_DB_DIR}'.")
            vectorstore = Chroma.from_documents(splits, embeddings_model, persist_directory=CHROMA_DB_DIR)
            vectorstore.persist() # Explicitly save the embeddings to disk
        
        logger.info("Vector store created/loaded.")

        # 4. Initialize Conversational Memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        logger.info("Conversation memory initialized.")

        # 5. Create Conversational Retrieval Chain (the heart of RAG)
        if llm and vectorstore:
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory,
                return_source_documents=True, # Essential to get the retrieved documents back
                verbose=False # Set to True for detailed chain execution logs (useful for debugging)
            )
            logger.info("Conversational Retrieval Chain initialized.")
        else:
            logger.warning("LLM or Vector Store not initialized. Conversation Chain will not be fully functional.")
            conversation_chain = None

    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}", exc_info=True)
        # In a production system, you might want to health check this state.

# --- Pydantic Models for API Request/Response ---

class ChatMessage(BaseModel):
    """Represents an incoming chat message from the frontend."""
    message: str = Field(min_length=1, max_length=500) # Basic input validation
    session_id: str = "default_session" # To manage multiple users/sessions (not fully implemented in memory yet)

class SourceDocument(BaseModel):
    """Represents a source document retrieved by the RAG system."""
    content_snippet: str
    title: str
    metadata: Dict[str, Any]

class ChatResponse(BaseModel):
    """Represents the outgoing chat response to the frontend."""
    response: str
    sources: List[SourceDocument]

class ReindexResponse(BaseModel):
    """Response model for the re-indexing endpoint."""
    status: str
    message: str
    documents_indexed: int

# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    """
    Event handler that runs when the FastAPI application starts up.
    This is where we initialize our LangChain RAG components.
    """
    logger.info("FastAPI app starting up...")
    initialize_rag_components()
    logger.info("FastAPI app ready.")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    """
    Handles incoming chat messages.
    Invokes the LangChain ConversationalRetrievalChain to get a response
    and associated source documents.
    """
    if not conversation_chain:
        raise HTTPException(status_code=503, detail="AI services are not initialized. Please check backend logs.")

    try:
        logger.info(f"Received chat message for session {chat_message.session_id}: '{chat_message.message}'")
        # The ConversationalRetrievalChain handles the chat history internally via `memory`
        # and also performs the retrieval.
        result = conversation_chain.invoke({"question": chat_message.message})
        
        response_text = result.get("answer", "Sorry, I couldn't generate a response.")
        source_documents = result.get("source_documents", [])

        sources_list = []
        for doc in source_documents:
            # Extract relevant info from source document metadata
            # The 'source' metadata key typically holds the file path.
            doc_title = os.path.basename(doc.metadata.get('source', 'Unknown Source')).replace('.txt', '').replace('.pdf', '').replace('.docx', '').replace('.csv', '')
            sources_list.append(SourceDocument(
                content_snippet=doc.page_content[:200] + "...", # Provide a snippet of the content
                metadata=doc.metadata, # Include full metadata for debugging/detail
                title=doc_title
            ))
        logger.info(f"Generated response: '{response_text[:50]}...' with {len(sources_list)} sources.")
        return ChatResponse(response=response_text, sources=sources_list)
    except Exception as e:
        logger.error(f"Error in chat endpoint for session {chat_message.session_id}: {e}", exc_info=True)
        # Raise an HTTPException to send a proper error response to the frontend
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/reindex_kb", response_model=ReindexResponse)
async def reindex_kb_endpoint():
    """
    Triggers a re-indexing of the knowledge base from the DATA_DIR.
    This will delete the old ChromaDB and create a new one.
    """
    logger.info("Re-indexing knowledge base requested.")
    global vectorstore # Need to modify the global vectorstore instance

    try:
        # Clear existing ChromaDB persistent directory
        if os.path.exists(CHROMA_DB_DIR):
            shutil.rmtree(CHROMA_DB_DIR)
            logger.info(f"Removed old ChromaDB directory: {CHROMA_DB_DIR}")
        
        # Re-initialize only the vector store part
        documents = []
        loaded_count = 0
        for root, _, files in os.walk(DATA_DIR):
            for filename in files:
                file_path = os.path.join(root, filename)
                file_ext = os.path.splitext(filename)[1].lower()
                
                if file_ext in FILE_LOADER_MAP:
                    try:
                        loader_class = FILE_LOADER_MAP[file_ext]
                        loader = loader_class(file_path)
                        docs = loader.load()
                        documents.extend(docs)
                        loaded_count += 1
                        logger.info(f"Loaded document for re-indexing: {filename}")
                    except Exception as e:
                        logger.warning(f"Error loading {filename} during re-indexing: {e}")
                else:
                    logger.info(f"Skipping unsupported file type during re-indexing: {filename}")

        if not documents:
            logger.warning("No valid documents found for re-indexing. Creating a dummy document.")
            dummy_content = "The knowledge base is empty after re-indexing. Please add files to 'data' and re-index."
            splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(dummy_content)
            vectorstore = Chroma.from_texts(splits, embeddings_model, persist_directory=CHROMA_DB_DIR)
            vectorstore.persist()
            logger.info("Created dummy vector store after empty re-index.")
            return ReindexResponse(status="success", message="Knowledge base re-indexed, but no valid documents found. Dummy KB created.", documents_indexed=0)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        logger.info(f"Re-indexed {loaded_count} documents into {len(splits)} chunks.")

        vectorstore = Chroma.from_documents(splits, embeddings_model, persist_directory=CHROMA_DB_DIR)
        vectorstore.persist()
        logger.info("Knowledge base re-indexed and persisted successfully.")
        
        # Reset memory for current conversation chain if needed, or create a new chain
        # For simplicity, we'll just rely on the existing chain picking up new retriever
        # A more robust solution might recreate the chain or manage memory per session.
        global conversation_chain, memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        logger.info("Conversation chain re-linked to new vector store.")


        return ReindexResponse(status="success", message="Knowledge base re-indexed successfully!", documents_indexed=loaded_count)
    except Exception as e:
        logger.error(f"Error during knowledge base re-indexing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to re-index knowledge base: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "llm_initialized": llm is not None,
        "embeddings_initialized": embeddings_model is not None,
        "vectorstore_initialized": vectorstore is not None,
        "memory_initialized": memory is not None
    }

# To run this backend:
# 1. Create a virtual environment: python -m venv venv
# 2. Activate it:
#    On Windows: .\venv\Scripts\activate
#    On macOS/Linux: source venv/bin/activate
# 3. Install dependencies: pip install -r requirements.txt
# 4. Run the app: uvicorn main:app --reload --port 8000
