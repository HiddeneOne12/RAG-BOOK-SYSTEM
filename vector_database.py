"""
Vector Database Manager
Handles ChromaDB operations for document storage and retrieval
"""

import logging
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from config import CHROMA_PERSIST_DIRECTORY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class VectorDatabase:
    """Manages ChromaDB operations for document storage and retrieval"""
    
    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIRECTORY,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            
            logger.info("Vector database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector database: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document], source_name: str):
        """
        Add documents to ChromaDB collection
        
        **How Vector Storage Works:**
        1. **Text Extraction**: Extract text content from documents
        2. **Embedding Generation**: Use SentenceTransformer to create vector embeddings
        3. **Metadata Creation**: Add source, chunk ID, and type information
        4. **Storage**: Store embeddings, text, and metadata in ChromaDB
        
        **Packages Used:**
        - `chromadb`: Vector database for storing and retrieving embeddings
        - `sentence_transformers`: Generate embeddings from text
        - `langchain.schema.Document`: Document structure
        
        **Embedding Model:**
        - Model: `all-MiniLM-L6-v2` (384-dimensional vectors)
        - Purpose: Convert text to numerical vectors for similarity search
        - Distance: Cosine similarity for finding related content
        """
        try:
            # Extract text content
            texts = [doc.page_content for doc in documents]
            
            # Create metadata
            metadatas = [
                {
                    "source": source_name, 
                    "chunk_id": i, 
                    "type": documents[0].metadata.get("type", "pdf")
                } 
                for i in range(len(documents))
            ]
            
            # Generate unique IDs
            ids = [f"{source_name}_{i}" for i in range(len(documents))]
            
            # Generate embeddings using SentenceTransformer
            embeddings = self.embedding_model.encode(texts).tolist()
            
            # Add to ChromaDB collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to collection")
            
        except Exception as e:
            logger.error(f"Error adding documents to collection: {str(e)}")
            raise
    
    def search_similar_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Search for similar documents using vector similarity
        
        **How Vector Search Works:**
        1. **Query Embedding**: Convert search query to vector using same model
        2. **Similarity Search**: Find most similar document vectors using cosine distance
        3. **Relevance Filtering**: Filter results by similarity threshold (< 0.8)
        4. **Relevance Scoring**: Calculate relevance score (1 - distance)
        5. **Ranking**: Sort by relevance score (highest first)
        
        **Search Process:**
        1. Generate query embedding
        2. Search ChromaDB for similar vectors
        3. Filter by distance threshold
        4. Calculate relevance scores
        5. Return top 3 most relevant results
        
        **Parameters:**
        - n_results: Number of results to retrieve (default: 5)
        - Distance threshold: < 0.8 for good similarity
        - Return limit: Top 3 most relevant results
        """
        try:
            # Convert query to embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format and filter results
            formatted_results = []
            if results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    # Only include results with good similarity (distance < 0.8)
                    if results['distances'][0][i] < 0.8:
                        formatted_results.append({
                            'document': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'distance': results['distances'][0][i],
                            'relevance_score': 1 - results['distances'][0][i]  # Higher score = more relevant
                        })
            
            # Sort by relevance score (highest first)
            formatted_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Return top 3 most relevant results
            return formatted_results[:3]
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": "documents",
                "embedding_model": EMBEDDING_MODEL
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {"total_documents": 0, "collection_name": "documents", "embedding_model": EMBEDDING_MODEL}
