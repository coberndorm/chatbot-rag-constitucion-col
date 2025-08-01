"""
RAG Chatbot Implementation

This code creates a simple but effective RAG (Retrieval-Augmented Generation) chatbot that:
1. Ingests PDF, text, or other documents
2. Creates vector embeddings from document chunks
3. Retrieves relevant information based on user queries
4. Generates responses using an LLM enhanced with retrieved context
"""

import os
import re
import numpy as np
from typing import List, Dict, Any, Tuple
import requests
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

from sentence_transformers import SentenceTransformer
from google import genai
from scipy.spatial.distance import cosine


class RAGChatbot:
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.7):
        """Initialize the RAG chatbot with specified model and settings."""
        # Check for API key
        self.api_key = os.getenv("GEMINI_API_KEY", "your-api-key")
        
        if not self.api_key:
            raise ValueError("Gemini API key not found. Please set OPENAI_API_KEY environment variable.")

        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", "your-api-key"))


        self.embedding_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        self.vector_store = None
        
        # Set up the prompt template for RAG
        self.template = """
        Eres un especialiste en la ley colombiana, Tienes mas de 20 años de experiencia como abogado en colombia. 
        Te dedicaste a entender la constitucion colombiana y la ley colombiana.
        Eres un experto en derecho penal, civil, administrativo y comercial.
        Utiliza la siguiente información para responder la pregunta del usuario.
        Si no sabes la respuesta, simplemente di que no lo sabes. No intentes inventar una respuesta.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """

    
    def read_files(self, file_paths: List[str]):
        """
        Read and decode files from the given paths.
        """
        documents = []
        
        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                # Read the file and decode it
                doc = f.read().decode("utf-8" , errors="ignore")
            documents.append(doc)
        
        self.documents = documents
        return documents
    def ingest_documents(self, file_paths: List[str]):
        
        """Generate embeddings for all texts"""
        self.read_files(file_paths)
        # Split documents into chunks
        if self.documents:
            self.vector_store = self.embedding_model.encode(self.documents, show_progress_bar=True)
            print(f"Successfully ingested {len(self.documents)} document chunks.")
            return self.vector_store
            
        else:
            print("No documents were loaded. Check file paths and formats.")
    
    def retrieve_context(self, query: str, top_n: int = 10, top_score: float = 0.8) -> str:
        """Retrieve relevant document chunks based on the query."""
        def cosine_similarity(a, b):
            dot_product = sum([x * y for x, y in zip(a, b)])
            norm_a = sum([x ** 2 for x in a]) ** 0.5
            norm_b = sum([x ** 2 for x in b]) ** 0.5
            return dot_product / (norm_a * norm_b)
        
        # Calculate cosine similarity between query and document embeddings
        query_embedding = self.embedding_model.encode([query])
        
        similarities = []
        for idx, embedding in enumerate(self.vector_store):
            # Calculate cosine similarity
            similarity = cosine_similarity(query_embedding[0], embedding)
            similarities.append((idx, similarity))
            
        # sort by similarity in descending order, because higher similarity means more relevant chunks
        similarities.sort(key=lambda x: x[1], reverse=True)
        # finally, return the top N most relevant chunks
        return similarities[:top_n]
    

    
    def answer_question(self, question: str) -> str:
        """Generate an answer to the question using RAG."""
        if self.vector_store is None:
            return "I need some documents to learn from first. Please use the ingest_documents method."
        
        # Retrieve relevant context
        context = self.retrieve_context(question)
        
        # Generate answer
        response = self.client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=self.template.format(context='\n'.join([f' - {self.documents[chunk]}' for chunk, similarity in context]), 
                                        question=question)
            )
        print(self.template.format(context='\n'.join([f' - {self.documents[chunk]}' for chunk, similarity in context]), 
                                        question=question))
        return response

    def save_vector_store(self, output_path: str = "article_embeddings.npy"):
        """Save the vector store to disk."""
        if self.vector_store is not None:
            # Save embeddings
            np.save(output_path, self.vector_store)
            print(f"Vector store saved to {output_path}")
        else:
            print("No vector store to save.")
            
    def load_vector_store(self, directory: str = "./chroma_db"):
        """Load the vector store from disk or load precomputed embeddings."""
        if os.path.exists(directory):
            # Load vector store from disk
            self.vector_store = np.load(directory, allow_pickle=True)
            print(f"Vector store loaded from {directory}")
        else:
            print(f"No vector store found at {directory} and no embeddings file provided.")

# Example usage
if __name__ == "__main__":
    # Create the chatbot
    chatbot = RAGChatbot()
    
    # Uncomment to ingest documents
    articulos = [f"./Data/articulos/articulo_{x}.txt" for x in range(1,len(os.listdir("./Data/articulos"))+1)]
    #chatbot.ingest_documents(articulos)
    
    # Uncomment to load vector store
    chatbot.read_files(articulos)
    chatbot.load_vector_store("./Data/Embeddings/article_embeddings.npy")
    # Uncomment to save vector store
    #chatbot.save_vector_store("./Data/Embeddings/article_embeddings.npy") 
    # Interactive chat loop
    print("RAG Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        response = chatbot.answer_question(user_input).text
        # Print the response
        print(f"Chatbot: {response}")
        