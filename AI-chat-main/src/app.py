"""
API Flask simple con CORS habilitado y un endpoint de saludo.
Establece una conexión con un chatbot RAG y permite la interacción a través de un endpoint.
"""
import os
from flask import Flask, render_template, jsonify
from flask_cors import CORS
import requests
from rag import RAGChatbot

app = Flask(__name__, template_folder='template')
CORS(app)  # Permitir todos los CORS


@app.route('/', methods=['GET'])
def hello_world():
    """Renderiza el template HTML principal."""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Recibe un mensaje y retorna respuesta del chatbot."""
    data = requests.json.get()["message"]  # No se usa
    try:
        response = chatbot.answer_question(data).text
    except requests.exceptions.RequestException as e:
        response = f"Error obteniendo cita: {e}"
    return jsonify({'response': response})


if __name__ == '__main__':
    chatbot = RAGChatbot()
    
    # Uncomment to ingest documents
    articulos = [f"./Data/articulos/articulo_{x}.txt" for x in range(1,len(os.listdir("./Data/articulos"))+1)]
    #chatbot.ingest_documents(articulos)
    
    # Uncomment to load vector store
    chatbot.read_files(articulos)
    chatbot.load_vector_store("./Data/Embeddings/article_embeddings.npy")
    # Exponer en 0.0.0.0 es intencional para desarrollo de APIs
    app.run(host='127.0.0.1', port=5001)
