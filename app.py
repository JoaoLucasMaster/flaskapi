from flask import Flask, jsonify, request
import numpy as np
import google.generativeai as genai
import pickle
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)
CORS(app)  # Initialize CORS for the entire application

model = 'models/text-embedding-004'
modeloEmbeddings = pickle.load(open('datasetEmbeddings.pkl', 'rb'))
chave_secreta = os.getenv('API_KEY')
genai.configure(api_key=chave_secreta)

# Função para buscar a consulta
def gerarBuscarConsulta(consulta, dataset):
    embedding_consulta = genai.embed_content(
        model=model,
        content=consulta,
        task_type="retrieval_query",
    )
    
    embeddings_array = np.array(dataset["Embeddings"].tolist())
    produtos_escalares = np.dot(embeddings_array, embedding_consulta['embedding'])
    indice = np.argmax(produtos_escalares)
    
    return dataset.iloc[indice]['Conteúdo']

model2 = genai.GenerativeModel(model_name="gemini-1.0-pro")

@app.route("/")
def home():
    consulta = "Quem é você ?"
    resposta = gerarBuscarConsulta(consulta, modeloEmbeddings)
    prompt = f"Considere a consulta, {consulta}, Reescreva as sentenças de resposta de uma forma alternativa, não apresente opções de reescrita, {resposta}"
    response = model2.generate_content(prompt)
    return response.text

@app.route("/api", methods=["POST"])
def results():
    auth_key = request.headers.get("Authorization")
    if auth_key != chave_secreta:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json(force=True)
    consulta = data.get("consulta", "")
    
    if not consulta:
        return jsonify({"error": "Consulta não fornecida"}), 400
    
    resultado = gerarBuscarConsulta(consulta, modeloEmbeddings)
    prompt = f"Considere a consulta, {consulta}, Reescreva as sentenças de resposta de uma forma alternativa, não apresente opções de reescrita, {resultado}"
    response = model2.generate_content(prompt)
    
    # Retorna apenas o texto da resposta sem formatação adicional
    return jsonify({"mensagem": response.text.strip()})
