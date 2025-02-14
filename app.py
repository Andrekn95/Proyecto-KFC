from flask import Flask, request, jsonify, render_template
import ollama
import json
import os
from sentence_transformers import SentenceTransformer, util
from docx import Document
import numpy as np
import subprocess

app = Flask(__name__)

DOCUMENTO_DOCX = "documento.docx"
ENTRENAMIENTO_JSONL = "entrenamiento.jsonl"
DATA_JSON = "data.json"
NOMBRE_MODELO = "llama3"

def descargar_modelo():
    try:
        print(f"📥 Descargando modelo {NOMBRE_MODELO}...")
        subprocess.run(["ollama", "pull", NOMBRE_MODELO], check=True)
        print("✅ Modelo descargado correctamente.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error al descargar el modelo: {e}")

descargar_modelo()

modelo_embeddings = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def cargar_data_json():
    if os.path.exists(DATA_JSON):
        with open(DATA_JSON, "r", encoding="utf-8") as archivo:
            return json.load(archivo)
    return {}

def guardar_data_json(data):
    with open(DATA_JSON, "w", encoding="utf-8") as archivo:
        json.dump(data, archivo, ensure_ascii=False, indent=4)

def extraer_parrafos_docx(docx_path):
    try:
        if not os.path.exists(docx_path):
            print(f"⚠️ Archivo DOCX '{docx_path}' no encontrado.")
            return []
        
        doc = Document(docx_path)
        parrafos = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        print(f"📖 Documento cargado con {len(parrafos)} párrafos.")
        return parrafos
    except Exception as e:
        print(f"⚠️ Error al leer el DOCX: {e}")
        return []

def cargar_entrenamiento_jsonl(jsonl_path):
    datos = {}
    try:
        if not os.path.exists(jsonl_path):
            print(f"⚠️ Archivo JSONL '{jsonl_path}' no encontrado.")
            return datos
        
        with open(jsonl_path, "r", encoding="utf-8") as archivo:
            for linea in archivo:
                try:
                    data = json.loads(linea)
                    datos[data["input"].strip().lower()] = data["output"]
                except json.JSONDecodeError:
                    print("⚠️ Error en una línea del JSONL, se omitió.")
        
        print(f"📂 Cargadas {len(datos)} preguntas del JSONL.")
    except Exception as e:
        print(f"⚠️ Error al cargar JSONL: {e}")
    
    return datos

def encontrar_parrafos_relevantes(pregunta, parrafos, top_n=3):
    try:
        if not parrafos:
            print("⚠️ No hay párrafos para buscar.")
            return []

        pregunta_embedding = modelo_embeddings.encode(pregunta, convert_to_tensor=True)
        parrafos_embeddings = modelo_embeddings.encode(parrafos, convert_to_tensor=True)

        if pregunta_embedding is None or parrafos_embeddings is None:
            print("⚠️ No se pudieron generar embeddings.")
            return []

        similitudes = util.pytorch_cos_sim(pregunta_embedding, parrafos_embeddings)[0]

        if similitudes.shape[0] == 0:
            print("⚠️ No se encontraron similitudes.")
            return []

        indices_relevantes = np.argsort(similitudes.cpu().numpy(), axis=0)[-top_n:][::-1]
        return [parrafos[i] for i in indices_relevantes]
    except Exception as e:
        print(f"⚠️ Error al buscar párrafos relevantes: {e}")
        return []

parrafos_docx = extraer_parrafos_docx(DOCUMENTO_DOCX)
entrenamiento_jsonl = cargar_entrenamiento_jsonl(ENTRENAMIENTO_JSONL)
data_json = cargar_data_json()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = request.json.get("question", "").strip().lower()
        if not question:
            return jsonify({"response": "Por favor, ingresa una pregunta válida."})

        print(f"🔍 Pregunta recibida: {question}")

        if question in data_json:
            respuesta = data_json[question]
            print("✅ Respuesta encontrada en DATA_JSON.")
        elif question in entrenamiento_jsonl:
            respuesta = entrenamiento_jsonl[question]
            print("✅ Respuesta encontrada en JSONL.")
        else:
            parrafos_relevantes = encontrar_parrafos_relevantes(question, parrafos_docx)
            if not parrafos_relevantes:
                return jsonify({"response": "No tengo información sobre eso."})

            contexto = "\n".join(parrafos_relevantes)
            print(f"📖 Contexto seleccionado: {contexto[:200]}...")

            prompt = f"Contexto: {contexto}\nPregunta: {question}\nRespuesta:"
            print(f"🔍 Consultando a Ollama...")

            response = ollama.chat(model=NOMBRE_MODELO, messages=[{"role": "user", "content": prompt}])
            respuesta = response.get("message", {}).get("content", "No tengo una respuesta en este momento.")
            print(f"📝 Respuesta de Ollama: {respuesta[:200]}...")
            
            data_json[question] = respuesta
            guardar_data_json(data_json)

        return jsonify({"response": respuesta})
    except Exception as e:
        print(f"⚠️ Error en el servidor: {e}")
        return jsonify({"response": "Hubo un error al procesar la pregunta. Inténtalo de nuevo."})

if __name__ == "__main__":
    app.run(debug=True)
