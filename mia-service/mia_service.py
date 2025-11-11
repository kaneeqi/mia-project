"""
MIA Emotion Service - Flask API
Expone MiaMotion y MiaPredict como endpoints REST
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import json
from pathlib import Path

from inference_emotion_classifier import predict as predict_user_emotion
from agent_emotion_predict_classifier import AgentEmotionPredictClassifier

app = Flask(__name__)
CORS(app)  # Permitir requests desde Node.js

# ==================== CONFIGURACIÓN ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Usando dispositivo: {DEVICE}")

# ==================== CARGAR MODELOS ====================
print("📦 Cargando MiaPredict...")

# Cargar configuración
with open("config_agent.json", "r", encoding="utf-8") as f:
    agent_config = json.load(f)

# Inicializar MiaPredict
agent_model = AgentEmotionPredictClassifier(
    model_name=agent_config.get("base_model_id", "dccuchile/bert-base-spanish-wwm-cased"),
    pretrained_encoder=agent_config.get("pretrained_encoder", "beto"),
    max_length=agent_config.get("max_length", 128),
    hidden1=agent_config.get("hidden1", 256),
    hidden2=agent_config.get("hidden2", 64),
    num_classes=agent_config.get("num_classes", 2),
    dropout=agent_config.get("dropout", 0.4),
    label_feature_dropout=agent_config.get("label_feature_dropout", 0.5),
    device=DEVICE
)

# Cargar pesos
checkpoint_path = Path("best_model_agent.pt")
if checkpoint_path.exists():
    state = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        agent_model.load_state_dict(state["model_state_dict"])
    else:
        agent_model.load_state_dict(state)
    agent_model.eval()
    print("✅ MiaPredict cargado correctamente")
else:
    print("⚠️  No se encontró best_model_agent.pt")

# Mapeo de emociones (índice interno → nombre español)
EMOTION_MAP = {
    0: "tristeza",
    1: "alegría", 
    2: "amor",
    3: "ira",
    4: "miedo",
    5: "sorpresa"
}

# Mapeo de emociones para el avatar (nombres en inglés para compatibilidad)
EMOTION_TO_FACIAL_EXPRESSION = {
    "tristeza": "sad",
    "alegría": "smile",
    "amor": "smile",  # o "default" si el avatar tiene expresión específica
    "ira": "angry",
    "miedo": "sad",
    "sorpresa": "surprised"
}

# ==================== ENDPOINTS ====================

@app.route('/health', methods=['GET'])
def health():
    """Endpoint para verificar que el servicio está activo"""
    return jsonify({
        "status": "healthy",
        "device": str(DEVICE),
        "models_loaded": True
    })


@app.route('/classify_emotion', methods=['POST'])
def classify_user_emotion():
    """
    Clasifica la emoción del USUARIO usando MiaMotion
    
    Request:
    {
        "text": "Estoy muy triste hoy"
    }
    
    Response:
    {
        "emotion": "tristeza",
        "label": 0,
        "confidence": 0.85,
        "all_probs": {
            "tristeza": 0.85,
            "alegría": 0.05,
            ...
        }
    }
    """
    try:
        data = request.json
        text = data.get("text", "")
        
        if not text:
            return jsonify({"error": "Campo 'text' requerido"}), 400
        
        # Usar MiaMotion (inference_emotion_classifier.py)
        emotion, probs = predict_user_emotion(text, return_probs=True)
        
        # Encontrar el label (índice)
        label = None
        for idx, name in EMOTION_MAP.items():
            if name == emotion:
                label = idx
                break
        
        # Construir dict de probabilidades
        all_probs = {EMOTION_MAP[i]: float(probs[i]) for i in range(len(probs))}
        
        return jsonify({
            "emotion": emotion,
            "label": label,
            "confidence": float(max(probs)),
            "all_probs": all_probs
        })
        
    except Exception as e:
        print(f"❌ Error en classify_emotion: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict_agent_emotion', methods=['POST'])
def predict_agent_emotion():
    """
    Predice la emoción con la que el AGENTE debe responder usando MiaPredict
    
    Request:
    {
        "text": "Estoy muy triste hoy",
        "user_label": 0  // label de la emoción del usuario
    }
    
    Response:
    {
        "agent_emotion": "amor",
        "agent_label": 2,
        "confidence": 0.65,
        "facial_expression": "smile",
        "all_probs": {
            "alegría": 0.35,
            "amor": 0.65
        }
    }
    """
    try:
        data = request.json
        text = data.get("text", "")
        user_label = data.get("user_label")
        
        if not text or user_label is None:
            return jsonify({"error": "Campos 'text' y 'user_label' requeridos"}), 400
        
        # Convertir a tensor
        user_label_tensor = torch.tensor([user_label], dtype=torch.long)
        
        # Predecir con MiaPredict
        with torch.no_grad():
            preds, probs = agent_model.predict([text], user_label_tensor)
        
        agent_label_internal = preds[0].item()  # 0 o 1 (índice en el modelo binario)
        agent_probs = probs[0].cpu().numpy()
        
        # Mapear índice interno a la emoción real
        # Según config_agent.json: present_classes = [1, 2] → alegría, amor
        present_classes = agent_config.get("present_classes", [1, 2])
        agent_label_global = present_classes[agent_label_internal]
        agent_emotion = EMOTION_MAP[agent_label_global]
        
        # Expresión facial para el avatar
        facial_expression = EMOTION_TO_FACIAL_EXPRESSION.get(agent_emotion, "default")
        
        # Construir dict de probabilidades (solo las 2 clases entrenadas)
        class_names = agent_config.get("class_names", ["alegría", "amor"])
        all_probs = {class_names[i]: float(agent_probs[i]) for i in range(len(agent_probs))}
        
        return jsonify({
            "agent_emotion": agent_emotion,
            "agent_label": agent_label_global,
            "confidence": float(max(agent_probs)),
            "facial_expression": facial_expression,
            "all_probs": all_probs
        })
        
    except Exception as e:
        print(f"❌ Error en predict_agent_emotion: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/full_pipeline', methods=['POST'])
def full_pipeline():
    """
    Pipeline completo: detecta emoción del usuario Y predice emoción del agente
    
    Request:
    {
        "text": "Estoy muy triste hoy"
    }
    
    Response:
    {
        "user": {
            "emotion": "tristeza",
            "label": 0,
            "confidence": 0.85
        },
        "agent": {
            "emotion": "amor",
            "label": 2,
            "confidence": 0.65,
            "facial_expression": "smile"
        }
    }
    """
    try:
        data = request.json
        text = data.get("text", "")
        
        if not text:
            return jsonify({"error": "Campo 'text' requerido"}), 400
        
        # Paso 1: Clasificar emoción del usuario
        emotion_user, probs_user = predict_user_emotion(text, return_probs=True)
        label_user = None
        for idx, name in EMOTION_MAP.items():
            if name == emotion_user:
                label_user = idx
                break
        
        # Paso 2: Predecir emoción del agente
        user_label_tensor = torch.tensor([label_user], dtype=torch.long)
        with torch.no_grad():
            preds, probs_agent = agent_model.predict([text], user_label_tensor)
        
        agent_label_internal = preds[0].item()
        present_classes = agent_config.get("present_classes", [1, 2])
        agent_label_global = present_classes[agent_label_internal]
        agent_emotion = EMOTION_MAP[agent_label_global]
        facial_expression = EMOTION_TO_FACIAL_EXPRESSION.get(agent_emotion, "default")
        
        return jsonify({
            "user": {
                "emotion": emotion_user,
                "label": label_user,
                "confidence": float(max(probs_user))
            },
            "agent": {
                "emotion": agent_emotion,
                "label": agent_label_global,
                "confidence": float(probs_agent[0].max().item()),
                "facial_expression": facial_expression
            }
        })
        
    except Exception as e:
        print(f"❌ Error en full_pipeline: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 MIA Emotion Service - Servidor iniciado")
    print("="*60)
    print(f"   📍 Endpoint: http://localhost:5000")
    print(f"   🔧 Device: {DEVICE}")
    print(f"   📊 Endpoints disponibles:")
    print(f"      • GET  /health")
    print(f"      • POST /classify_emotion")
    print(f"      • POST /predict_agent_emotion")
    print(f"      • POST /full_pipeline")
    print("="*60 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=True)
