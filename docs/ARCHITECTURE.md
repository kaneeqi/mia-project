# 🏗️ Arquitectura del Sistema MIA

## Tabla de Contenidos
- [Visión General](#visión-general)
- [Componentes](#componentes)
- [Flujo de Datos](#flujo-de-datos)
- [Tecnologías](#tecnologías)
- [Decisiones de Diseño](#decisiones-de-diseño)

---

## 📊 Visión General

MIA es un sistema de múltiples capas que combina:
1. **Frontend React**: Interfaz de usuario y avatar 3D
2. **Backend Node.js**: Orquestador principal
3. **MIA Service (Python)**: Modelos de IA para emociones
4. **APIs Externas**: Groq (texto), ElevenLabs (voz), Rhubarb (lip-sync)

```
┌─────────────┐
│   Usuario   │
└──────┬──────┘
       │ Escribe mensaje
       ▼
┌─────────────────────────────────────┐
│         FRONTEND (React)            │
│  - UI de chat                       │
│  - Avatar 3D (Three.js)             │
│  - Reproducción de audio            │
└──────┬──────────────────────────────┘
       │ POST /chat
       ▼
┌─────────────────────────────────────┐
│      BACKEND (Node.js)              │
│  1. Recibe mensaje                  │
│  2. Llama a MIA Service             │
│  3. Genera texto (Groq)             │
│  4. Genera audio (ElevenLabs)       │
│  5. Genera lip-sync (Rhubarb)       │
│  6. Retorna respuesta completa      │
└──────┬─────────────┬────────────────┘
       │             │
       │             │ POST /full_pipeline
       │             ▼
       │      ┌─────────────────────────┐
       │      │  MIA SERVICE (Python)   │
       │      │  - MiaMotion (BERT)     │
       │      │  - MiaPredict (BERT)    │
       │      │  - Mapeo a expresiones  │
       │      └─────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│         APIs EXTERNAS               │
│  - Groq (LLM)                       │
│  - ElevenLabs (TTS)                 │
│  - Rhubarb (Lip-sync)               │
└─────────────────────────────────────┘
```

---

## 🧩 Componentes

### **1. Frontend (React + Three.js)**

**Ubicación:** `/frontend/src/`

**Componentes Principales:**

#### `App.jsx`
- Punto de entrada de la aplicación
- Maneja estado global
- Renderiza Experience y UI

#### `Experience.jsx`
- Escena 3D de Three.js
- Iluminación y cámara
- Contenedor del Avatar

#### `Avatar.jsx`
- Carga modelo 3D (.glb)
- Maneja animaciones
- Reproduce audio y lip-sync
- Aplica expresiones faciales

```javascript
// Estructura de datos del Avatar
{
  animations: {
    Idle,
    Talking_1,
    Laughing,
    Crying,
    Angry,
    Terrified,
    Surprised
  },
  facialExpressions: {
    default,
    smile,
    sad,
    angry,
    surprised,
    funnyFace
  }
}
```

#### `UI.jsx`
- Input de texto
- Botón de enviar
- Indicador de carga

#### `useChat.jsx` (Hook)
- Lógica de comunicación con backend
- Cola de mensajes
- Estado de reproducción

**Flujo de renderizado:**
```
Usuario escribe → useChat envía POST → Backend responde → 
Avatar recibe mensaje → Reproduce audio → Sincroniza labios → 
Ejecuta animación → Aplica expresión facial
```

---

### **2. Backend (Node.js + Express)**

**Ubicación:** `/backend/index.js`

**Responsabilidades:**
1. **Orquestación**: Coordina todos los servicios
2. **Transformación**: Convierte MP3 → WAV para Rhubarb
3. **Limpieza**: Elimina archivos temporales
4. **Fallback**: Usa datos mock si servicios fallan

**Endpoints:**

#### `POST /chat`
```javascript
Request:
{
  "message": "Hola, ¿cómo estás?",
  "conversationHistory": [] // Opcional
}

Response:
{
  "messages": [
    {
      "text": "Hola, ¿cómo estás?",
      "audio": null,
      "lipsync": null,
      "facialExpression": "default",
      "animation": "Idle"
    },
    {
      "text": "¡Hola! Estoy muy bien...",
      "audio": "base64_encoded_audio",
      "lipsync": {
        "metadata": {...},
        "mouthCues": [...]
      },
      "facialExpression": "smile",
      "animation": "Talking_1",
      "emotions": {
        "user": { "emotion": "alegría", "confidence": 0.85 },
        "agent": { "emotion": "alegría", "confidence": 0.99 }
      }
    }
  ]
}
```

#### `GET /health`
```javascript
Response:
{
  "status": "healthy",
  "services": {
    "backend": "ok",
    "mia": { "status": "healthy", "models_loaded": true },
    "textMode": "groq",
    "elevenlabs": "configured"
  },
  "tools": {
    "ffmpeg": true,
    "rhubarb": true
  }
}
```

**Pipeline Interno:**

```javascript
// 1. Detectar emociones
const emotions = await getMiaEmotions(userMessage);

// 2. Generar texto
const textResponse = await generateTextResponse(
  userMessage, 
  emotions.agent.emotion
);

// 3. Generar audio
const audioBase64 = await generateAudio(textResponse);

// 4. Convertir MP3 → WAV
const wavFile = await convertMp3ToWav(mp3File);

// 5. Generar lip-sync
const lipsyncData = await generateLipSync(audioBase64);

// 6. Retornar respuesta completa
return {
  messages: [userMsg, agentMsg]
};
```

---

### **3. MIA Service (Python + Flask)**

**Ubicación:** `/backend/mia_service.py`

**Modelos:**

#### **MiaMotion** (Detección de Emoción del Usuario)
- **Arquitectura**: BERT fine-tuned
- **Input**: Texto del usuario
- **Output**: 6 emociones (alegría, amor, tristeza, ira, miedo, sorpresa)
- **Formato**: 
  ```python
  {
    "emotion": "alegría",
    "label": 1,
    "confidence": 0.92
  }
  ```

#### **MiaPredict** (Predicción de Respuesta del Agente)
- **Arquitectura**: BERT fine-tuned
- **Input**: Texto del usuario
- **Output**: 2 emociones (alegría, amor)
- **Formato**:
  ```python
  {
    "emotion": "amor",
    "label": 2,
    "confidence": 0.87,
    "facial_expression": "smile"
  }
  ```

**Endpoint Principal:**

```python
@app.route('/full_pipeline', methods=['POST'])
def full_pipeline():
    # 1. Recibir texto
    text = request.json['text']
    
    # 2. Detectar emoción usuario
    user_emotion = predict_emotion(text, mia_motion_model)
    
    # 3. Predecir respuesta agente
    agent_emotion = predict_emotion(text, mia_predict_model)
    
    # 4. Mapear a expresión facial
    facial_expression = emotion_to_facial_expression(
        agent_emotion['emotion']
    )
    
    # 5. Retornar
    return {
        "user": user_emotion,
        "agent": {
            **agent_emotion,
            "facial_expression": facial_expression
        }
    }
```

---

### **4. APIs Externas**

#### **Groq API**
- **Propósito**: Generación de texto conversacional
- **Modelo**: Llama 3.1 8B Instant
- **Latencia**: ~200-500ms
- **Rate Limit**: 30 req/min (free tier)

```javascript
const response = await axios.post(
  "https://api.groq.com/openai/v1/chat/completions",
  {
    model: "llama-3.1-8b-instant",
    messages: [
      {
        role: "system",
        content: "Eres MIA, empática y cálida..."
      },
      { role: "user", content: userMessage }
    ],
    max_tokens: 100,
    temperature: 0.7
  }
);
```

#### **ElevenLabs API**
- **Propósito**: Text-to-Speech
- **Modelo**: eleven_multilingual_v2
- **Latencia**: ~1-3s dependiendo del texto
- **Rate Limit**: 10,000 caracteres/mes (free tier)

```javascript
const response = await axios.post(
  `https://api.elevenlabs.io/v1/text-to-speech/${voiceID}`,
  {
    text: text,
    model_id: "eleven_multilingual_v2",
    voice_settings: {
      stability: 0.5,
      similarity_boost: 0.75
    }
  },
  {
    responseType: "arraybuffer"
  }
);
```

#### **Rhubarb Lip Sync**
- **Propósito**: Análisis de fonemas para lip-sync
- **Input**: Archivo WAV (16kHz, mono)
- **Output**: JSON con visemas y timestamps
- **Latencia**: ~500ms-2s

```bash
rhubarb -f json audio.wav -o output.json
```

**Formato de salida:**
```json
{
  "metadata": {
    "soundFile": "audio.wav",
    "duration": 4.5
  },
  "mouthCues": [
    { "start": 0.0, "end": 0.2, "value": "X" },
    { "start": 0.2, "end": 0.4, "value": "B" },
    { "start": 0.4, "end": 0.6, "value": "E" }
  ]
}
```

**Visemas (8 tipos):**
- **X**: Silencio
- **A**: Abierta (ah)
- **B**: Labios cerrados (b, p, m)
- **C**: Ligeramente abierta (d, t, n)
- **D**: Dientes visibles (th)
- **E**: Sonrisa (ee)
- **F**: Labios hacia adelante (f, v)
- **G**: Garganta (k, g)
- **H**: Abierta grande (i)

---

## 🔄 Flujo de Datos Completo

### **Escenario: Usuario envía "Hola, ¿cómo estás?"**

```
1. Frontend (UI.jsx)
   └─> Usuario escribe "Hola, ¿cómo estás?"
   └─> Click en "Enviar"
   
2. Frontend (useChat.jsx)
   └─> POST http://localhost:3000/chat
       Body: { "message": "Hola, ¿cómo estás?" }
   
3. Backend (index.js)
   └─> Recibe mensaje
   └─> POST http://localhost:5000/full_pipeline
       Body: { "text": "Hola, ¿cómo estás?" }
   
4. MIA Service (mia_service.py)
   └─> MiaMotion.predict() → "alegría" (0.85)
   └─> MiaPredict.predict() → "alegría" (0.92)
   └─> emotion_to_facial_expression() → "smile"
   └─> Return: { user: {...}, agent: {...} }
   
5. Backend (index.js)
   └─> Recibe emociones
   └─> POST https://api.groq.com/... 
       Prompt: "Eres MIA empática..." + mensaje
   └─> Recibe: "¡Hola! Estoy muy bien, gracias..."
   
6. Backend (index.js)
   └─> POST https://api.elevenlabs.io/...
       Text: "¡Hola! Estoy muy bien, gracias..."
   └─> Recibe: Audio MP3 en base64
   
7. Backend (index.js)
   └─> Guarda audio.mp3 temporalmente
   └─> ffmpeg -i audio.mp3 audio.wav
   └─> rhubarb -f json audio.wav -o audio.json
   └─> Lee audio.json
   └─> Elimina archivos temporales
   
8. Backend (index.js)
   └─> Construye respuesta completa:
       {
         messages: [
           { text: "Hola...", audio: null, ... },
           { 
             text: "¡Hola! ...",
             audio: "base64...",
             lipsync: { mouthCues: [...] },
             facialExpression: "smile",
             animation: "Laughing"
           }
         ]
       }
   └─> Return al frontend
   
9. Frontend (useChat.jsx)
   └─> Recibe respuesta
   └─> Agrega mensajes a la cola
   
10. Frontend (Avatar.jsx)
    └─> Procesa primer mensaje (usuario) → skip
    └─> Procesa segundo mensaje (MIA):
        - Decodifica audio base64
        - Crea AudioBuffer
        - Aplica expresión facial "smile"
        - Inicia animación "Laughing"
        - Reproduce audio
        - Sincroniza labios con mouthCues
        
11. Usuario
    └─> Ve avatar hablar con labios sincronizados
    └─> Escucha voz femenina cálida
    └─> Ve sonrisa y animación alegre
```

**Tiempo Total:** ~3-5 segundos

---

## 🔧 Tecnologías y Justificación

### **¿Por qué React + Three.js?**
- **React**: Gestión de estado eficiente, componentes reutilizables
- **Three.js**: Estándar de facto para 3D en web
- **React Three Fiber**: Integración declarativa de Three.js con React

### **¿Por qué Node.js en el Backend?**
- Ecosistema maduro para APIs
- Fácil integración con servicios externos
- Asíncrono (ideal para múltiples llamadas API)
- Compatible con FFmpeg y Rhubarb

### **¿Por qué Python para MIA Service?**
- TensorFlow/Keras para modelos de IA
- Transformers de Hugging Face
- Ecosistema de ML más maduro

### **¿Por qué Separar Backend y MIA Service?**
- **Escalabilidad**: Pueden correr en servidores diferentes
- **Mantenimiento**: Código más modular
- **Performance**: Node.js para I/O, Python para ML

### **¿Por qué BERT?**
- Pre-entrenado en español
- Alta precisión en clasificación de texto
- Relativamente ligero (~110M parámetros)

### **¿Por qué Rhubarb?**
- Open source y gratuito
- Alta precisión en lip-sync
- Multiplataforma
- No requiere GPU

---

## 🎯 Decisiones de Diseño

### **1. Lip-sync en Backend vs Frontend**
**Decisión:** Backend

**Razones:**
- Rhubarb es binario nativo (no disponible en browser)
- Procesamiento más rápido en servidor
- No consume recursos del cliente

### **2. Audio Base64 vs Stream**
**Decisión:** Base64

**Razones:**
- Más simple de implementar
- Compatible con todos los navegadores
- No requiere servidor de archivos estático

**Trade-off:** Mayor uso de ancho de banda

### **3. Modelos Locales vs API**
**Decisión:** Híbrido (BERT local, LLM en API)

**Razones:**
- Modelos pequeños (BERT) → Local para baja latencia
- Modelos grandes (Llama 3.1) → API por recursos

### **4. Limpieza de Archivos Temporales**
**Decisión:** Inmediata después de uso

**Razones:**
- Evita acumulación de archivos
- Reduce uso de disco
- Mejor para privacidad

### **5. Fallback a Mock**
**Decisión:** Siempre tener datos mock

**Razones:**
- Sistema funciona aunque servicios externos fallen
- Mejor experiencia de desarrollo
- Permite demo sin API keys

---

## 📈 Escalabilidad

### **Bottlenecks Actuales:**

1. **Groq API**: 30 req/min en free tier
   - **Solución**: Implementar rate limiting, cola de requests
   
2. **ElevenLabs**: 10k caracteres/mes gratis
   - **Solución**: Cachear audios comunes, plan pago
   
3. **Rhubarb**: Proceso síncrono
   - **Solución**: Cola de trabajos, workers paralelos

### **Mejoras Futuras:**

```
┌──────────────┐
│ Load Balancer│
└──────┬───────┘
       │
       ├─> Backend Instance 1 ──> MIA Service Instance 1
       ├─> Backend Instance 2 ──> MIA Service Instance 2
       └─> Backend Instance 3 ──> MIA Service Instance 3
                                  │
                                  └─> Redis Cache
                                  └─> Message Queue
```

**Optimizaciones:**
- Redis para cachear respuestas frecuentes
- Message queue (RabbitMQ) para lip-sync asíncrono
- CDN para assets estáticos
- WebSocket para latencia menor

---

## 🔐 Seguridad

### **Consideraciones Actuales:**

1. **API Keys en Backend**: ✅ Correcto
   - No expuestas al frontend
   - En variables de entorno

2. **CORS**: ✅ Configurado
   - Permite solo origins específicos en producción

3. **Rate Limiting**: ⚠️ Por implementar
   - Prevenir abuso de APIs

4. **Input Sanitization**: ⚠️ Por implementar
   - Validar inputs del usuario
   - Prevenir inyección de código

### **Mejoras Recomendadas:**

```javascript
// Rate limiting
const rateLimit = require('express-rate-limit');
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutos
  max: 100 // 100 requests por ventana
});
app.use('/chat', limiter);

// Input validation
const { body, validationResult } = require('express-validator');
app.post('/chat', [
  body('message').isLength({ min: 1, max: 500 }).trim().escape()
], async (req, res) => {
  // ...
});
```

---

## 📚 Referencias

- [React Three Fiber Docs](https://docs.pmnd.rs/react-three-fiber)
- [Three.js Manual](https://threejs.org/manual/)
- [Express Best Practices](https://expressjs.com/en/advanced/best-practice-performance.html)
- [Flask Patterns](https://flask.palletsprojects.com/en/2.3.x/patterns/)
- [BERT Paper](https://arxiv.org/abs/1810.04805)

---

**Última actualización:** Noviembre 2024
