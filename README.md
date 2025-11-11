# 💖 MIA - Compañera Virtual Empática

**MIA** (Mi Inteligencia Artificial) es un agente conversacional empático en español que combina:
- 🧠 Detección de emociones con IA (BERT)
- 💬 Generación de respuestas contextuales (Groq/Llama)
- 🎤 Síntesis de voz natural (ElevenLabs)
- 👄 Lip-sync realista (Rhubarb)
- 🎨 Avatar 3D animado (Three.js)

---

## 🎬 Demo

*(Agrega aquí un video o GIF de tu proyecto funcionando)*

---

## ✨ Características

### 🧠 **Detección de Emociones**
- **MiaMotion**: Detecta 6 emociones del usuario (alegría, amor, tristeza, ira, miedo, sorpresa)
- **MiaPredict**: Predice respuesta empática del agente (2 clases: alegría, amor)
- Modelos BERT fine-tuned en español

### 💬 **Respuestas Inteligentes**
- Generación con Groq API (Llama 3.1 8B)
- Respuestas adaptadas a la emoción detectada
- Modo predefinido como fallback

### 🎤 **Voz Natural**
- Síntesis con ElevenLabs
- Voces en español de alta calidad
- Audio sincronizado con labios

### 👄 **Lip-sync Preciso**
- Rhubarb Lip Sync para análisis de fonemas
- 8 visemas diferentes
- Sincronización palabra por palabra

### 🎨 **Avatar 3D**
- Modelo de Ready Player Me
- 9 animaciones corporales
- 6 expresiones faciales
- Renderizado en tiempo real

---

## 🏗️ Arquitectura

```
Usuario → Frontend (React + Three.js)
            ↓
        Backend (Node.js)
            ↓
        MIA Service (Python + BERT)
            ↓
        APIs Externas (Groq, ElevenLabs, Rhubarb)
```

**Stack Técnico:**
- **Frontend**: React 18, Three.js, React Three Fiber, Vite
- **Backend**: Node.js 18, Express.js
- **IA**: Python 3.8+, TensorFlow, BERT (Hugging Face)
- **APIs**: Groq (LLM), ElevenLabs (TTS), Rhubarb (Lip-sync)

---

## 📋 Requisitos

### Software
- Node.js 18+
- Python 3.8+
- FFmpeg
- Rhubarb Lip Sync 1.13.0

### API Keys (Gratuitas)
- **Groq**: https://console.groq.com
- **ElevenLabs**: https://elevenlabs.io (10k chars gratis/mes)

---

## 🚀 Instalación

### 1. Clonar Repositorio
```bash
git clone https://github.com/tu-usuario/mia-project.git
cd mia-project
```

### 2. Instalar FFmpeg

**Windows:**
```powershell
# Opción 1: Chocolatey
choco install ffmpeg

# Opción 2: Descargar desde ffmpeg.org y agregar al PATH
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

### 3. Instalar Rhubarb

**Windows:**
1. Descargar: https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.13.0/rhubarb-lip-sync-1.13.0-win32.zip
2. Extraer `rhubarb.exe`
3. Colocar en: `backend/bin/rhubarb.exe`

**macOS:**
```bash
brew install rhubarb-lip-sync
```

**Linux:**
```bash
wget https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.13.0/rhubarb-lip-sync-1.13.0-linux.zip
unzip rhubarb-lip-sync-1.13.0-linux.zip
sudo cp rhubarb /usr/local/bin/
```

### 4. Instalar Dependencias

**Backend (Node.js):**
```bash
cd backend
npm install
```

**MIA Service (Python):**
```bash
cd backend  # o cd mia-service si está separado
pip install -r requirements.txt

# Con virtual environment (recomendado):
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

---

## ⚙️ Configuración

### 1. Variables de Entorno

```bash
cd backend
cp .env.example .env
```

Editar `backend/.env`:
```env
PORT=3000
MIA_SERVICE_URL=http://localhost:5000
TEXT_MODE=groq

# Obtener en: https://console.groq.com
GROQ_API_KEY=tu_groq_api_key_aqui

# Obtener en: https://elevenlabs.io
ELEVEN_LABS_API_KEY=tu_elevenlabs_api_key_aqui

# Voice ID (Bella - español femenino)
ELEVEN_LABS_VOICE_ID=EXAVITQu4vr4xnSDxMaL
```

### 2. Verificar Modelos de IA

Los modelos deben estar en:
```
backend/models/
├── MiaMotion.pt     ← Detección de emoción
└── MiaPredict.pt    ← Predicción de respuesta
```

O en:
```
mia-service/
├── MiaMotion.pt
└── MiaPredict.pt
```

---

## 🎮 Uso

### Iniciar Todo (3 terminales)

**Terminal 1 - MIA Service:**
```bash
cd backend  # o cd mia-service
python mia_service.py
# Debe mostrar: * Running on http://localhost:5000
```

**Terminal 2 - Backend:**
```bash
cd backend
npm start
# Debe mostrar: 🚀 MIA Backend Server iniciado
```

**Terminal 3 - Frontend:**
```bash
cd frontend
npm run dev
# Debe mostrar: Local: http://localhost:5173
```

### Acceder
1. Abrir: http://localhost:5173
2. Escribir mensaje: "Hola, ¿cómo estás?"
3. Ver respuesta del avatar con voz

### Script Automático (Windows)
```bash
start-dev.bat
```

---

## 🛠️ Tecnologías

### Frontend
- React 18
- Three.js / React Three Fiber
- Vite
- Axios

### Backend
- Node.js 18
- Express.js
- Axios

### MIA Service
- Python 3.8+
- Flask
- TensorFlow/Keras
- Transformers (Hugging Face)
- BERT

### APIs Externas
- Groq (Llama 3.1)
- ElevenLabs (TTS)
- Rhubarb (Lip-sync)

---

## 🗺️ Roadmap

### ✅ Completado
- [x] Detección de emociones
- [x] Generación de texto
- [x] Síntesis de voz
- [x] Lip-sync realista
- [x] Avatar 3D animado

### 🚧 En Desarrollo
- [ ] Memoria de conversación
- [ ] Más emociones (6 clases completas)
- [ ] Reconocimiento de voz

### 💡 Futuro
- [ ] Modo voz continua
- [ ] Personalización de avatar
- [ ] App móvil
- [ ] Múltiples idiomas

---

## ⚠️ Problemas Comunes

### Windows: Rhubarb no se ejecuta
```
Solución:
1. Verificar que rhubarb.exe está en: backend/bin/rhubarb.exe
2. Si Windows Defender bloquea, permitir ejecución
```

### FFmpeg no encontrado
```
Solución:
1. Instalar FFmpeg
2. Agregar al PATH del sistema
3. Reiniciar terminal
4. Verificar: ffmpeg -version
```

### Lip-sync usa mock
```
Verificar:
1. Rhubarb instalado: rhubarb --version
2. FFmpeg instalado: ffmpeg -version
3. Ver logs del backend para error específico
```

### Error en modelos de IA
```
Verificar:
1. Archivos MiaMotion.h5 y MiaPredict.h5 existen
2. Están en backend/models/ o mia-service/models/
3. Python tiene permisos de lectura
```

---

## 🤝 Contribución

¡Las contribuciones son bienvenidas!

1. Fork el proyecto
2. Crear rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -m 'feat: agregar nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Abrir Pull Request

Ver [CONTRIBUTING.md](CONTRIBUTING.md) para más detalles.

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](LICENSE).

---

## 👏 Agradecimientos

- Ready Player Me - Avatar 3D
- Mixamo - Animaciones
- Groq - API LLM gratuita
- ElevenLabs - TTS de calidad
- Rhubarb Lip Sync - Open source
- Hugging Face - Modelos BERT

---

<div align="center">

**Hecho con ❤️ en Perú**

[⬆ Volver arriba](#-mia---compañera-virtual-empática)

</div>
