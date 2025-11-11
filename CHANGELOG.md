# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 🎉 Planeado
- Memoria de conversación
- Reconocimiento de voz (Speech-to-Text)
- Más emociones en MiaPredict (6 clases)
- Personalización del avatar
- App móvil

---

## [1.0.0] - 2024-11-XX

### 🎉 Primera Versión Estable

#### ✨ Agregado
- **Sistema completo de detección de emociones**
  - MiaMotion: Detecta emoción del usuario (6 emociones)
  - MiaPredict: Predice respuesta empática del agente (2 emociones)
  - Modelos BERT fine-tuned en español

- **Generación de respuestas con IA**
  - Integración con Groq API (Llama 3.1 8B)
  - Modo de respuestas predefinidas como fallback
  - Instrucciones empáticas adaptadas a la emoción

- **Síntesis de voz natural**
  - Integración con ElevenLabs API
  - Voces en español de alta calidad
  - Audio en formato MP3 convertido a WAV

- **Lip-sync realista**
  - Integración con Rhubarb Lip Sync
  - 8 visemas diferentes
  - Sincronización precisa palabra por palabra
  - Soporte para Windows, macOS y Linux

- **Avatar 3D interactivo**
  - Modelo de Ready Player Me
  - 9 animaciones (Idle, Talking, Laughing, Crying, Angry, etc.)
  - 6 expresiones faciales (smile, sad, angry, surprised, etc.)
  - Renderizado con Three.js y React Three Fiber

- **Backend robusto**
  - Servidor Express.js
  - Pipeline completo de procesamiento
  - Manejo de errores con fallbacks
  - Limpieza automática de archivos temporales

- **Frontend moderno**
  - Interfaz React con Vite
  - UI de chat intuitiva
  - Reproducción de audio sincronizada
  - Animaciones fluidas

#### 🐛 Fixed
- Lip-sync no funcionaba en Windows (fix: usar archivo temporal en lugar de stdout)
- Audio MP3 incompatible con Rhubarb (fix: conversión a WAV con FFmpeg)
- Errores de path en Windows vs Unix (fix: detección de OS y paths dinámicos)
- Archivos temporales acumulándose (fix: limpieza automática después de uso)

#### 🔧 Changed
- Optimización de conversión de audio
- Mejora en logs de debug
- Estructura de respuesta más clara

#### 📚 Documentación
- README completo con instrucciones de instalación
- ARCHITECTURE.md con diagrama y explicaciones técnicas
- DEPLOYMENT.md con guías para múltiples plataformas
- CONTRIBUTING.md con guías de estilo
- Templates para Issues y PRs

---

## [0.2.0] - 2024-11-XX (Beta)

### ✨ Agregado
- Integración de Rhubarb Lip Sync
- Conversión automática MP3 a WAV
- Fallback a lip-sync mock cuando Rhubarb falla

### 🐛 Fixed
- Lip-sync generaba datos vacíos
- Rhubarb no se ejecutaba correctamente en Windows

---

## [0.1.0] - 2024-11-XX (Alpha)

### 🎉 Primera Versión

#### ✨ Agregado
- Backend básico con Express.js
- MIA Service con modelos BERT
- Frontend con avatar 3D
- Integración con Groq y ElevenLabs
- Animaciones básicas del avatar

---

## Formato de Versionado

- **MAJOR** (X.0.0): Cambios incompatibles con API anterior
- **MINOR** (0.X.0): Nuevas funcionalidades compatibles
- **PATCH** (0.0.X): Bug fixes compatibles

---

## Tipos de Cambios

- `Added` - Nuevas funcionalidades
- `Changed` - Cambios en funcionalidades existentes
- `Deprecated` - Funcionalidades que se eliminarán pronto
- `Removed` - Funcionalidades eliminadas
- `Fixed` - Corrección de bugs
- `Security` - Parches de seguridad

---

[Unreleased]: https://github.com/tu-usuario/mia-project/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/tu-usuario/mia-project/releases/tag/v1.0.0
[0.2.0]: https://github.com/tu-usuario/mia-project/releases/tag/v0.2.0
[0.1.0]: https://github.com/tu-usuario/mia-project/releases/tag/v0.1.0
