# 🤝 Guía de Contribución - MIA Project

¡Gracias por tu interés en contribuir a MIA! Esta guía te ayudará a empezar.

---

## 📋 Tabla de Contenidos

- [Código de Conducta](#código-de-conducta)
- [¿Cómo Puedo Contribuir?](#cómo-puedo-contribuir)
- [Configuración del Entorno](#configuración-del-entorno)
- [Flujo de Trabajo](#flujo-de-trabajo)
- [Guías de Estilo](#guías-de-estilo)
- [Estructura del Proyecto](#estructura-del-proyecto)

---

## 📜 Código de Conducta

Este proyecto sigue un código de conducta. Al participar, se espera que mantengas un comportamiento respetuoso y profesional.

---

## 💡 ¿Cómo Puedo Contribuir?

### **Reportar Bugs**

¿Encontraste un bug? Ayúdanos creando un **Issue** con:

1. **Título claro**: "Bug: Avatar no se mueve en Windows"
2. **Descripción**: Qué esperabas vs. qué pasó
3. **Pasos para reproducir**:
   ```
   1. Abrir la aplicación
   2. Enviar mensaje "Hola"
   3. El avatar no responde
   ```
4. **Sistema**: Windows 11, Node.js 18.0.0, Python 3.9
5. **Logs**: Incluir logs del backend/frontend si es posible

### **Sugerir Mejoras**

¿Tienes una idea? Abre un **Issue** con:
- Etiqueta: `enhancement`
- Descripción clara de la mejora
- Beneficio que aporta
- Implementación sugerida (opcional)

### **Contribuir Código**

1. Busca un issue con etiqueta `good first issue` o `help wanted`
2. Comenta que quieres trabajar en ello
3. Sigue el [flujo de trabajo](#flujo-de-trabajo)

---

## 🛠️ Configuración del Entorno

### **1. Fork y Clone**

```bash
# Fork en GitHub (botón Fork)
git clone https://github.com/TU-USUARIO/mia-project.git
cd mia-project
```

### **2. Configurar Upstream**

```bash
git remote add upstream https://github.com/REPO-ORIGINAL/mia-project.git
git fetch upstream
```

### **3. Instalar Dependencias**

Ver [README.md - Instalación](README.md#instalación)

### **4. Configurar Pre-commit Hooks (opcional)**

```bash
# Instalar pre-commit
pip install pre-commit

# Configurar hooks
pre-commit install
```

---

## 🔄 Flujo de Trabajo

### **1. Crear Rama**

```bash
# Actualizar main
git checkout main
git pull upstream main

# Crear rama descriptiva
git checkout -b feature/add-emotion-detection
# o
git checkout -b fix/lip-sync-windows
# o
git checkout -b docs/improve-readme
```

**Convenciones de nombres de ramas:**
- `feature/` - Nueva funcionalidad
- `fix/` - Corrección de bug
- `docs/` - Documentación
- `refactor/` - Refactorización de código
- `test/` - Agregar tests

### **2. Hacer Cambios**

```bash
# Hacer cambios en el código
# ...

# Agregar archivos modificados
git add .

# Commit con mensaje descriptivo
git commit -m "feat: agregar detección de emoción 'sorpresa'"
```

**Formato de mensajes de commit:**
```
tipo: descripción corta

[Descripción larga opcional]

[Footer opcional: referencias a issues]
```

**Tipos de commit:**
- `feat`: Nueva característica
- `fix`: Corrección de bug
- `docs`: Cambios en documentación
- `style`: Formato, punto y coma faltante, etc.
- `refactor`: Refactorización de código
- `test`: Agregar tests
- `chore`: Cambios en build, herramientas, etc.

**Ejemplos:**
```bash
git commit -m "feat: agregar reconocimiento de voz con Web Speech API"
git commit -m "fix: corregir lip-sync en Windows 11"
git commit -m "docs: actualizar README con instrucciones de macOS"
git commit -m "refactor: simplificar lógica de emociones en backend"
```

### **3. Push a Tu Fork**

```bash
git push origin feature/add-emotion-detection
```

### **4. Crear Pull Request**

1. Ir a tu fork en GitHub
2. Click en "Compare & pull request"
3. Llenar el template:

```markdown
## Descripción
Breve descripción de los cambios

## Tipo de Cambio
- [ ] Bug fix
- [ ] Nueva característica
- [ ] Breaking change
- [ ] Documentación

## Checklist
- [ ] Mi código sigue el estilo del proyecto
- [ ] He comentado el código en áreas difíciles
- [ ] He actualizado la documentación
- [ ] Mis cambios no generan nuevos warnings
- [ ] He probado localmente que funciona

## Screenshots (si aplica)
[Agregar capturas de pantalla]
```

---

## 🎨 Guías de Estilo

### **Python (Backend - MIA Service)**

Seguir **PEP 8**:

```python
# ✅ CORRECTO
def detect_emotion(text):
    """
    Detecta la emoción en un texto.
    
    Args:
        text (str): Texto a analizar
        
    Returns:
        dict: Emoción y confianza
    """
    emotion = model.predict(text)
    return {
        "emotion": emotion,
        "confidence": 0.95
    }

# ❌ INCORRECTO
def detectEmotion(text):
    emotion=model.predict(text)
    return {"emotion":emotion,"confidence":0.95}
```

**Reglas:**
- Nombres de funciones: `snake_case`
- Nombres de clases: `PascalCase`
- Constantes: `UPPER_SNAKE_CASE`
- 4 espacios de indentación
- Docstrings en todas las funciones
- Máximo 79 caracteres por línea

### **JavaScript/Node.js (Backend)**

Seguir **JavaScript Standard Style**:

```javascript
// ✅ CORRECTO
async function generateTextResponse(userMessage, emotion) {
  try {
    const response = await groqAPI.generate({
      prompt: userMessage,
      emotion: emotion
    });
    return response.text;
  } catch (error) {
    console.error('Error generando respuesta:', error);
    throw error;
  }
}

// ❌ INCORRECTO
async function generateTextResponse(userMessage,emotion){
try{
const response=await groqAPI.generate({prompt:userMessage,emotion:emotion})
return response.text
}catch(error){console.error('Error',error);throw error}}
```

**Reglas:**
- Nombres de funciones y variables: `camelCase`
- Nombres de clases: `PascalCase`
- Constantes: `UPPER_SNAKE_CASE`
- 2 espacios de indentación
- Punto y coma al final de cada statement
- Template literals en lugar de concatenación
- async/await en lugar de .then()

### **React (Frontend)**

```jsx
// ✅ CORRECTO
export function Avatar({ message, onComplete }) {
  const [isPlaying, setIsPlaying] = useState(false);
  
  useEffect(() => {
    if (message?.audio) {
      playAudio(message.audio);
    }
  }, [message]);
  
  return (
    <group>
      <primitive object={nodes.Head} />
    </group>
  );
}

// ❌ INCORRECTO
export function avatar(props) {
  var isPlaying = false;
  
  if (props.message.audio) {
    playAudio(props.message.audio);
  }
  
  return <group><primitive object={nodes.Head}/></group>;
}
```

**Reglas:**
- Componentes: `PascalCase`
- Hooks personalizados: `useCamelCase`
- Props destructuring
- Funciones arrow en components
- JSX con 2 espacios de indentación

### **Commits**

```bash
# ✅ CORRECTO
git commit -m "feat: agregar detección de emoción 'miedo'"
git commit -m "fix: corregir sincronización de labios en Windows"
git commit -m "docs: actualizar README con troubleshooting"

# ❌ INCORRECTO
git commit -m "cambios"
git commit -m "fix bug"
git commit -m "update"
```

---

## 📁 Estructura del Proyecto

```
mia-project/
├── backend/                    # Backend Node.js
│   ├── bin/                    # Binarios (Rhubarb)
│   │   └── rhubarb.exe
│   ├── models/                 # Modelos de IA
│   │   ├── MiaMotion.h5
│   │   └── MiaPredict.h5
│   ├── audios/                 # Archivos temporales de audio
│   ├── index.js                # Servidor principal
│   ├── mia_service.py          # Servicio de detección de emociones
│   ├── requirements.txt        # Dependencias Python
│   ├── package.json            # Dependencias Node.js
│   └── .env                    # Variables de entorno (NO subir)
│
├── frontend/                   # Frontend React
│   ├── public/                 # Assets estáticos
│   │   └── models/             # Modelos 3D
│   ├── src/
│   │   ├── components/         # Componentes React
│   │   │   ├── Avatar.jsx      # Avatar 3D
│   │   │   ├── Experience.jsx  # Escena 3D
│   │   │   └── UI.jsx          # Interfaz de usuario
│   │   ├── hooks/              # Custom hooks
│   │   │   └── useChat.jsx     # Lógica de chat
│   │   ├── App.jsx             # Componente raíz
│   │   └── main.jsx            # Entry point
│   ├── package.json
│   └── vite.config.js
│
├── docs/                       # Documentación adicional
│   ├── ARCHITECTURE.md         # Arquitectura del sistema
│   ├── API.md                  # Documentación de API
│   └── DEPLOYMENT.md           # Guía de deployment
│
├── .gitignore                  # Archivos a ignorar por Git
├── .env.example                # Template de variables de entorno
├── README.md                   # Documentación principal
├── CONTRIBUTING.md             # Esta guía
├── LICENSE                     # Licencia del proyecto
└── package.json                # Scripts del proyecto
```

### **Dónde Agregar Nuevas Funcionalidades**

| Funcionalidad | Archivo | Ubicación |
|---------------|---------|-----------|
| Nueva emoción en detección | `mia_service.py` | Función `predict_emotion()` |
| Nueva animación de avatar | `Avatar.jsx` | Array `animations` |
| Nueva expresión facial | `Avatar.jsx` | Object `facialExpressions` |
| Nuevo endpoint de API | `index.js` | Después de `/chat` |
| Nueva integración externa | `index.js` | Función auxiliar nueva |
| Nuevo componente UI | `frontend/src/components/` | Nuevo archivo `.jsx` |

---

## 🧪 Testing

### **Backend (Node.js)**

```bash
cd backend
npm test
```

### **Frontend (React)**

```bash
cd frontend
npm test
```

### **MIA Service (Python)**

```bash
cd backend
pytest
```

---

## 🎓 Recursos Útiles

### **Documentación**
- [React Three Fiber](https://docs.pmnd.rs/react-three-fiber)
- [Three.js](https://threejs.org/docs/)
- [Express.js](https://expressjs.com/)
- [Flask](https://flask.palletsprojects.com/)

### **APIs**
- [Groq API Docs](https://console.groq.com/docs)
- [ElevenLabs API Docs](https://elevenlabs.io/docs)
- [Rhubarb Lip Sync](https://github.com/DanielSWolf/rhubarb-lip-sync)

---

## ❓ ¿Necesitas Ayuda?

- **Issues**: Busca issues similares o crea uno nuevo
- **Discussions**: Para preguntas generales
- **Discord**: [Link al servidor] (si existe)

---

¡Gracias por contribuir a MIA! 💖
