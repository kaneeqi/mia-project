// ==================== BACKEND MIA - VERSIÓN WINDOWS CORREGIDA ====================
// Solución: Rhubarb en carpeta bin/ para Windows

import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import axios from "axios";
import { exec } from "child_process";
import { writeFile, unlink, access } from "fs/promises";
import { existsSync } from "fs";
import { v4 as uuidv4 } from "uuid";
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { promisify } from 'util';

const execPromise = promisify(exec);

dotenv.config();

const app = express();
app.use(express.json());
app.use(cors());

const port = process.env.PORT || 3000;
const __dirname = dirname(fileURLToPath(import.meta.url));

// ==================== CONFIGURACIÓN ====================
const elevenLabsApiKey = process.env.ELEVEN_LABS_API_KEY;
const voiceID = process.env.ELEVEN_LABS_VOICE_ID || "EXAVITQu4vr4xnSDxMaL";
const MIA_SERVICE_URL = process.env.MIA_SERVICE_URL || "http://localhost:5000";
const TEXT_MODE = process.env.TEXT_MODE || "groq";

// ==================== RESPUESTAS PREDEFINIDAS ====================
const PREDEFINED_RESPONSES = {
  "alegría": [
    "¡Qué bueno escuchar eso! Me alegra mucho que estés feliz.",
    "¡Me encanta tu energía! Cuéntame más sobre lo que te hace sentir así.",
    "¡Eso suena maravilloso! Tu felicidad es contagiosa.",
  ],
  "amor": [
    "Estoy aquí para ti, siempre. ¿Quieres hablar sobre cómo te sientes?",
    "Lamento que estés pasando por esto. No estás solo, estoy aquí contigo.",
    "Comprendo cómo te sientes. A veces las cosas son difíciles, pero siempre hay esperanza.",
    "Me duele saber que estás sufriendo. Por favor, cuéntame qué está pasando.",
  ]
};

// ==================== FUNCIONES AUXILIARES ====================

async function getMiaEmotions(userMessage) {
  try {
    console.log("🤖 Llamando a MIA service...");
    const response = await axios.post(`${MIA_SERVICE_URL}/full_pipeline`, {
      text: userMessage
    }, { timeout: 10000 });
    
    console.log("✅ MIA response:", response.data);
    return response.data;
  } catch (error) {
    console.error("❌ Error llamando a MIA service:", error.message);
    return {
      user: { emotion: "neutral", label: 0, confidence: 0.5 },
      agent: { emotion: "amor", label: 2, confidence: 0.5, facial_expression: "smile" }
    };
  }
}

async function generateTextResponse(userMessage, agentEmotion) {
  console.log(`💬 Generando respuesta (modo: ${TEXT_MODE})...`);
  
  try {
    switch (TEXT_MODE) {
      case "groq":
        return await generateTextWithGroq(userMessage, agentEmotion);
      case "predefined":
      default:
        return generatePredefinedResponse(agentEmotion);
    }
  } catch (error) {
    console.error("❌ Error generando texto:", error.message);
    return "Estoy aquí para ti. ¿Cómo puedo ayudarte?";
  }
}

function generatePredefinedResponse(agentEmotion) {
  const responses = PREDEFINED_RESPONSES[agentEmotion] || PREDEFINED_RESPONSES["amor"];
  const randomResponse = responses[Math.floor(Math.random() * responses.length)];
  console.log("✅ Respuesta predefinida seleccionada");
  return randomResponse;
}

async function generateTextWithGroq(userMessage, agentEmotion) {
  const groqApiKey = process.env.GROQ_API_KEY;
  if (!groqApiKey) {
    throw new Error("GROQ_API_KEY no configurada");
  }
  
  const emotionInstructions = {
    "alegría": "Responde con entusiasmo y positividad.",
    "amor": "Responde con calidez, empatía y ternura."
  };
  
  const instruction = emotionInstructions[agentEmotion] || "Responde con empatía.";
  
  const response = await axios.post(
    "https://api.groq.com/openai/v1/chat/completions",
    {
      model: "llama-3.1-8b-instant",
      messages: [
        {
          role: "system",
          content: `Eres MIA, una compañera virtual empática en español. ${instruction} Responde en máximo 2 oraciones cortas. Sé natural y cálida.`
        },
        { role: "user", content: userMessage }
      ],
      max_tokens: 100,
      temperature: 0.7
    },
    {
      headers: {
        "Authorization": `Bearer ${groqApiKey}`,
        "Content-Type": "application/json"
      },
      timeout: 10000
    }
  );
  
  return response.data.choices[0].message.content.trim();
}

async function generateAudio(text) {
  if (!elevenLabsApiKey) {
    console.log("⚠️  ElevenLabs no configurado, sin audio");
    return null;
  }
  
  try {
    console.log("🎤 Generando audio con ElevenLabs...");
    
    const response = await axios.post(
      `https://api.elevenlabs.io/v1/text-to-speech/${voiceID}`,
      {
        text: text,
        model_id: "eleven_multilingual_v2",
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.75,
        },
      },
      {
        headers: {
          "xi-api-key": elevenLabsApiKey,
          "Content-Type": "application/json",
          Accept: "audio/mpeg",
        },
        responseType: "arraybuffer",
        timeout: 15000
      }
    );
    
    console.log("✅ Audio generado");
    return Buffer.from(response.data).toString("base64");
    
  } catch (error) {
    console.error("❌ Error generando audio:", error.message);
    return null;
  }
}

async function convertMp3ToWav(mp3Path) {
  const wavPath = mp3Path.replace('.mp3', '.wav');
  
  try {
    console.log("🔄 Convirtiendo MP3 a WAV...");
    // Usar ffmpeg para convertir MP3 a WAV con formato compatible con Rhubarb
    const command = `ffmpeg -i "${mp3Path}" -ar 16000 -ac 1 -y "${wavPath}"`;
    await execPromise(command);
    console.log("   ✅ Conversión completada");
    return wavPath;
  } catch (error) {
    console.error("❌ Error convirtiendo audio:", error.message);
    console.log("⚠️  Asegúrate de tener ffmpeg instalado:");
    console.log("   Windows: choco install ffmpeg (o descargar desde ffmpeg.org)");
    console.log("   macOS: brew install ffmpeg");
    throw error;
  }
}

async function generateLipSync(audioBase64) {
  if (!audioBase64) {
    console.log("⚠️  Sin audio, usando lip-sync mockeado");
    return mockLipSync();
  }
  
  try {
    console.log("👄 Generando lip-sync con Rhubarb...");
    
    const audioFilename = join(__dirname, `audio_${uuidv4()}.mp3`);
    const audioBuffer = Buffer.from(audioBase64, "base64");
    await writeFile(audioFilename, audioBuffer);
    
    // Convertir MP3 a WAV
    let wavFilename;
    try {
      wavFilename = await convertMp3ToWav(audioFilename);
    } catch (conversionError) {
      console.log("⚠️  No se pudo convertir a WAV, usando lip-sync mockeado");
      await unlink(audioFilename).catch(() => {});
      return mockLipSync();
    }
    
    // Buscar Rhubarb
    const isWindows = process.platform === "win32";
    let rhubarbPath;
    
    if (isWindows) {
      // En Windows, usar el binario local
      rhubarbPath = join(__dirname, 'bin', 'rhubarb.exe');
      console.log(`   🔧 Buscando Rhubarb en: ${rhubarbPath}`);
      
      // Verificar que existe
      if (!existsSync(rhubarbPath)) {
        console.log(`   ❌ Rhubarb NO encontrado`);
        console.log(`   📥 Descargar desde: https://github.com/DanielSWolf/rhubarb-lip-sync/releases`);
        console.log(`   📂 Colocar en: ${rhubarbPath}`);
        console.log(`   ⚠️  Usando lip-sync mockeado`);
        
        // Limpiar y usar mock
        await unlink(audioFilename).catch(() => {});
        await unlink(wavFilename).catch(() => {});
        return mockLipSync();
      }
      
      console.log(`   ✅ Rhubarb encontrado`);
    } else {
      // En macOS/Linux, buscar en el sistema
      try {
        const { stdout } = await execPromise('which rhubarb');
        rhubarbPath = stdout.trim();
        console.log(`   ✅ Rhubarb encontrado en: ${rhubarbPath}`);
      } catch (error) {
        console.log(`   ❌ Rhubarb no instalado`);
        console.log(`   📥 Instalar: brew install rhubarb-lip-sync`);
        
        await unlink(audioFilename).catch(() => {});
        await unlink(wavFilename).catch(() => {});
        return mockLipSync();
      }
    }
    
  return new Promise((resolve, reject) => {
    // En Windows, guardar en archivo temporal en lugar de stdout
    const jsonFilename = wavFilename.replace('.wav', '.json');
    const command = isWindows 
      ? `"${rhubarbPath}" -f json "${wavFilename}" -o "${jsonFilename}"`
      : `"${rhubarbPath}" -f json -o - "${wavFilename}"`;
    
    console.log(`   🔧 Ejecutando Rhubarb...`);
    
    exec(command, { timeout: 15000 }, async (error, stdout, stderr) => {
      // Limpiar archivos temporales
      await unlink(audioFilename).catch(() => {});
      await unlink(wavFilename).catch(() => {});
      
      if (error) {
        console.error("❌ Error en Rhubarb:", error.message);
        if (stderr) console.error("   Stderr:", stderr);
        console.log("⚠️  Usando lip-sync mockeado como fallback");
        await unlink(jsonFilename).catch(() => {});
        resolve(mockLipSync());
        return;
      }
      
      try {
        let lipsyncData;
        
        if (isWindows) {
          // En Windows, leer desde archivo
          const { readFile } = await import('fs/promises');
          const jsonContent = await readFile(jsonFilename, 'utf8');
          lipsyncData = JSON.parse(jsonContent);
          await unlink(jsonFilename).catch(() => {});
        } else {
          // En Unix, usar stdout
          lipsyncData = JSON.parse(stdout);
        }
        
        console.log("✅ Lip-sync generado correctamente");
        console.log(`   📊 Duración: ${lipsyncData.metadata.duration}s`);
        console.log(`   👄 Cues: ${lipsyncData.mouthCues.length}`);
        resolve(lipsyncData);
      } catch (parseError) {
        console.error("❌ Error parseando lip-sync:", parseError.message);
        console.log("   Stdout recibido:", stdout.substring(0, 100));
        await unlink(jsonFilename).catch(() => {});
        resolve(mockLipSync());
      }
    });
  });
    
  } catch (error) {
    console.error("❌ Error generando lip-sync:", error.message);
    return mockLipSync();
  }
}

function mockLipSync() {
  console.log("🎭 Usando lip-sync mock (genérico)");
  return {
    metadata: {
      soundFile: "mock",
      duration: 3.0
    },
    mouthCues: [
      { start: 0.0, end: 0.2, value: "X" },
      { start: 0.2, end: 0.4, value: "B" },
      { start: 0.4, end: 0.6, value: "E" },
      { start: 0.6, end: 0.8, value: "C" },
      { start: 0.8, end: 1.0, value: "B" },
      { start: 1.0, end: 1.2, value: "F" },
      { start: 1.2, end: 1.4, value: "C" },
      { start: 1.4, end: 1.6, value: "D" },
      { start: 1.6, end: 1.8, value: "B" },
      { start: 1.8, end: 2.0, value: "E" },
      { start: 2.0, end: 2.2, value: "B" },
      { start: 2.2, end: 2.4, value: "C" },
      { start: 2.4, end: 2.6, value: "F" },
      { start: 2.6, end: 2.8, value: "B" },
      { start: 2.8, end: 3.0, value: "X" }
    ]
  };
}

function getAnimationForEmotion(agentEmotion) {
  // Solo 2 emociones: alegría y amor
  const animationMap = {
    "alegría": "Laughing",
    "amor": "Talking_1"
  };
  
  return animationMap[agentEmotion] || "Talking_1";
}

// ==================== ENDPOINT PRINCIPAL ====================

app.post("/chat", async (req, res) => {
  try {
    const { message, conversationHistory = [] } = req.body;
    
    if (!message) {
      return res.status(400).json({ error: "Campo 'message' requerido" });
    }
    
    console.log("\n" + "=".repeat(60));
    console.log(`💬 Mensaje del usuario: "${message}"`);
    console.log("=".repeat(60));
    
    // ============ PASO 1: Detectar emociones con MIA ============
    const miaEmotions = await getMiaEmotions(message);
    const { user: userEmotion, agent: agentEmotion } = miaEmotions;
    
    console.log(`😊 Emoción usuario: ${userEmotion.emotion} (${(userEmotion.confidence * 100).toFixed(0)}%)`);
    console.log(`💖 Emoción agente: ${agentEmotion.emotion} (${(agentEmotion.confidence * 100).toFixed(0)}%)`);
    
    // ============ PASO 2: Generar respuesta de texto ============
    const textResponse = await generateTextResponse(message, agentEmotion.emotion);
    console.log(`📝 Respuesta: "${textResponse}"`);
    
    // ============ PASO 3: Generar audio (opcional) ============
    const audioBase64 = await generateAudio(textResponse);
    
    // ============ PASO 4: Generar lip-sync ============
    const lipsyncData = await generateLipSync(audioBase64);
    
    // ============ PASO 5: Preparar respuesta ============
    const animation = getAnimationForEmotion(agentEmotion.emotion);
    
    const response = {
      messages: [
        { 
          text: message, 
          audio: null, 
          lipsync: null, 
          facialExpression: "default", 
          animation: "Idle" 
        },
        { 
          text: textResponse, 
          audio: audioBase64, 
          lipsync: lipsyncData,
          facialExpression: agentEmotion.facial_expression,
          animation: animation,
          emotions: {
            user: userEmotion,
            agent: agentEmotion
          }
        }
      ]
    };
    
    console.log("✅ Respuesta completa generada");
    console.log("=".repeat(60) + "\n");
    
    res.json(response);
    
  } catch (error) {
    console.error("❌ Error en /chat:", error);
    res.status(500).json({ 
      error: "Error procesando mensaje", 
      details: error.message 
    });
  }
});

app.get("/health", async (req, res) => {
  try {
    const miaHealth = await axios.get(`${MIA_SERVICE_URL}/health`, { timeout: 5000 });
    
    // Verificar herramientas
    const tools = {
      ffmpeg: false,
      rhubarb: false
    };
    
    try {
      await execPromise('ffmpeg -version');
      tools.ffmpeg = true;
    } catch {}
    
    const isWindows = process.platform === "win32";
    if (isWindows) {
      const rhubarbPath = join(__dirname, 'bin', 'rhubarb.exe');
      tools.rhubarb = existsSync(rhubarbPath);
    } else {
      try {
        await execPromise('rhubarb --version');
        tools.rhubarb = true;
      } catch {}
    }
    
    res.json({ 
      status: "healthy",
      services: {
        backend: "ok",
        mia: miaHealth.data,
        textMode: TEXT_MODE,
        elevenlabs: elevenLabsApiKey ? "configured" : "not configured"
      },
      tools
    });
  } catch (error) {
    res.status(503).json({ 
      status: "unhealthy",
      error: error.message 
    });
  }
});

// ==================== INICIAR SERVIDOR ====================
app.listen(port, async () => {
  console.log("\n" + "=".repeat(60));
  console.log("🚀 MIA Backend Server iniciado");
  console.log("=".repeat(60));
  console.log(`   📍 Puerto: ${port}`);
  console.log(`   🤖 MIA Service: ${MIA_SERVICE_URL}`);
  console.log(`   💬 Generación texto: ${TEXT_MODE}`);
  console.log(`   🎤 ElevenLabs: ${elevenLabsApiKey ? "✅ Configurado" : "⚠️  No configurado"}`);
  console.log(`   🖥️  Sistema: ${process.platform}`);
  
  // Verificar herramientas
  console.log("\n   Verificando herramientas...");
  
  try {
    await execPromise('ffmpeg -version');
    console.log("   ✅ ffmpeg instalado");
  } catch {
    console.log("   ❌ ffmpeg NO encontrado");
    console.log("      Descargar desde: https://ffmpeg.org/download.html");
  }
  
  const isWindows = process.platform === "win32";
  if (isWindows) {
    const rhubarbPath = join(__dirname, 'bin', 'rhubarb.exe');
    if (existsSync(rhubarbPath)) {
      console.log(`   ✅ rhubarb instalado en: ${rhubarbPath}`);
    } else {
      console.log("   ❌ rhubarb NO encontrado");
      console.log("      Descargar desde: https://github.com/DanielSWolf/rhubarb-lip-sync/releases");
      console.log(`      Colocar en: ${rhubarbPath}`);
    }
  } else {
    try {
      const { stdout } = await execPromise('which rhubarb');
      console.log(`   ✅ rhubarb instalado en: ${stdout.trim()}`);
    } catch {
      console.log("   ❌ rhubarb NO encontrado - instalar: brew install rhubarb-lip-sync");
    }
  }
  
  console.log("=".repeat(60) + "\n");
});