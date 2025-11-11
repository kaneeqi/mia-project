// frontend/src/hooks/useChat.jsx
import { createContext, useContext, useEffect, useState } from "react";

const backendUrl = import.meta.env.VITE_API_URL || "http://localhost:3000";

const ChatContext = createContext();

export const ChatProvider = ({ children }) => {
  const [messages, setMessages] = useState([]);
  const [message, setMessage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [cameraZoomed, setCameraZoomed] = useState(true);

  const chat = async (userMessage) => {
    if (!userMessage || loading) return;

    setLoading(true);

    try {
      console.log("📤 Enviando mensaje:", userMessage);

      const response = await fetch(`${backendUrl}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: userMessage }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("📦 Datos recibidos:", data);

      if (data.messages && Array.isArray(data.messages)) {
        setMessages((prevMessages) => [...prevMessages, ...data.messages]);

        // Reproducir audio si existe en el último mensaje
        const lastMessage = data.messages[data.messages.length - 1];
        if (lastMessage && lastMessage.audio) {
          playAudio(lastMessage.audio);
        }
      }
    } catch (error) {
      console.error("❌ Error en chat:", error);
      
      // Agregar mensaje de error
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          text: "Lo siento, tuve un problema. ¿Puedes intentar de nuevo?",
          audio: null,
          lipsync: null,
          facialExpression: "sad",
          animation: "Idle",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const playAudio = (audioBase64) => {
    try {
      console.log("🎵 Reproduciendo audio...");
      
      // Convertir base64 a blob
      const byteCharacters = atob(audioBase64);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const audioBlob = new Blob([byteArray], { type: "audio/mpeg" });
      
      // Crear URL y reproducir
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      
      audio.play().catch((error) => {
        console.error("❌ Error reproduciendo audio:", error);
      });
      
      // Limpiar URL después de reproducir
      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
      };
    } catch (error) {
      console.error("❌ Error procesando audio:", error);
    }
  };

  const onMessagePlayed = () => {
    console.log("✅ Mensaje reproducido, removiendo de la cola");
    setMessages((prevMessages) => prevMessages.slice(1));
  };

  // Actualizar message (singular) cuando cambian messages (plural)
  useEffect(() => {
    if (messages.length > 0) {
      setMessage(messages[0]);
    } else {
      setMessage(null);
    }
  }, [messages]);

  return (
    <ChatContext.Provider
      value={{
        chat,
        message,
        messages,
        onMessagePlayed,
        loading,
        cameraZoomed,
        setCameraZoomed,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
};

export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error("useChat must be used within a ChatProvider");
  }
  return context;
};