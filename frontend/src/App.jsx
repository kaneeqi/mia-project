// frontend/src/App.jsx
import { Loader } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { Leva } from "leva";
import { Experience } from "./components/Experience";
import { UI } from "./components/UI";
import { ChatProvider } from "./hooks/useChat";  // ← IMPORTANTE: Importar

function App() {
  return (
    <ChatProvider>  {/* ← IMPORTANTE: Envolver todo */}
      <Loader />
      <Leva hidden />
      <UI />
      <Canvas shadows camera={{ position: [0, 0, 1], fov: 30 }}>
        <Experience />
      </Canvas>
    </ChatProvider>  
  );
}

export default App;