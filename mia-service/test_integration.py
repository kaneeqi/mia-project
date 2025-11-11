"""
Script de Testing para verificar que todos los componentes de MIA están funcionando
Ejecutar: python test_integration.py
"""

import requests
import json
import sys
import time
from colorama import init, Fore, Style

init(autoreset=True)

# ==================== CONFIGURACIÓN ====================
MIA_SERVICE_URL = "http://localhost:5000"
BACKEND_URL = "http://localhost:3000"

# ==================== TESTS ====================

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_success(text):
    print(f"{Fore.GREEN}✅ {text}{Style.RESET_ALL}")

def print_error(text):
    print(f"{Fore.RED}❌ {text}{Style.RESET_ALL}")

def print_info(text):
    print(f"{Fore.CYAN}ℹ️  {text}{Style.RESET_ALL}")

def test_mia_service_health():
    """Test 1: Verificar que el servicio Python de MIA está activo"""
    print_header("TEST 1: MIA Service Health")
    
    try:
        response = requests.get(f"{MIA_SERVICE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"MIA Service está activo")
            print_info(f"   Device: {data.get('device', 'N/A')}")
            print_info(f"   Models loaded: {data.get('models_loaded', False)}")
            return True
        else:
            print_error(f"MIA Service respondió con código {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("No se pudo conectar al MIA Service")
        print_info("   ¿Está corriendo? Ejecutar: python mia_service.py")
        return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_classify_emotion():
    """Test 2: Clasificar emoción del usuario"""
    print_header("TEST 2: Clasificación de Emoción del Usuario (MiaMotion)")
    
    test_cases = [
        ("Estoy muy feliz hoy", "alegría"),
        ("Me siento triste y solo", "tristeza"),
        ("Te quiero mucho", "amor"),
        ("Estoy muy enojado", "ira"),
        ("Tengo miedo de fracasar", "miedo"),
        ("¡Wow, qué sorpresa!", "sorpresa")
    ]
    
    passed = 0
    for text, expected_emotion in test_cases:
        try:
            response = requests.post(
                f"{MIA_SERVICE_URL}/classify_emotion",
                json={"text": text},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                detected = data.get("emotion")
                confidence = data.get("confidence", 0)
                
                if detected == expected_emotion:
                    print_success(f"\"{text}\"")
                    print_info(f"   Detectado: {detected} (confianza: {confidence:.2f})")
                    passed += 1
                else:
                    print_error(f"\"{text}\"")
                    print_info(f"   Esperado: {expected_emotion}, Detectado: {detected}")
            else:
                print_error(f"Error {response.status_code} para: \"{text}\"")
                
        except Exception as e:
            print_error(f"Error procesando: \"{text}\" - {str(e)}")
    
    print_info(f"\n   Pasados: {passed}/{len(test_cases)}")
    return passed == len(test_cases)

def test_predict_agent_emotion():
    """Test 3: Predecir emoción del agente"""
    print_header("TEST 3: Predicción de Emoción del Agente (MiaPredict)")
    
    test_cases = [
        ("Estoy muy feliz", 1, ["alegría", "amor"]),  # label 1 = alegría
        ("Me siento solo", 0, ["alegría", "amor"]),   # label 0 = tristeza
        ("Te amo", 2, ["alegría", "amor"]),           # label 2 = amor
    ]
    
    passed = 0
    for text, user_label, expected_options in test_cases:
        try:
            response = requests.post(
                f"{MIA_SERVICE_URL}/predict_agent_emotion",
                json={"text": text, "user_label": user_label},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                agent_emotion = data.get("agent_emotion")
                confidence = data.get("confidence", 0)
                
                if agent_emotion in expected_options:
                    print_success(f"User: \"{text}\" (label={user_label})")
                    print_info(f"   Agente responde con: {agent_emotion} (confianza: {confidence:.2f})")
                    passed += 1
                else:
                    print_error(f"User: \"{text}\"")
                    print_info(f"   Esperado: {expected_options}, Recibido: {agent_emotion}")
            else:
                print_error(f"Error {response.status_code}")
                
        except Exception as e:
            print_error(f"Error: {str(e)}")
    
    print_info(f"\n   Pasados: {passed}/{len(test_cases)}")
    return passed >= len(test_cases) // 2  # Al menos 50%

def test_full_pipeline():
    """Test 4: Pipeline completo MiaMotion + MiaPredict"""
    print_header("TEST 4: Pipeline Completo")
    
    test_text = "Estoy muy triste y necesito hablar con alguien"
    
    try:
        print_info(f"Procesando: \"{test_text}\"")
        
        response = requests.post(
            f"{MIA_SERVICE_URL}/full_pipeline",
            json={"text": test_text},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            
            user = data.get("user", {})
            agent = data.get("agent", {})
            
            print_success("Pipeline completado exitosamente")
            print_info(f"   Usuario:")
            print_info(f"      Emoción: {user.get('emotion')}")
            print_info(f"      Confianza: {user.get('confidence', 0):.2f}")
            print_info(f"   Agente:")
            print_info(f"      Emoción: {agent.get('emotion')}")
            print_info(f"      Expresión facial: {agent.get('facial_expression')}")
            print_info(f"      Confianza: {agent.get('confidence', 0):.2f}")
            
            return True
        else:
            print_error(f"Pipeline falló con código {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_backend_health():
    """Test 5: Verificar que el backend Node.js está activo"""
    print_header("TEST 5: Backend Node.js Health")
    
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Backend está activo")
            print_info(f"   Status: {data.get('status', 'N/A')}")
            
            services = data.get('services', {})
            for service, status in services.items():
                print_info(f"   {service}: {status}")
            
            return True
        else:
            print_error(f"Backend respondió con código {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("No se pudo conectar al Backend")
        print_info("   ¿Está corriendo? Ejecutar: npm start en /backend")
        return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_full_chat_endpoint():
    """Test 6: Probar endpoint /chat completo"""
    print_header("TEST 6: Endpoint /chat (Backend + MIA + LLM)")
    
    test_message = "Hola, ¿cómo estás?"
    
    try:
        print_info(f"Enviando: \"{test_message}\"")
        print_info("Esto puede tardar 5-10 segundos...")
        
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={"message": test_message},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            messages = data.get("messages", [])
            
            if len(messages) >= 2:
                user_msg = messages[0]
                agent_msg = messages[1]
                
                print_success("Chat endpoint funcionando")
                print_info(f"   Usuario: {user_msg.get('text')}")
                print_info(f"   Agente: {agent_msg.get('text')}")
                print_info(f"   Audio generado: {'Sí' if agent_msg.get('audio') else 'No'}")
                print_info(f"   Lip-sync generado: {'Sí' if agent_msg.get('lipsync') else 'No'}")
                print_info(f"   Expresión facial: {agent_msg.get('facialExpression')}")
                print_info(f"   Animación: {agent_msg.get('animation')}")
                
                # Verificar emociones de MIA
                emotions = agent_msg.get('emotions', {})
                if emotions:
                    print_info(f"   Emociones MIA:")
                    print_info(f"      Usuario: {emotions.get('user', {}).get('emotion')}")
                    print_info(f"      Agente: {emotions.get('agent', {}).get('emotion')}")
                
                return True
            else:
                print_error("Respuesta inválida: mensajes insuficientes")
                return False
        else:
            print_error(f"Chat endpoint falló con código {response.status_code}")
            print_info(f"   Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print_error("Timeout esperando respuesta del chat")
        print_info("   Esto puede indicar problemas con:")
        print_info("   - OpenAI API (verificar API key)")
        print_info("   - ElevenLabs API (verificar API key)")
        print_info("   - Rhubarb (verificar instalación)")
        return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

# ==================== RUNNER ====================

def run_all_tests():
    """Ejecuta todos los tests"""
    print("\n")
    print(Fore.CYAN + "╔" + "="*58 + "╗")
    print(Fore.CYAN + "║" + " "*15 + "MIA INTEGRATION TESTS" + " "*22 + "║")
    print(Fore.CYAN + "╚" + "="*58 + "╝" + Style.RESET_ALL)
    
    results = []
    
    # Tests críticos (deben pasar para continuar)
    critical_tests = [
        ("MIA Service Health", test_mia_service_health),
        ("Backend Health", test_backend_health),
    ]
    
    for test_name, test_func in critical_tests:
        result = test_func()
        results.append((test_name, result))
        if not result:
            print_error(f"\n❌ Test crítico falló: {test_name}")
            print_info("   No se pueden ejecutar más tests.")
            print_info("   Por favor, corrige este error primero.\n")
            return
        time.sleep(0.5)
    
    # Tests funcionales
    functional_tests = [
        ("Classify Emotion", test_classify_emotion),
        ("Predict Agent Emotion", test_predict_agent_emotion),
        ("Full Pipeline", test_full_pipeline),
        ("Full Chat Endpoint", test_full_chat_endpoint),
    ]
    
    for test_name, test_func in functional_tests:
        result = test_func()
        results.append((test_name, result))
        time.sleep(0.5)
    
    # Resumen
    print_header("RESUMEN DE TESTS")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        color = Fore.GREEN if result else Fore.RED
        print(f"{color}{status}{Style.RESET_ALL} - {test_name}")
    
    print("\n" + "="*60)
    percentage = (passed / total) * 100
    
    if percentage == 100:
        print(f"{Fore.GREEN}🎉 ¡TODOS LOS TESTS PASARON! ({passed}/{total}){Style.RESET_ALL}")
        print(f"{Fore.GREEN}   MIA está completamente integrada y funcionando.{Style.RESET_ALL}")
    elif percentage >= 75:
        print(f"{Fore.YELLOW}⚠️  Mayoría de tests pasaron ({passed}/{total} - {percentage:.0f}%){Style.RESET_ALL}")
        print(f"{Fore.YELLOW}   Revisa los tests fallidos antes de usar en producción.{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}❌ Varios tests fallaron ({passed}/{total} - {percentage:.0f}%){Style.RESET_ALL}")
        print(f"{Fore.RED}   Se requiere debugging antes de continuar.{Style.RESET_ALL}")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠️  Tests interrumpidos por el usuario{Style.RESET_ALL}\n")
        sys.exit(0)
