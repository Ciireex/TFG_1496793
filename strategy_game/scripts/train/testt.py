import os
import sys
import time
from datetime import datetime
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] {msg}")

def test_environment():
    """Versión simplificada sin multiprocesamiento"""
    try:
        log("1. Importando módulo...")
        from gym_strategy.envs import StrategyEnv_Fase0_v3 as env_module
        log("✅ Módulo importado")
        
        log("2. Verificando clase...")
        if not hasattr(env_module, 'StrategyEnv_Fase0_v3'):
            raise AttributeError("Clase StrategyEnv_Fase0_v3 no encontrada")
        log(f"🔍 Clases disponibles: {[x for x in dir(env_module) if not x.startswith('__')]}")
        
        log("3. Instanciando entorno (timeout de 30 segundos)...")
        start_time = time.time()
        env = None
        
        try:
            # Intento directo con timeout manual
            env = env_module.StrategyEnv_Fase0_v3()
            elapsed = time.time() - start_time
            log(f"🎉 Entorno creado en {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            log(f"❌ Fallo después de {elapsed:.2f}s: {str(e)}")
            traceback.print_exc()
            return False
        
        log("4. Probando reset()...")
        try:
            obs = env.reset()
            log(f"✅ Reset completado. Observación: {type(obs)}")
            return True
        except Exception as e:
            log(f"❌ Error en reset(): {str(e)}")
            traceback.print_exc()
            return False
            
    except Exception as e:
        log(f"🔥 Error crítico: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == '__main__':
    log("🚀 Iniciando test simplificado")
    if test_environment():
        log("✔️ El entorno funciona correctamente")
        sys.exit(0)
    else:
        log("✖️ Se encontraron problemas")
        sys.exit(1)