import os
import time
import numpy as np
from fastapi.testclient import TestClient
from app import app

cliente = TestClient(app)

# Umbrales justificados por los criterios de aceptación:
# - Accuracy >= 98.0% (máximo 2 pp por debajo del 100.0% de la línea base)
# - Latencia p95 <= 20.0 ms (restricción de tiempo real del contexto en 1 hilo CPU)
# - Tamaño <= 81.12 KB (50% de los 162.24 KB del modelo TFLite float32 base)
ACC_MIN = 0.98
LAT_P95_MAX_MS = 20.0
TAM_MAX_KB = 81.12
#TAM_MAX_KB = 20.0

# Subconjunto fijo de 30 comentarios reales de X_test (10 por clase, semilla 42)
CASOS_X_TEST = [('La licuadora justo lo que necesitaba para la u, un 7', 'bueno'), ('El cargador buenísimo, material resistente', 'bueno'), ('El cargador buenísimo, material resistente', 'bueno'), ('La freidora de aire le tenía poca fe por el precio, pero es a todo dar', 'bueno'), ('Los audífonos pedazo de compra, la mejor inversión', 'bueno'), ('El cargador buenísimo, material resistente', 'bueno'), ('La silla gamer le tenía poca fe por el precio, pero es a todo dar', 'bueno'), ('El monitor cumple todo lo que promete, un manjarsh', 'bueno'), ('Los audífonos pedazo de compra, la mejor inversión', 'bueno'), ('El hervidor 100x100 Recomendadisimo, apañó súper bien', 'bueno'), ('La licuadora terrible malo, el soporte se hace el loco', 'malo'), ('El notebook súper decepcionado, cruje caleta al usarlo', 'malo'), ('El notebook un cacho total', 'malo'), ('El notebook lo mejor... para botar plata xd', 'malo'), ('El hervidor plata botada a la basura, no prende', 'malo'), ('El hervidor me duró menos que un candy', 'malo'), ('La licuadora una estafa, llegó una miniatura', 'malo'), ('La freidora de aire me duró menos que un candy', 'malo'), ('La freidora de aire penca la cuestión, murió a los dos días', 'malo'), ('La licuadora un cacho total', 'malo'), ('La silla gamer está bien, un producto del montón', 'neutro'), ('La licuadora llegó bien, es estándar', 'neutro'), ('La silla gamer está bien, un producto del montón', 'neutro'), ('El notebook funciona, pero el cable es súper corto', 'neutro'), ('Los audífonos llegó bien, es estándar', 'neutro'), ('El monitor llegó bien, es estándar', 'neutro'), ('La freidora de aire hace el trabajo, sin sorpresas', 'neutro'), ('La estufa está bien, un producto del montón', 'neutro'), ('La licuadora funciona, pero el cable es súper corto', 'neutro'), ('El cargador es piola, hace la pega y punto', 'neutro')]

def pedir(texto: str):
    return cliente.post('/clasificar', json={'texto': texto})

def test_endpoint_salud():
    r = cliente.get('/salud')
    assert r.status_code == 200
    assert r.json()['estado'] == 'ok'

def test_formato_respuesta():
    r = pedir('excelente producto, llegó al toque')
    assert r.status_code == 200
    data = r.json()
    assert data['clase'] in ('bueno', 'malo', 'neutro')
    assert set(data['probabilidades'].keys()) == {'bueno', 'malo', 'neutro'}
    assert abs(sum(data['probabilidades'].values()) - 1.0) < 1e-2

def test_entradas_invalidas_rechazadas():
    assert cliente.post('/clasificar', json={'texto': ''}).status_code == 422
    assert cliente.post('/clasificar', json={'texto': '    '}).status_code == 422

def test_accuracy_minima_x_test():
    aciertos = [pedir(texto).json()['clase'] == esperado for texto, esperado in CASOS_X_TEST]
    acc = float(np.mean(aciertos))
    assert acc >= ACC_MIN, f"Accuracy {acc:.2%} cayó bajo el umbral {ACC_MIN:.2%}"

def test_latencia_p95():
    for _ in range(15):  # calentamiento
        pedir('cumple su función, producto estándar')
    tiempos = []
    for _ in range(100):
        t0 = time.perf_counter()
        pedir('cumple su función, producto estándar')
        tiempos.append((time.perf_counter() - t0) * 1000.0)
    p95 = float(np.percentile(tiempos, 95))
    assert p95 <= LAT_P95_MAX_MS, f"Latencia p95 {p95:.2f} ms excede {LAT_P95_MAX_MS} ms"

def test_tamano_artefacto():
    tam_kb = os.path.getsize('modelo_dr.tflite') / 1024.0
    assert tam_kb <= TAM_MAX_KB, f"Tamaño {tam_kb:.2f} KB excede {TAM_MAX_KB} KB"
