import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("comentarios_LoquillosChile.csv")
mapping = {'bueno': 0, 'malo': 1, 'neutro': 2}
df['label'] = df['sentimiento'].map(mapping)

_, X_test_texto, _, y_test = train_test_split(
    df['comentario'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
)

df_test = pd.DataFrame({'texto': X_test_texto, 'sentimiento': df.loc[X_test_texto.index, 'sentimiento']})
muestra_fija = df_test.groupby('sentimiento', group_keys=False).sample(n=10, random_state=42)
casos_lista = list(zip(muestra_fija['texto'], muestra_fija['sentimiento']))

codigo_test = f"""import os
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

# Subconjunto fijo de 30 comentarios reales de X_test (10 por clase, semilla 42)
CASOS_X_TEST = {repr(casos_lista)}

def pedir(texto: str):
    return cliente.post('/clasificar', json={{'texto': texto}})

def test_endpoint_salud():
    r = cliente.get('/salud')
    assert r.status_code == 200
    assert r.json()['estado'] == 'ok'

def test_formato_respuesta():
    r = pedir('excelente producto, llegó al toque')
    assert r.status_code == 200
    data = r.json()
    assert data['clase'] in ('bueno', 'malo', 'neutro')
    assert set(data['probabilidades'].keys()) == {{'bueno', 'malo', 'neutro'}}
    assert abs(sum(data['probabilidades'].values()) - 1.0) < 1e-2

def test_entradas_invalidas_rechazadas():
    assert cliente.post('/clasificar', json={{'texto': ''}}).status_code == 422
    assert cliente.post('/clasificar', json={{'texto': '    '}}).status_code == 422

def test_accuracy_minima_x_test():
    aciertos = [pedir(texto).json()['clase'] == esperado for texto, esperado in CASOS_X_TEST]
    acc = float(np.mean(aciertos))
    assert acc >= ACC_MIN, f"Accuracy {{acc:.2%}} cayó bajo el umbral {{ACC_MIN:.2%}}"

def test_latencia_p95():
    for _ in range(15):  # calentamiento
        pedir('cumple su función, producto estándar')
    tiempos = []
    for _ in range(100):
        t0 = time.perf_counter()
        pedir('cumple su función, producto estándar')
        tiempos.append((time.perf_counter() - t0) * 1000.0)
    p95 = float(np.percentile(tiempos, 95))
    assert p95 <= LAT_P95_MAX_MS, f"Latencia p95 {{p95:.2f}} ms excede {{LAT_P95_MAX_MS}} ms"

def test_tamano_artefacto():
    tam_kb = os.path.getsize('modelo_dr.tflite') / 1024.0
    assert tam_kb <= TAM_MAX_KB, f"Tamaño {{tam_kb:.2f}} KB excede {{TAM_MAX_KB}} KB"
"""

with open("test_servicio.py", "w", encoding="utf-8") as f:
    f.write(codigo_test)

print("[OK] test_servicio.py generado con 30 casos fijos de X_test.")