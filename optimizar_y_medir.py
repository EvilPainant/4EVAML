import os
import re
import json
import gzip
import time
import platform
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import tf2onnx
import onnxruntime as ort

# ==========================================
# 0. INFORMACIÓN DEL EQUIPO (PARA EL INFORME)
# ==========================================
def obtener_info_equipo():
    so = f"{platform.system()} {platform.release()}"
    cpu = platform.processor()
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output("wmic cpu get name", shell=True).decode()
            lines = [l.strip() for l in out.splitlines() if l.strip() and "Name" not in l]
            if lines:
                cpu = lines[0]
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu = line.split(":")[1].strip()
                        break
    except Exception:
        pass

    ram_gb = "No detectada"
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
            ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            ram_gb = f"{ram_bytes / (1024**3):.1f} GB"
        elif platform.system() == "Windows":
            out = subprocess.check_output("wmic ComputerSystem get TotalPhysicalMemory", shell=True).decode()
            lines = [l.strip() for l in out.splitlines() if l.strip().isdigit()]
            if lines:
                ram_gb = f"{int(lines[0]) / (1024**3):.1f} GB"
    except Exception:
        pass

    print("=== EQUIPO DE MEDICIÓN ===")
    print(f"Sistema Operativo : {so}")
    print(f"Procesador (CPU)  : {cpu}")
    print(f"Memoria RAM       : {ram_gb}\n")

obtener_info_equipo()

# ==========================================
# 1. RECONSTRUCCIÓN EXACTA DE DATOS Y VOCABULARIO (EVA 3)
# ==========================================
tf.keras.utils.set_random_seed(42)
df = pd.read_csv("comentarios_LoquillosChile.csv")
mapping = {'bueno': 0, 'malo': 1, 'neutro': 2}
df['label'] = df['sentimiento'].map(mapping)

X_train_texto, X_test_texto, y_train, y_test = train_test_split(
    df['comentario'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
)

LARGO = 20
vectorizador = layers.TextVectorization(max_tokens=2000, output_mode='int', output_sequence_length=LARGO)
vectorizador.adapt(X_train_texto)

X_train = vectorizador(X_train_texto).numpy().astype(np.int32)
X_test = vectorizador(X_test_texto).numpy().astype(np.int32)
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

# Guardar vocabulario.json para el servicio sin TensorFlow
vocab = vectorizador.get_vocabulary()
with open("vocabulario.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

# Verificación de equivalencia del vectorizado puro Python vs TextVectorization
indice = {palabra: i for i, palabra in enumerate(vocab)}
PUNT = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]")

def vectorizar_python(texto: str) -> np.ndarray:
    palabras = PUNT.sub('', texto.lower()).split()
    ids = [indice.get(p, 1) for p in palabras][:LARGO]
    return np.array([ids + [0] * (LARGO - len(ids))], dtype=np.int32)

# Comprobar en todo X_test y en frases con tildes y signos
vecs_py_test = np.vstack([vectorizar_python(t) for t in X_test_texto])
assert np.array_equal(vecs_py_test, X_test), "Diferencia detectada en X_test!"

casos_especiales = [
    "¡Apañó súper bien! ¿Lo recomiendan?",
    "Pésimo... ¡un cacho total! 100% malo.",
    "Ni fu ni fa, reguleque; funciona."
]
vecs_keras_esp = vectorizador(casos_especiales).numpy().astype(np.int32)
vecs_py_esp = np.vstack([vectorizar_python(t) for t in casos_especiales])
assert np.array_equal(vecs_py_esp, vecs_keras_esp), "Diferencia detectada en tildes/signos!"
print("[OK] vocabulario.json exportado y vectorizado Python 100% equivalente a TextVectorization.\n")

# ==========================================
# FUNCIÓN AUXILIAR DE LATENCIA (1 HILO CPU)
# ==========================================
def medir_latencia(fn, x_muestra, n=250, warmup=30):
    for _ in range(warmup):
        fn(x_muestra)
    tiempos = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(x_muestra)
        tiempos.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(tiempos)), float(np.percentile(tiempos, 95))

x_uno = X_test[0:1]  # forma (1, 20), int32
resultados = []

def registrar(nombre, acc, acc_base, path_archivo, lat_media, lat_p95, ref_kb=None):
    tam_kb = os.path.getsize(path_archivo) / 1024.0
    with open(path_archivo, "rb") as f:
        tam_gzip_kb = len(gzip.compress(f.read())) / 1024.0
    diff_acc = (acc - acc_base) * 100.0
    cumple_calidad = diff_acc >= -2.0
    cumple_vel = lat_p95 <= 20.0
    cumple_rec = (tam_kb <= ref_kb * 0.5) if ref_kb else False
    resultados.append({
        "Variante": nombre,
        "Accuracy": f"{acc*100:.2f}%",
        "Diff_Base": f"{diff_acc:+.2f} pp",
        "Tam_KB": round(tam_kb, 2),
        "Gzip_KB": round(tam_gzip_kb, 2),
        "Lat_Media_ms": round(lat_media, 3),
        "Lat_p95_ms": round(lat_p95, 3),
        "Cumple_3": "SÍ" if (cumple_calidad and cumple_vel and cumple_rec) else "NO"
    })

# ==========================================
# 2. LÍNEA BASE (.keras + TFLite float32 de referencia)
# ==========================================
modelo = tf.keras.models.load_model("modelo_resenas_unidad3.keras")
acc_base = float(modelo.evaluate(X_test, y_test_np, verbose=0)[1])
print(f"Parámetros totales modelo base: {modelo.count_params()}")
print(f"Accuracy modelo base en X_test: {acc_base*100:.2f}%")

# Medición de latencia en Keras (usando __call__ directo, más rápido que .predict para batch=1)
x_uno_tf = tf.constant(x_uno)
lat_m_keras, lat_p95_keras = medir_latencia(lambda x: modelo(x, training=False), x_uno_tf)
registrar("Base (.keras)", acc_base, acc_base, "modelo_resenas_unidad3.keras", lat_m_keras, lat_p95_keras)

# Clonar con batch_size=1 para conversión a TFLite (evita error TensorListReserve)
def clonar_batch_fijo(m):
    fijo = models.Sequential([
        layers.Input(shape=(LARGO,), batch_size=1, dtype='int32'),
        *[type(l).from_config(l.get_config()) for l in m.layers]
    ])
    fijo.set_weights(m.get_weights())
    return fijo

def evaluar_tflite(ruta_tflite, X_data, y_data):
    interp = tf.lite.Interpreter(model_path=ruta_tflite, num_threads=1)
    interp.allocate_tensors()
    i_in = interp.get_input_details()[0]['index']
    i_out = interp.get_output_details()[0]['index']

    def predecir_uno(x_row):
        interp.set_tensor(i_in, x_row)
        interp.invoke()
        return interp.get_tensor(i_out)

    preds = []
    for i in range(len(X_data)):
        out = predecir_uno(X_data[i:i+1])
        preds.append(int(np.argmax(out)))
    acc = float(np.mean(np.array(preds) == y_data))
    lat_m, lat_p95 = medir_latencia(predecir_uno, x_uno)
    return acc, lat_m, lat_p95

# Referencia justa de tamaño: TFLite float32 sin optimizar
fijo_base = clonar_batch_fijo(modelo)
conv_f32 = tf.lite.TFLiteConverter.from_keras_model(fijo_base)
with open("modelo_base_f32.tflite", "wb") as f:
    f.write(conv_f32.convert())

ref_tam_kb = os.path.getsize("modelo_base_f32.tflite") / 1024.0
acc_f32, lat_m_f32, lat_p95_f32 = evaluar_tflite("modelo_base_f32.tflite", X_test, y_test_np)
registrar("TFLite float32 (ref)", acc_f32, acc_base, "modelo_base_f32.tflite", lat_m_f32, lat_p95_f32, ref_tam_kb)

# ==========================================
# 3A. CUANTIFICACIÓN POST-ENTRENAMIENTO (TFLite)
# ==========================================
# Dynamic Range
conv_dr = tf.lite.TFLiteConverter.from_keras_model(fijo_base)
conv_dr.optimizations = [tf.lite.Optimize.DEFAULT]
with open("modelo_dr.tflite", "wb") as f:
    f.write(conv_dr.convert())

acc_dr, lat_m_dr, lat_p95_dr = evaluar_tflite("modelo_dr.tflite", X_test, y_test_np)
registrar("TFLite Dynamic Range", acc_dr, acc_base, "modelo_dr.tflite", lat_m_dr, lat_p95_dr, ref_tam_kb)

# Float16 (opcional para enriquecer la tabla)
conv_f16 = tf.lite.TFLiteConverter.from_keras_model(fijo_base)
conv_f16.optimizations = [tf.lite.Optimize.DEFAULT]
conv_f16.target_spec.supported_types = [tf.float16]
with open("modelo_f16.tflite", "wb") as f:
    f.write(conv_f16.convert())

acc_f16, lat_m_f16, lat_p95_f16 = evaluar_tflite("modelo_f16.tflite", X_test, y_test_np)
registrar("TFLite Float16", acc_f16, acc_base, "modelo_f16.tflite", lat_m_f16, lat_p95_f16, ref_tam_kb)

# ==========================================
# 3B. PODA POR MAGNITUD CON FINE-TUNING (50% y 80%)
# ==========================================
def aplicar_poda(m, sparsity):
    nuevos, masks = [], []
    ceros, total_matriz = 0, 0
    for w in m.get_weights():
        if w.ndim >= 2:
            umbral = np.quantile(np.abs(w), sparsity)
            mask = (np.abs(w) > umbral).astype('float32')
            ceros += int(np.sum(mask == 0))
            total_matriz += mask.size
        else:
            mask = np.ones_like(w)
        nuevos.append(w * mask)
        masks.append(mask)
    m.set_weights(nuevos)
    sparsity_real = (ceros / total_matriz) * 100.0
    return masks, sparsity_real

for sp in [0.50, 0.80]:
    pct = int(sp * 100)
    podado = tf.keras.models.clone_model(modelo)
    podado.set_weights(modelo.get_weights())
    podado.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    masks, sp_real = aplicar_poda(podado, sp)
    acc_sin_ft = float(podado.evaluate(X_test, y_test_np, verbose=0)[1])

    # Fine-tuning de 3 épocas manteniendo ceros
    for _ in range(3):
        podado.fit(X_train, y_train_np, epochs=1, batch_size=32, verbose=0)
        podado.set_weights([w * m for w, m in zip(podado.get_weights(), masks)])

    ruta_podado = f"podado_{pct}.keras"
    podado.save(ruta_podado)
    acc_con_ft = float(podado.evaluate(X_test, y_test_np, verbose=0)[1])
    lat_m_p, lat_p95_p = medir_latencia(lambda x: podado(x, training=False), x_uno_tf)

    print(f"Poda {pct}% -> Sparsity real matrices: {sp_real:.2f}% | Acc sin FT: {acc_sin_ft*100:.2f}% | Acc con FT: {acc_con_ft*100:.2f}%")
    registrar(f"Podado {pct}% + FT", acc_con_ft, acc_base, ruta_podado, lat_m_p, lat_p95_p, ref_tam_kb)

# ==========================================
# 3C. CONVERSIÓN A ONNX Y ONNX RUNTIME (1 HILO CPU)
# ==========================================
@tf.function(input_signature=[tf.TensorSpec((None, LARGO), tf.int32, name='entrada')])
def funcion_inferencia(x):
    return modelo(x)

tf2onnx.convert.from_function(
    funcion_inferencia,
    input_signature=funcion_inferencia.input_signature,
    opset=17,
    output_path="modelo.onnx"
)

opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
sesion = ort.InferenceSession("modelo.onnx", opts, providers=["CPUExecutionProvider"])

probs_onnx = sesion.run(None, {"entrada": X_test})[0]
probs_keras = modelo.predict(X_test, verbose=0)
coincide_100 = bool((probs_onnx.argmax(axis=1) == probs_keras.argmax(axis=1)).all())
acc_onnx = float(np.mean(probs_onnx.argmax(axis=1) == y_test_np))
print(f"\n[OK] ONNX coincide 100% en clases con Keras sobre X_test: {coincide_100}")

lat_m_onnx, lat_p95_onnx = medir_latencia(lambda x: sesion.run(None, {"entrada": x}), x_uno)
registrar("ONNX Runtime", acc_onnx, acc_base, "modelo.onnx", lat_m_onnx, lat_p95_onnx, ref_tam_kb)

# ==========================================
# IMPRIMIR TABLA FINAL PARA EL INFORME
# ==========================================
print("\n=== TABLA COMPARATIVA DE VARIANTES ===")
df_res = pd.DataFrame(resultados)
print(df_res.to_string(index=False))