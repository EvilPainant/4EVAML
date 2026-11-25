import json
import re
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

CLASES = ['bueno', 'malo', 'neutro']
LARGO = 20

with open('vocabulario.json', encoding='utf-8') as f:
    vocab = json.load(f)

indice = {palabra: i for i, palabra in enumerate(vocab)}
PUNT = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]")

def vectorizar(texto: str) -> np.ndarray:
    palabras = PUNT.sub('', texto.lower()).split()
    ids = [indice.get(p, 1) for p in palabras][:LARGO]
    return np.array([ids + [0] * (LARGO - len(ids))], dtype=np.int32)

# Cargar modelo TFLite cuantificado (Dynamic Range) con 1 hilo de CPU
interprete = Interpreter(model_path='modelo_dr.tflite', num_threads=1)
interprete.allocate_tensors()
i_in = interprete.get_input_details()[0]['index']
i_out = interprete.get_output_details()[0]['index']

app = FastAPI(title='Clasificador de comentarios - Tienda Chile')

class Comentario(BaseModel):
    texto: str = Field(min_length=1, max_length=1000)

    @field_validator('texto')
    @classmethod
    def no_solo_espacios(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('El comentario no puede estar vacío o contener solo espacios.')
        return v

@app.get('/salud')
def salud():
    return {
        'estado': 'ok',
        'modelo': 'modelo_dr.tflite',
        'hilos_cpu': 1
    }

@app.post('/clasificar')
def clasificar(c: Comentario):
    x = vectorizar(c.texto)
    interprete.set_tensor(i_in, x)
    interprete.invoke()
    probs = interprete.get_tensor(i_out)[0]
    idx_clase = int(np.argmax(probs))
    return {
        'clase': CLASES[idx_clase],
        'probabilidades': {k: round(float(v), 4) for k, v in zip(CLASES, probs)}
    }