import requests

URL = "http://127.0.0.1:8000"

print("1. GET /salud:")
print(requests.get(f"{URL}/salud").json())

print("\n2. Entrada inválida (texto vacío):")
r_err = requests.post(f"{URL}/clasificar", json={"texto": "   "})
print(f"Status HTTP: {r_err.status_code} -> {r_err.json()}")

print("\n3. Clasificación de los 5 comentarios de la Evaluación 3:")
comentarios_eva3 = [
    "me encantó, apañó súper bien",
    "se rompió al primer uso, penca la calidad",
    "es normal, ni bueno ni malo",
    "llegó rápido y es de buen material",
    "excelente para tirar la plata a la basura"
]

for texto in comentarios_eva3:
    resp = requests.post(f"{URL}/clasificar", json={"texto": texto}).json()
    print(f"- '{texto}'\n  -> Clase: {resp['clase']} | Probs: {resp['probabilidades']}")