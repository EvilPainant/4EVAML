import os

with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write("fastapi\nuvicorn\npydantic\nnumpy\nai-edge-litert\npytest\nhttpx\n")

os.makedirs(".github/workflows", exist_ok=True)
ci_yaml = """name: CI - Validación Modelo en Producción

on:
  push:
    branches: [ main, master ]
  pull_request:

jobs:
  pruebas-calidad-y-rendimiento:
    runs-on: ubuntu-latest
    steps:
      - name: Descargar código del repositorio
        uses: actions/checkout@v4

      - name: Configurar Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Instalar dependencias de inferencia y pruebas
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Ejecutar suite de pruebas automáticas (pytest)
        run: pytest -v
"""
with open(".github/workflows/ci.yml", "w", encoding="utf-8") as f:
    f.write(ci_yaml)

print("[OK] requirements.txt y .github/workflows/ci.yml creados.")