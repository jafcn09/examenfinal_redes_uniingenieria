# Artículo (formato IEEE, IEEEtran)

`main.tex` + `refs.bib`. Las tablas y figuras **no se escriben a mano**: se generan desde `results/`.

1. Correr el experimento completo (GPU T4): `python -m cwru_dl --config configs/full.yaml data|diffusion|train|report`.
2. Generar tablas y copiar figuras al artículo (desde la raíz del repositorio):
   ```bash
   python paper/make_tables.py            # escribe paper/generated/*.tex y paper/figs/*.png
   ```
3. Compilar: `cd paper && latexmk -pdf main.tex` (o subir la carpeta `paper/` a Overleaf).
4. Completar los textos marcados en **rojo** (`\pend{...}`) con la lectura de las tablas y figuras. Antes de entregar, `grep -n pend main.tex` no debe devolver nada fuera de la definición del comando.

Máximo 12 páginas (enunciado). Secciones obligatorias incluidas: Resumen, I–VI, Contribuciones, Declaración de uso de IA y Referencias (23 referencias, estilo IEEE).
