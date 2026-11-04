# results/

Lo genera `python -m cwru_dl --config configs/full.yaml report` a partir de la corrida `full` del artículo (GPU T4).

| Carpeta / archivo | Contenido | ¿En el repositorio? |
|---|---|---|
| `tables/` | `distribucion_particiones`, `tabla_resumen`, `tabla_recall_por_clase`, `tabla_gradcam_frecuencias`, `tabla_ast_atencion_bandas`, `tabla_tstr`, `tabla_costo` | Sí |
| `figures/` | `fig01` a `fig11` del artículo | Sí |
| `runs/` | un JSON por corrida (modelo, aumento, semilla): métricas, historial por época, matriz de confusión | Sí (logs) |
| `manifest.json` | fecha, hardware, versiones, configuración, tamaños de partición, horas de GPU | Sí |
| `requirements-lock.txt` | versiones exactas de las dependencias de la corrida | Sí |
| `weights/` | mejor estado por (modelo, aumento) con semilla 0 y `ddpm_unet1d.pt` | No: enlace en el README principal |
| `synthetic/` | ventanas sintéticas del DDPM (`cwru_synthetic.pt`) | No: se regeneran con `diffusion` |
