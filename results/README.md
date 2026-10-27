# results/

Generado por `python -m cwru_dl report` (vacío en el repositorio; el zip entregado contiene la corrida del artículo).

```
runs/        un JSON por corrida (modelo, aumento, semilla): métricas, historial, matriz de confusión
tables/      distribucion_particiones, tabla_resumen, tabla_recall_por_clase, tabla_gradcam_frecuencias, tabla_ast_atencion_bandas, tabla_tstr, tabla_costo
figures/     fig01 ... fig11
weights/     mejor estado por (modelo, aumento) semilla 0, y ddpm_unet1d.pt
synthetic/   ventanas sintéticas (cwru_synthetic.pt) e info del DDPM
manifest.json
```
