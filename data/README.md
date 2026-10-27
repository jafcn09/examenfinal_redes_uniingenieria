# Datos: CWRU Bearing Data Center

Fuente original: Case Western Reserve University Bearing Data Center, https://engineering.case.edu/bearingdatacenter (uso académico).

Se usan 44 archivos `.mat`: `Normal/` (4 cargas) y `12k_Drive_End_Bearing_Fault_Data/` (B, IR, OR@6 x 0.007/0.014/0.021 in x 4 cargas), canal Drive-End a 12 kHz.
Los datos no se suben al repositorio. Descarga automática (≈190 MB, sin credenciales) desde un espejo público de GitHub con los `.mat` originales:

```bash
python -m cwru_dl data        # clona https://github.com/s-whynot/CWRU-dataset con sparse-checkout en data/CWRU
```

Alternativa manual: descargar desde la página oficial y colocar los archivos respetando la estructura `data/CWRU/Normal/*.mat` y `data/CWRU/12k_Drive_End_Bearing_Fault_Data/<B|IR|OR>/<007|014|021>/...`.
Alternativa Kaggle: `kagglehub.dataset_download("brjapon/cwru-bearing-datasets")` y apuntar `--data` a la carpeta descargada (misma estructura).
