# Datos: CWRU Bearing Data Center

Fuente original: Case Western Reserve University Bearing Data Center, https://engineering.case.edu/bearingdatacenter (uso académico).

Se usan 40 archivos `.mat` (10 clases x 4 cargas): `Normal/` (4 cargas, registrada a 48 kHz y remuestreada a 12 kHz en `load_signal`) y `12k_Drive_End_Bearing_Fault_Data/` (B, IR y OR en la posición @6, con daños de 0.007/0.014/0.021 in x 4 cargas), canal Drive-End a 12 kHz.

Se excluyen: el daño de 0.028 in (no existe para OR), las posiciones OR @3 y @12 (una sola clase por severidad), los datos del lado del ventilador (Fan-End) y los de 48 kHz (otra configuración de sensor y muestreo).
El espejo trae 161 archivos (~690 MB); la descarga automática solo baja `Normal/` y `12k_Drive_End_Bearing_Fault_Data/`.

| Clase | Carpeta | Archivos (carga 0-3 HP) |
|---|---|---|
| Normal | `Normal/` | 97, 98, 99, 100 |
| B007 / B014 / B021 | `12k_Drive_End_Bearing_Fault_Data/B/007`, `014`, `021` | 118-121, 185-188, 222-225 |
| IR007 / IR014 / IR021 | `12k_Drive_End_Bearing_Fault_Data/IR/007`, `014`, `021` | 105-108, 169-172, 209-212 |
| OR007 / OR014 / OR021 | `12k_Drive_End_Bearing_Fault_Data/OR/007/@6`, `014`, `021/@6` | 130-133, 197-200, 234-237 |
Los datos no se suben al repositorio. Descarga automática (≈190 MB, sin credenciales) desde un espejo público de GitHub con los `.mat` originales:

```bash
python -m cwru_dl data        # clona https://github.com/s-whynot/CWRU-dataset con sparse-checkout en data/CWRU
```

Alternativa manual: descargar desde la página oficial y colocar los archivos respetando la estructura `data/CWRU/Normal/*.mat` y `data/CWRU/12k_Drive_End_Bearing_Fault_Data/<B|IR|OR>/<007|014|021>/...`.
Alternativa Kaggle: `kagglehub.dataset_download("brjapon/cwru-bearing-datasets")` y apuntar `--data` a la carpeta descargada (misma estructura).
