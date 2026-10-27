# Diagnóstico de fallas en rodamientos (CWRU) con aumento por difusión condicional

[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A5%202.2-ee4c2c)](https://pytorch.org/)
[![Licencia: MIT](https://img.shields.io/badge/licencia-MIT-green)](LICENSE)

Clasificación de 10 condiciones de rodamiento a partir de señales de vibración, comparando **tres arquitecturas** (CNN 1D, CNN 2D, Audio Spectrogram Transformer) bajo **tres estrategias de aumento de datos** (ninguno, clásico, difusión condicional), en un escenario con **cambio de dominio por carga**, **desbalance de cola larga** y **ruido aditivo**.

## Integrantes

Curso **Redes Neuronales (MIA-201)** · Maestría en Inteligencia Artificial · Universidad Nacional de Ingeniería (UNI) · **Grupo 6**

| # | Integrante |
|---|---|
| 1 | Hipólito Coaguila Berrios |
| 2 | Lenin Herrera Padilla |
| 3 | Jhafet Cánepa Maceda |

## Resumen

Se clasifican 10 condiciones de rodamiento (Normal; Bola, Pista interna y Pista externa a 0.007, 0.014 y 0.021 in) a partir de vibración CWRU a 12 kHz, bajo dos dificultades reales de planta: **cambio de dominio por carga** (se entrena con 0-2 HP y se prueba con 3 HP) y **desbalance de cola larga** en las clases de falla (30 %, 10 %, 3 % por severidad).

Se comparan tres arquitecturas (CNN 1D tipo WDCNN como línea base, CNN 2D sobre espectrograma log-STFT y Audio Spectrogram Transformer preentrenado en AudioSet con fine-tuning) bajo tres condiciones de aumento (sin aumento, clásico, difusión condicional 1D con un único DDPM + classifier-free guidance), con 3 semillas, F1-macro limpio y con ruido (10 dB, 0 dB), Grad-CAM contrastado con las frecuencias teóricas de falla (BPFO/BPFI), atención del AST, t-SNE y TSTR, y costo computacional en horas de GPU.

## Cómo funciona: el flujo de extremo a extremo

```
                 data/CWRU/*.mat  (44 archivos, 12 kHz, canal Drive-End)
                         |
   [1] data   ventanas de 2048 muestras, z-score por ventana
             train: cargas 0,1,2 (80 % temporal de cada señal) + desbalance inducido
             val:   cargas 0,1,2 (20 % temporal restante)
             test:  carga 3 (nunca vista, balanceada)
                         |
   [2] diffusion   DDPM condicional 1D (U-Net + FiLM, CFG w=2.0, muestreo DDIM)
                   entrenado SOLO con el train desbalanceado
                   -> genera ventanas sintéticas hasta igualar la clase mayoritaria
                         |
   [3] train    rejilla 3 modelos x 3 aumentos x 3 semillas = 27 corridas
                none      -> train tal cual
                classic   -> ruido (10-30 dB) + desplazamiento circular + escala + SpecAugment
                diffusion -> train + ventanas sintéticas del DDPM
                   -> un JSON por corrida en results/runs/
                         |
   [4] report   tablas (media ± std), matrices de confusión, Grad-CAM 1D/2D vs BPFO/BPFI,
                atención del AST por banda mel, t-SNE real vs sintético, TSTR, costo en horas GPU
                   -> results/tables/, results/figures/, manifest.json, results_cwru.zip
                         |
   [5] demo     inferencia por ventana con los pesos guardados
```

Puntos clave del diseño, verificados en el código:

- **Sin fuga de datos:** la partición train/val es *temporal dentro de cada señal* (ninguna ventana se comparte) y el test es una carga nunca vista (`data.build_split`).
- **El desbalance se induce solo en train** (`imbalance: {007: 0.3, 014: 0.1, 021: 0.03}`); val y test quedan completos, así el F1-macro mide de verdad el efecto del aumento.
- **Un solo generador para las 10 clases:** el DDPM es condicional por clase vía embedding + FiLM, con *classifier-free guidance* (10 % de etiquetas nulas en entrenamiento).
- **Idempotencia:** cada etapa cachea su salida (`results/runs/*.json`, `results/synthetic/cwru_synthetic.pt`), así que volver a lanzar un comando no repite el trabajo ya hecho.

## Instalación

```bash
git clone  https://github.com/jafcn09/examenfinal_redes_uniingenieria
cd cwru-bearing-diffusion
python -m venv .venv && source .venv/bin/activate     # opcional
pip install -r requirements.txt
pip install -e .            # o: export PYTHONPATH=src
```

Python ≥ 3.10, PyTorch ≥ 2.2. GPU recomendada (una T4 basta); sin GPU el paquete entra automáticamente en perfil `smoke` (valida el flujo en minutos, sin valor científico).

| Perfil | Semillas | Épocas CNN / AST / DDPM | Pasos DDIM | Uso |
|---|---|---|---|---|
| `smoke` | 1 | 2 / 1 / 2 | 5 | por defecto sin GPU: comprobar que todo corre |
| `fast` | 1 | 10 / 2 / 8 | 20 | vista previa, ~20 min en T4 (`syn_cap=300`) |
| `full` | 3 | 30 / 8 / 60 | 50 | por defecto con GPU: resultados del artículo, 2-3 h en T4 |

## Ejemplo de uso

> Los bloques de salida de esta sección ilustran el **formato** que imprime cada comando; los valores concretos (métricas, tiempos, hardware) dependen de la corrida y de la máquina.

### A. Recorrido mínimo en 4 comandos (CPU, perfil `smoke`)

Sirve para ver el proyecto funcionando de principio a fin sin GPU y en minutos. La descarga de datos (≈190 MB) ocurre sola en el primer comando.

```bash
python -m cwru_dl --profile smoke data                                  # descarga + particiones + fig01-02
python -m cwru_dl --profile smoke diffusion                             # DDPM + ventanas sintéticas + fig03-04
python -m cwru_dl --profile smoke train --models CNN1D --augs none,diffusion
python -m cwru_dl --profile smoke report                                # tablas, figuras y manifest.json
```

Todos los comandos empiezan imprimiendo el hardware detectado:

```
device cpu | hardware CPU x8
```

`data` imprime dos tablas: el inventario de los 44 archivos (muestras y rpm por clase y carga) y el reparto de ventanas por partición, con esta forma:

```
carga_HP      0      1      2      3        <- muestras por archivo .mat
clase
Normal      ...    ...    ...    ...
B007        ...    ...    ...    ...
...

         train (0-2 HP, desbalanceado)  val (0-2 HP)  test (3 HP)
Normal                             N_tr          N_va         N_te
B007                      0.30 x N_tr            N_va         N_te
B014                      0.10 x N_tr            N_va         N_te
B021                      0.03 x N_tr            N_va         N_te
...
```

En esa segunda tabla se lee de un golpe el escenario del trabajo: el submuestreo por severidad (30 / 10 / 3 %) afecta **solo** a la columna de train, mientras val y test quedan íntegros. Los conteos exactos salen del largo de cada `.mat` y quedan guardados en `results/tables/distribucion_particiones.csv`.

`train` imprime el progreso por época y una línea de cierre por corrida:

```
    [CNN1D/diffusion/s0] época 10/10 | loss 0.2143 | val F1 0.961 | 48 s
CNN1D  diffusion s0 | test F1 0.887 (10 dB 0.731, 0 dB 0.402) | val F1 0.961 | 48 s
```

Al terminar, `report` deja todo listo para el artículo:

```
results/
├── runs/CNN1D_diffusion_s0.json       # métricas, historial por época, matriz de confusión
├── tables/tabla_resumen.csv           # F1-macro media ± std por (modelo, aumento)
├── figures/fig05_f1_por_modelo_aumento.png
├── weights/CNN1D_diffusion_s0.pt, ddpm_unet1d.pt
├── synthetic/cwru_synthetic.pt
└── manifest.json                      # hardware, config, versiones, horas de GPU
```

### B. Inferencia sobre una señal real (comando `demo`)

Con los pesos ya entrenados, clasifica 8 ventanas consecutivas de un archivo de prueba:

```bash
python -m cwru_dl demo --cls OR014 --load 3
```

```
device cpu | hardware CPU x8
Archivo OR014@6_3.mat (1730 rpm) | predicción por ventana: ['OR014', 'OR014', 'OR014', 'OR014', 'OR014', 'OR014', 'OR021', 'OR014']
Confianza media: 0.912
```

Opciones: `--cls` acepta cualquiera de las 10 clases (`Normal`, `B007`, `B014`, `B021`, `IR007`, `IR014`, `IR021`, `OR007`, `OR014`, `OR021`), `--load` la carga en HP (0-3) y `--aug` el juego de pesos a usar (`none`, `classic`, `diffusion`).

### C. Uso del paquete desde Python

Entrenar un modelo y evaluarlo en la carga no vista:

```python
from cwru_dl import load_config
from cwru_dl.data import build_split, download_cwru
from cwru_dl.models import build_model
from cwru_dl.train import metrics, predict, train_classifier

cfg = load_config("configs/cnn1d.yaml", profile="fast")
download_cwru(cfg.root)
splits = build_split(cfg)
Xtr, ytr = splits["train"]; Xva, yva = splits["val"]; Xte, yte = splits["test"]
print(Xtr.shape, Xte.shape)          # (N_train, 2048) y (N_test, 2048), ventanas ya normalizadas

model, info = train_classifier(cfg, build_model("CNN1D", cfg), Xtr, ytr, Xva, yva,
                               epochs=cfg.epochs_cnn, lr=cfg.lr_cnn, aug="classic",
                               seed=0, tag="ejemplo")
m = metrics(yte.numpy(), predict(model, Xte))
print(f"F1-macro en 3 HP: {m['f1_macro']:.3f} | accuracy: {m['acc']:.3f}")
```

Clasificar una señal propia con pesos ya guardados:

```python
import torch
from cwru_dl import CLASSES, DEVICE, load_config
from cwru_dl.data import load_signal, make_windows, zscore
from cwru_dl.train import load_trained

cfg = load_config()
model = load_trained(cfg, "CNN1D", "diffusion")        # results/weights/CNN1D_diffusion_s0.pt
x, rpm, name = load_signal(cfg, "IR021", 3)            # o su propio array de 12 kHz

w = torch.from_numpy(zscore(make_windows(x, cfg.win, cfg.win))).to(DEVICE)
with torch.no_grad():
    probs = model(w).softmax(1)
pred = probs.mean(0).argmax().item()
print(f"{name} ({rpm:.0f} rpm) -> {CLASSES[pred]} (p = {probs.mean(0)[pred]:.3f})")
```

Generar ventanas sintéticas nuevas con el DDPM entrenado:

```python
import torch
from cwru_dl import CLASSES, DEVICE, load_config
from cwru_dl.diffusion import Diffusion, UNet1D

cfg = load_config()
net = UNet1D(len(CLASSES)).to(DEVICE)
net.load_state_dict(torch.load(cfg.out / "weights" / "ddpm_unet1d.pt", map_location=DEVICE))
diff = Diffusion(cfg.diff_T, DEVICE)

y = torch.full((16,), CLASSES.index("OR021"), device=DEVICE)   # 16 ventanas de la clase más escasa
Xsyn = diff.ddim_sample(net, y, steps=cfg.ddim_steps, w=cfg.cfg_w, null_id=len(CLASSES), L=cfg.win)
print(Xsyn.shape)        # torch.Size([16, 2048])
```

## Reproducir el artículo paso a paso

Cada comando es idempotente: las corridas ya hechas se leen de `results/runs/` y no se repiten.

| Paso | Comando | Genera |
|---|---|---|
| 1. Datos y particiones | `python -m cwru_dl --config configs/full.yaml data` | descarga `data/CWRU`, `tables/distribucion_particiones.csv`, `fig01`, `fig02` |
| 2. Generador | `python -m cwru_dl --config configs/full.yaml diffusion` | `weights/ddpm_unet1d.pt`, `synthetic/cwru_synthetic.pt`, `fig03`, `fig04` |
| 3. Rejilla 3×3×3 | `python -m cwru_dl --config configs/full.yaml train` | `runs/*.json`, `weights/<modelo>_<aumento>_s0.pt` |
| 4. Tablas, figuras, interpretabilidad, costo | `python -m cwru_dl --config configs/full.yaml report` | `tabla_resumen`, `tabla_recall_por_clase`, `tabla_gradcam_frecuencias`, `tabla_ast_atencion_bandas`, `tabla_tstr`, `tabla_costo`, `fig05` a `fig11`, `manifest.json`, `results_cwru.zip` |
| 5. Demo | `python -m cwru_dl demo --cls OR014 --load 3` | predicción por ventana con los pesos guardados |

Un solo modelo o una sola condición: `python -m cwru_dl --config configs/full.yaml train --models CNN1D --augs none,diffusion`.
Vista previa rápida (1 semilla, menos épocas, ~20 min en T4): añadir `--profile fast`.
Configuración por experimento: `configs/cnn1d.yaml`, `cnn2d.yaml`, `ast.yaml`, `diffusion.yaml`; `base.yaml` fija datos, particiones, desbalance y semillas.

En Colab: abrir `notebooks/01_pipeline_colab.ipynb`, elegir GPU T4, editar `REPO_URL` y Ejecutar todo (2 a 3 h en T4 para la configuración completa).

### Opciones de la línea de comandos

```
python -m cwru_dl [--config configs/X.yaml] [--profile smoke|fast|full] [--data DIR] [--out DIR]
                  {data,diffusion,train,report,demo} [opciones del subcomando]
```

| Variable de entorno | Efecto |
|---|---|
| `CWRU_PROFILE` | fija el perfil sin pasar `--profile` (útil en Colab) |
| `CWRU_SYN_CAP` | tope de ventanas sintéticas por clase (0 = sin tope) |
| `CWRU_ALLOW_RANDOM_AST` | `1` permite seguir con un AST reducido aleatorio si falla la descarga del preentrenado (solo para validar el flujo) |

## Resultados principales

Se completan con la corrida en Colab (T4); los archivos exactos son `results/tables/tabla_resumen.csv`, `tabla_recall_por_clase.csv`, `tabla_gradcam_frecuencias.csv`, `tabla_tstr.csv` y `tabla_costo.csv`.

| Modelo | Aumento | F1-macro 3 HP | F1 a 10 dB | F1 a 0 dB |
|---|---|---|---|---|
| CNN1D | none / classic / diffusion | _tabla_resumen.csv_ | | |
| CNN2D | none / classic / diffusion | | | |
| AST | none / classic / diffusion | | | |

Referencia obtenida en validación en CPU (perfil `fast`, 1 semilla, 10 épocas, sin AST preentrenado): CNN1D sin aumento F1 = 0.85 en 3 HP; el aumento clásico duplica la robustez a 0 dB (0.40 vs 0.20).

## Qué tabla o figura produce cada número del artículo

| Elemento del artículo | Archivo | Comando |
|---|---|---|
| Distribución de particiones (§III) | `tables/distribucion_particiones.csv`, `figures/fig01_distribucion.png` | `data` |
| Espectro de envolvente vs BPFO/BPFI (§III) | `figures/fig02_envolvente_teorica.png` | `data` |
| Curva del DDPM y real vs sintético (§IV) | `figures/fig03_difusion_loss.png`, `fig04_real_vs_sintetico.png` | `diffusion` |
| Tabla principal media ± std (§IV) | `tables/tabla_resumen.csv`, `figures/fig05_f1_por_modelo_aumento.png` | `report` |
| Recall por clase, matrices de confusión, curvas (§IV) | `tables/tabla_recall_por_clase.csv`, `fig06`, `fig07` | `report` |
| Grad-CAM 2D/1D y periodicidad vs BPFO/BPFI (§IV, R5) | `tables/tabla_gradcam_frecuencias.csv`, `fig08`, `fig09` | `report` |
| Atención del AST por banda (§IV, R5) | `tables/tabla_ast_atencion_bandas.csv`, `fig10` | `report` |
| t-SNE y TSTR (§IV, R5) | `figures/fig11_tsne.png`, `tables/tabla_tstr.csv` | `report` |
| Costo computacional (§IV, R6) | `tables/tabla_costo.csv`, `manifest.json` | `report` |

## Estructura del repositorio

```
cwru-bearing-diffusion/
├── README.md                 # este archivo
├── LICENSE                   # MIT (código); los datos conservan los términos de CWRU
├── requirements.txt           # dependencias con versiones mínimas
├── pyproject.toml             # instalación del paquete (pip install -e .)
├── data/                      # README con fuente, licencia y descarga (los .mat no se suben)
├── src/cwru_dl/
│   ├── config.py              # CFG, perfiles, clases, constantes del rodamiento SKF 6205
│   ├── data.py                # descarga, ventaneo, particiones, desbalance, envolvente de Hilbert
│   ├── augment.py             # aumento clásico (ruido/desplazamiento/escala) y SpecAugment
│   ├── models.py              # CNN1D (WDCNN), CNN2D (log-STFT), AST con pos-emb interpolados
│   ├── diffusion.py           # U-Net 1D + FiLM, DDPM con CFG, muestreo DDIM, balanceo sintético
│   ├── train.py               # bucle de entrenamiento, métricas, rejilla de experimentos, caché
│   ├── interpret.py           # Grad-CAM, periodicidad vs BPFO/BPFI, rollout de atención, t-SNE, TSTR
│   ├── report.py              # fig01-fig11, tablas, tabla de costo, manifest y zip
│   └── cli.py                 # subcomandos data | diffusion | train | report | demo
├── notebooks/                 # 01_pipeline_colab.ipynb (orquestador), 02_demo_inferencia.ipynb
├── configs/                   # base.yaml + un YAML por experimento (cnn1d, cnn2d, ast, diffusion, full)
└── results/                   # runs/, tables/, figures/, weights/, synthetic/, manifest.json (generado)
```

## Semillas y pesos

Semillas fijadas en `configs/base.yaml` (`seeds: [0, 1, 2]`); cada corrida fija NumPy, Python y PyTorch con `seed_all`. Los pesos del mejor estado por (modelo, aumento) para la semilla 0 y el DDPM se guardan en `results/weights/`.
Enlace a los pesos entrenados: _completar (Drive o Hugging Face)_.

## Decisiones de diseño verificadas

- Partición train/val temporal dentro de cada señal (sin ventanas compartidas) y test en una carga nunca vista: sin fuga de datos.
- El AST conserva sus embeddings posicionales preentrenados mediante interpolación bilineal de 12×101 a 12×9 parches en lugar de reinicializarlos.
- El DDPM usa EMA con calentamiento (`decay = min(0.999, (1+s)/(10+s))`): sin él, con pocos pasos las muestras degeneran en ruido blanco.
- La periodicidad del Grad-CAM se mide en la segunda etapa convolucional (375 Hz de muestreo temporal) para poder resolver BPFO ≈ 103 Hz y BPFI ≈ 156 Hz a 1730 rpm.

## Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `subprocess.CalledProcessError` en `data` | la descarga usa `git clone` con sparse-checkout: verifique red y que `git` esté instalado, o coloque los `.mat` a mano (ver `data/README.md`) |
| El AST falla al descargar los pesos | sin red hacia Hugging Face; para validar solo el flujo: `CWRU_ALLOW_RANDOM_AST=1` (los números dejan de ser comparables) |
| `report` se queja de pesos faltantes | las figuras de interpretabilidad requieren la etapa `train` previa; `report` omite las que no tienen pesos |
| Quiero rehacer una corrida | borre su JSON en `results/runs/` (y `results/synthetic/` para regenerar el DDPM) |
| Entrenar es muy lento | use `--profile fast`, o limite la rejilla con `--models` y `--augs` |

## Licencia

Código y documentación: **MIT** (ver [LICENSE](LICENSE)).
Los datos de vibración pertenecen al Case Western Reserve University Bearing Data Center y se rigen por sus términos de uso académico; **no se redistribuyen** en este repositorio. Los pesos del AST preentrenado conservan la licencia de sus autores.

## Referencias de datos

- Case Western Reserve University Bearing Data Center: https://engineering.case.edu/bearingdatacenter
- Espejo público de los `.mat`: https://github.com/s-whynot/CWRU-dataset


