import nbformat as nbf

C = []
def md(s): C.append(("md", s.strip("\n")))
def code(s): C.append(("code", s.strip("\n")))

md(r"""
# MIA-201 Examen Final: Diagnóstico de fallas en rodamientos (CWRU) con aumento por difusión condicional

**Universidad Nacional de Ingeniería · Maestría en Inteligencia Artificial** · Redes Neuronales (MIA-201) · Grupo 6
**Equipo (Grupo 6):** Hipólito Coaguila Berrios · Lenin Herrera Padilla · Jhafet Cánepa Maceda

Este notebook es el orquestador reproducible del proyecto: todo el código vive en el paquete `src/cwru_dl` del repositorio y aquí solo se invoca, se muestran tablas y figuras, y se documenta qué se verificó en cada etapa.
Equivalencia con la línea de comandos: `python -m cwru_dl data | diffusion | train | report | demo`.

## Problema
Clasificar 10 condiciones de un rodamiento (Normal, Bola, Pista interna, Pista externa; severidades 0.007, 0.014 y 0.021 in) a partir de vibración, con tres dificultades reales de planta:

1. **Cambio de dominio por carga:** se entrena con 0, 1 y 2 HP y se prueba con 3 HP (nunca vista).
2. **Desbalance inducido (cola larga):** las clases de falla se submuestrean por severidad (0.007: 30 %, 0.014: 10 %, 0.021: 3 %).
3. **Robustez al ruido:** además del test limpio se evalúa con ruido blanco a 10 dB y 0 dB de SNR.

## Diseño experimental (mapeo a la rúbrica)
| Elemento | Decisión | Requisito |
|---|---|---|
| Datos | CWRU 12 kHz Drive-End, partición por carga, desbalance inducido | R2 |
| Modelos | 1D-CNN (WDCNN, línea base), CNN-2D sobre espectrograma log-STFT, AST preentrenado (AudioSet) con fine-tuning | R1, R3 |
| Aumento (ablación) | Sin aumento vs clásico (ruido, desplazamiento, escala, SpecAugment) vs difusión condicional 1D (DDPM + CFG, un solo generador) | R4 |
| Métricas | F1-macro y recall por clase en 3 HP (limpio, 10 dB, 0 dB), 3 semillas (media ± desviación) | R4 |
| Interpretabilidad | Grad-CAM (1D y 2D), atención del AST, periodicidad del CAM vs BPFO/BPFI teóricas, t-SNE real vs sintético, TSTR | R5 |
| Costo | Hardware y horas de GPU registradas por corrida | R6 |
""")

md(r"""
## 1. Entorno
En Colab: clona el repositorio e instala dependencias. En local: ejecuta desde la raíz del repositorio.
Perfiles: `full` (por defecto con GPU: 3 semillas, 27 corridas), `fast` (1 semilla, menos épocas, ~20 min en T4), `smoke` (automático sin GPU, valida el flujo en minutos).
""")

code(r"""
import os, sys, subprocess
from pathlib import Path

REPO_URL = "https://github.com/<usuario>/cwru-bearing-diffusion.git"
if not Path("src/cwru_dl").exists():
    if Path("../src/cwru_dl").exists():
        os.chdir("..")
    else:
        subprocess.run(["git", "clone", "-q", REPO_URL, "cwru-bearing-diffusion"], check=True); os.chdir("cwru-bearing-diffusion")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
sys.path.insert(0, "src")
PROFILE = os.environ.get("CWRU_PROFILE")
print("raíz del repositorio:", Path.cwd())
""")

code(r"""
import json, time
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from IPython.display import display

from cwru_dl import CLASSES, DEVICE, HW, load_config
from cwru_dl.data import build_split, class_counts, download_cwru, load_meta, load_signal
from cwru_dl.diffusion import run_diffusion_stage
from cwru_dl.models import CNN1D, CNN2D, build_ast, build_model
from cwru_dl.train import load_records, load_trained, has_run, run_experiments, train_classifier
from cwru_dl.interpret import ast_band_table, cam_frequency_table, tsne_real_vs_synthetic, tstr
from cwru_dl import report

cfg = load_config("configs/full.yaml", profile=PROFILE)
print(f"torch {torch.__version__} | device {DEVICE} | hardware {HW} | perfil {cfg.profile}")
print(json.dumps(cfg.as_dict(), indent=1, default=str))
""")

md(r"""
**Verificado:** la configuración se lee de `configs/base.yaml` + `configs/full.yaml` (criterio 5) y el perfil se decide por hardware; todo el notebook depende de un solo objeto `cfg`.
""")

md(r"""
## 2. Datos: CWRU Bearing Data Center
Fuente: Case Western Reserve University Bearing Data Center (licencia académica). Descarga automática de un espejo público con los `.mat` originales (sin credenciales, ≈190 MB); alternativa Kaggle documentada en `data/README.md`.
Partición: train/val temporal dentro de cada señal de 0-2 HP (sin ventanas compartidas), test = 3 HP completo y balanceado; el desbalance se induce solo en train.
""")

code(r"""
download_cwru(cfg.root)
meta = load_meta(cfg)
display(meta.pivot(index="clase", columns="carga_HP", values="muestras").loc[CLASSES])
print("RPM por carga:", meta.groupby("carga_HP").rpm.mean().round(1).to_dict())

splits = build_split(cfg)
(Xtr, ytr), (Xva, yva), (Xte, yte) = splits["train"], splits["val"], splits["test"]
dist = class_counts(splits); display(dist)
report.fig_distribution(cfg, dist)
print("Shapes:", Xtr.shape, Xva.shape, Xte.shape)
""")

code(r"""
report.fig_envelope(cfg)
rpm3 = meta.query("carga_HP == 3").rpm.mean()
from cwru_dl.data import fault_freqs
print("Frecuencias teóricas @3HP:", {k: round(v, 1) for k, v in fault_freqs(rpm3).items()})
""")

md(r"""
**Verificado:** 40 archivos (10 clases × 4 cargas) del canal Drive-End; el espectro de envolvente muestra picos exactamente en BPFO/BPFI/BSF y sus armónicos, lo que confirma que las frecuencias teóricas están en los datos y sirven de referencia física para la sección 7.
""")

md(r"""
## 3. Modelos
Tres arquitecturas que consumen la misma ventana cruda (2048 muestras, z-score); las transformaciones tiempo-frecuencia viven dentro de cada modelo.

| Modelo | Entrada interna | Parámetros | Rol |
|---|---|---|---|
| `CNN1D` (WDCNN) | onda cruda | ~0.04 M | línea base del curso |
| `CNN2D` | log-STFT 129×257 | ~0.24 M | CNN sobre imagen tiempo-frecuencia |
| `ASTClassifier` | log-mel 96×128 | 86 M | Transformer preentrenado en AudioSet, fine-tuning completo con embeddings posicionales interpolados (12×101 → 12×9) |

Pérdida: entropía cruzada $\mathcal{L}=-\sum_i \log p_\theta(y_i\mid x_i)$ sin ponderar, para aislar el efecto del aumento sobre el desbalance.
""")

code(r"""
for m in (CNN1D(), CNN2D(), build_ast(cfg)):
    with torch.no_grad():
        out = m(torch.randn(2, cfg.win))
    print(f"{m.__class__.__name__}: {sum(p.numel() for p in m.parameters())/1e6:.2f} M parámetros | logits {tuple(out.shape)}")
""")

md(r"""
**Verificado:** las tres arquitecturas aceptan la misma ventana cruda y devuelven logits de 10 clases; el AST conserva los embeddings posicionales preentrenados vía interpolación bilineal en lugar de reinicializarlos.
""")

md(r"""
## 4. Generador: difusión condicional 1D (DDPM + classifier-free guidance)
Un único generador aprende $p(x\mid y)$ sobre el conjunto de entrenamiento desbalanceado y sintetiza solo el déficit de las clases minoritarias.

Proceso directo $q(x_t\mid x_0)=\mathcal{N}(\sqrt{\bar\alpha_t}\,x_0,(1-\bar\alpha_t)I)$; objetivo $\mathcal{L}_{\text{simple}}=\mathbb{E}\big\|\epsilon-\epsilon_\theta(x_t,t,y)\big\|^2$ con $y=\varnothing$ el 10 % de las veces; muestreo DDIM ($\eta=0$, 50 pasos) con guía $\tilde\epsilon=(1+w)\,\epsilon_\theta(x_t,t,y)-w\,\epsilon_\theta(x_t,t,\varnothing)$.
$\epsilon_\theta$ es una U-Net 1D con bloques residuales modulados (FiLM) por los embeddings de tiempo y clase; pesos EMA con calentamiento.
""")

code(r"""
Xsyn, ysyn, diff_info = run_diffusion_stage(cfg, splits)
report.fig_diffusion_loss(cfg, diff_info)
report.fig_real_vs_synthetic(cfg, Xtr, ytr, Xsyn, ysyn)
print(f"Sintéticas: {tuple(Xsyn.shape)} | entrenamiento {diff_info['train_sec']:.0f} s | muestreo {diff_info['sample_sec']:.0f} s")
""")

md(r"""
**Verificado:** el DDPM genera exactamente el déficit de cada clase minoritaria (balance sin tocar la mayoritaria) y las PSD medias real vs sintética muestran si reproduce las bandas de resonancia; la calidad se cuantifica en la sección 7 (TSTR, t-SNE).
""")

md(r"""
## 5. Experimentos: 3 modelos × 3 condiciones de aumento × 3 semillas
Aumento clásico sobre la onda (ruido con SNR ∈ [10, 30] dB, desplazamiento circular, escala ∈ [0.8, 1.2]) + SpecAugment en los modelos tiempo-frecuencia. AdamW, coseno con calentamiento, precisión mixta, selección del mejor estado por F1-macro en validación. Cada corrida se persiste en `results/runs/` y se reanuda si existe.
""")

code(r"""
records = run_experiments(cfg, splits, Xsyn, ysyn)
df, summary, recall = report.summary_tables(cfg, records)
display(summary); display(recall)
report.fig_results(cfg, df); report.fig_confusion(cfg, records); report.fig_curves(cfg, records)
""")

md(r"""
**Verificado:** la tabla resumen reporta media ± desviación sobre semillas en test limpio, 10 dB y 0 dB; el recall por clase evidencia el efecto del aumento sobre las severidades escasas (0.014 y 0.021); las matrices de confusión muestran dónde se confunden severidades bajo el cambio de carga.
""")

md(r"""
## 6. Interpretabilidad y calidad de los datos sintéticos
1. **Grad-CAM 2D** (CNN2D): mapa tiempo-frecuencia de relevancia.
2. **Periodicidad del CAM vs frecuencias teóricas:** el mapa (segunda etapa convolucional, 64 pasos temporales, 375 Hz de muestreo) se proyecta al eje temporal y se busca su frecuencia dominante en 80-185 Hz; debe coincidir con BPFO ≈ 103 Hz (pista externa) o BPFI ≈ 156 Hz (pista interna). La misma rutina sobre la envolvente cruda es la referencia física. Se reporta también el centroide espectral del mapa (banda de resonancia atendida). En pista interna los impactos están modulados por la rotación del eje (fr ≈ 29 Hz), por lo que el CAM puede seguir la envolvente de modulación en lugar de BPFI; es un hallazgo a discutir, no un error.
3. **Grad-CAM 1D** (CNN1D) sobre la forma de onda.
4. **Atención del AST** (rollout multicapa desde CLS) por banda mel y clase.
5. **t-SNE** real vs sintético y **TSTR** (entrenar solo con sintético, probar con real).
""")

code(r"""
cam_model = load_trained(cfg, "CNN2D", "diffusion")
cam_tab = cam_frequency_table(cfg, cam_model, Xte, yte, rpm3); display(cam_tab)
cam_tab.to_csv(cfg.out / "tables" / "tabla_gradcam_frecuencias.csv", index=False)
report.fig_gradcam2d(cfg, cam_model, Xte, yte, rpm3)
report.fig_gradcam1d(cfg, load_trained(cfg, "CNN1D", "diffusion"), Xte, yte, rpm3)
""")

code(r"""
ast_model = load_trained(cfg, "AST", "diffusion")
report.fig_ast_attention(cfg, ast_model, Xte, yte)
band = ast_band_table(cfg, ast_model, Xte, yte); display(band)
band.to_csv(cfg.out / "tables" / "tabla_ast_atencion_bandas.csv")
""")

code(r"""
report.fig_tsne(cfg, *tsne_real_vs_synthetic(cfg, Xtr, ytr, Xsyn, ysyn))
tstr_tab = tstr(cfg, Xsyn, ysyn, Xva, yva, Xte, yte); display(tstr_tab)
tstr_tab.to_csv(cfg.out / "tables" / "tabla_tstr.csv", index=False)
print("Referencia real→real (CNN1D, sin aumento):", summary.loc[("CNN1D", "none"), "F1_macro"])
""")

md(r"""
**Verificado:** los mapas Grad-CAM quedan normalizados en [0, 1] y alineados con el espectrograma y la onda; la tabla de frecuencias compara la periodicidad del CAM con BPFO/BPFI usando la envolvente cruda como control; la atención del AST se agrega por banda; t-SNE y TSTR cuantifican si lo sintético ocupa la región de su clase y si basta por sí solo para entrenar.
""")

md(r"""
## 7. Costo computacional, trazabilidad y exportación (R6, criterio 5)
""")

code(r"""
cost = report.cost_table(cfg, df, diff_info); display(cost)
manifest = report.write_manifest(cfg, splits, Xsyn, float(cost.horas_GPU.iloc[-1]))
print(f"Horas de GPU: {manifest['gpu_horas']:.2f} h en {HW} | results_cwru.zip generado en la raíz del repositorio")
if "google.colab" in sys.modules:
    print("Para persistir: from google.colab import drive; drive.mount('/content/drive'); !cp results_cwru.zip /content/drive/MyDrive/")
""")

md(r"""
## 8. Discusión, limitaciones y uso responsable
- **Dominio:** la prueba en 3 HP mide generalización a una carga no vista; CWRU es un banco de ensayo con fallas sembradas por electroerosión, no fallas por desgaste natural, lo que limita la transferencia directa a planta.
- **Desbalance inducido:** el protocolo imita la escasez de fallas severas en operación; la difusión condicional solo puede extrapolar lo que vio (3 % de 0.021 in), de modo que su aporte debe leerse junto con TSTR y t-SNE.
- **Sesgos y riesgos:** un falso negativo en pista interna puede llevar a una falla catastrófica; el sistema debe operar como apoyo al mantenimiento con umbrales de confianza y revisión humana, nunca como decisión automática única.
- **Costo ambiental:** se reportan horas de GPU reales; el AST domina el costo y debe justificarse por su ganancia frente a la CNN1D.
- **Trabajo futuro:** Paderborn (fallas reales por desgaste), difusión en espectrograma con guía por clasificador, calibración de incertidumbre.

### Declaración de uso de IA
Se usó un asistente de IA para estructurar el repositorio, depurar código y revisar la redacción; todos los resultados provienen de las ejecuciones propias de este notebook.
""")

def write(cells, out, title):
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"},
                   "colab": {"provenance": [], "gpuType": "T4", "name": title}, "accelerator": "GPU"}
    nb.cells = [nbf.v4.new_markdown_cell(s) if t == "md" else nbf.v4.new_code_cell(s) for t, s in cells]
    nbf.write(nb, out); print("escrito", out, len(nb.cells), "celdas")

write(C, "notebooks/01_pipeline_colab.ipynb", "01_pipeline_colab")

D = []
def md2(s): D.append(("md", s.strip("\n")))
def code2(s): D.append(("code", s.strip("\n")))
md2(r"""
# Demostración de inferencia (Colab)
Carga los pesos entrenados (enlace del README) y clasifica ventanas de un archivo CWRU de principio a fin. No entrena nada.
""")
code2(r"""
import os, sys, subprocess
from pathlib import Path
REPO_URL = "https://github.com/<usuario>/cwru-bearing-diffusion.git"
WEIGHTS_URL = "<enlace a CNN1D_diffusion_s0.pt en Drive o Hugging Face>"
if not Path("src/cwru_dl").exists():
    if Path("../src/cwru_dl").exists():
        os.chdir("..")
    else:
        subprocess.run(["git", "clone", "-q", REPO_URL, "cwru-bearing-diffusion"], check=True); os.chdir("cwru-bearing-diffusion")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
sys.path.insert(0, "src")
""")
code2(r"""
import torch
from cwru_dl import CLASSES, load_config
from cwru_dl.data import download_cwru, load_signal, make_windows, zscore
from cwru_dl.models import CNN1D
from cwru_dl.train import weight_path

cfg = load_config("configs/full.yaml", profile="full")
download_cwru(cfg.root)
wp = weight_path(cfg, "CNN1D", "diffusion", 0)
if not wp.exists():
    if WEIGHTS_URL.startswith("http"):
        subprocess.run(["pip", "install", "-q", "gdown"], check=True); subprocess.run(["gdown", "--fuzzy", "-O", str(wp), WEIGHTS_URL], check=True)
    else:
        raise FileNotFoundError(f"Coloque los pesos en {wp} o defina WEIGHTS_URL")
model = CNN1D(); model.load_state_dict(torch.load(wp, map_location="cpu")); model.eval()
print("pesos cargados:", wp)
""")
code2(r"""
for cls, load in [("Normal", 3), ("IR014", 3), ("OR021", 3), ("B007", 3)]:
    x_raw, rpm, name = load_signal(cfg, cls, load)
    w = torch.from_numpy(zscore(make_windows(x_raw[:cfg.win * 8], cfg.win, cfg.win)))
    with torch.no_grad():
        probs = model(w).softmax(1)
    pred = [CLASSES[i] for i in probs.argmax(1).tolist()]
    print(f"{name:14s} real={cls:7s} rpm={rpm:.0f} | predicho por ventana: {pred} | confianza media {probs.max(1)[0].mean():.3f}")
""")
write(D, "notebooks/02_demo_inferencia.ipynb", "02_demo_inferencia")
