"""Genera las tablas LaTeX y copia las figuras del artículo a partir de results/.

Uso (desde la raíz del repositorio, después de `python -m cwru_dl --config configs/full.yaml report`):
    python paper/make_tables.py            # lee results/, escribe paper/generated/*.tex y paper/figs/*.png
    python paper/make_tables.py --results /ruta/a/results

Así cada número del artículo sale de un archivo de results/ (lista de verificación del enunciado).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

PAPER = Path(__file__).resolve().parent
MODEL_NAMES = {"CNN1D": "CNN 1D", "CNN2D": "CNN 2D", "AST": "AST"}
AUG_NAMES = {"none": "Ninguno", "classic": "Clásico", "diffusion": "Difusión"}
FIGS = ["fig01_distribucion", "fig02_envolvente_teorica", "fig03_difusion_loss", "fig04_real_vs_sintetico",
        "fig05_f1_por_modelo_aumento", "fig06_matrices_confusion", "fig07_curvas_val", "fig08_gradcam_2d",
        "fig09_gradcam_1d", "fig10_ast_atencion", "fig11_tsne"]


def tex(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("%", r"\%"), ("_", r"\_"), ("&", r"\&"), ("#", r"\#"), ("±", r"$\pm$")):
        s = s.replace(a, b)
    return "--" if s in ("nan", "") else s


def tabular(df: pd.DataFrame, align: str | None = None) -> str:
    align = align or "l" + "c" * (df.shape[1] - 1)
    rows = [" & ".join(tex(c) for c in df.columns) + r" \\ \midrule"]
    rows += [" & ".join(tex(v) for v in r) + r" \\" for r in df.itertuples(index=False)]
    return "\\begin{tabular}{" + align + "}\n\\toprule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def write(name: str, body: str) -> None:
    (PAPER / "generated" / f"{name}.tex").write_text(body, encoding="utf-8")
    print("  generated/" + name + ".tex")


def bold_best(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Marca en negrita la mejor media de una columna 'media ± desv'."""
    means = pd.to_numeric(df[col].astype(str).str.split("±").str[0], errors="coerce")
    if means.notna().any():
        i = means.idxmax()
        df.loc[i, col] = "BOLD" + str(df.loc[i, col])
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(PAPER.parent / "results"))
    R = Path(ap.parse_args().results)
    T = R / "tables"
    (PAPER / "generated").mkdir(exist_ok=True); (PAPER / "figs").mkdir(exist_ok=True)

    if (T / "distribucion_particiones.csv").exists():
        d = pd.read_csv(T / "distribucion_particiones.csv").rename(columns={"Unnamed: 0": "Clase"})
        d.columns = ["Clase", "Train (0--2 HP)", "Val (0--2 HP)", "Test (3 HP)"]
        d.loc[len(d)] = ["Total", *d.iloc[:, 1:].sum().tolist()]
        write("tabla_particiones", tabular(d))

    if (T / "tabla_resumen.csv").exists():
        s = pd.read_csv(T / "tabla_resumen.csv").fillna("")
        s["model"] = s["model"].map(MODEL_NAMES).fillna(s["model"]); s["aug"] = s["aug"].map(AUG_NAMES).fillna(s["aug"])
        s = s[["model", "aug", "F1_macro", "F1_SNR10dB", "F1_SNR0dB", "Accuracy"]]
        s.columns = ["Modelo", "Aumento", "F1-macro", "F1 (10 dB)", "F1 (0 dB)", "Exactitud"]
        for c in ("F1-macro", "F1 (10 dB)", "F1 (0 dB)"):
            s = bold_best(s, c)
        write("tabla_resumen", _bold_tex(tabular(s, "llcccc")))

    if (T / "tabla_recall_por_clase.csv").exists():
        r = pd.read_csv(T / "tabla_recall_por_clase.csv")
        r["model"] = r["model"].map(MODEL_NAMES).fillna(r["model"]); r["aug"] = r["aug"].map(AUG_NAMES).fillna(r["aug"])
        cols = ["model", "aug", "B021", "IR021", "OR021", "B014", "IR014", "OR014"]
        r = r[[c for c in cols if c in r.columns]].rename(columns={"model": "Modelo", "aug": "Aumento"})
        write("tabla_recall", tabular(r.round(3), "ll" + "c" * (r.shape[1] - 2)))

    if (T / "tabla_gradcam_frecuencias.csv").exists():
        g = pd.read_csv(T / "tabla_gradcam_frecuencias.csv").round(1)
        write("tabla_gradcam", "\\resizebox{\\columnwidth}{!}{%\n" + tabular(g) + "}\n")

    if (T / "tabla_tstr.csv").exists():
        t = pd.read_csv(T / "tabla_tstr.csv").round(3)
        write("tabla_tstr", tabular(t))

    if (T / "tabla_costo.csv").exists():
        c = pd.read_csv(T / "tabla_costo.csv")
        c = c[["componente", "minutos", "horas_GPU"]].round(3)
        c.columns = ["Componente", "Minutos", "Horas GPU"]
        write("tabla_costo", tabular(c, "lcc"))

    if (R / "manifest.json").exists():
        m = json.loads((R / "manifest.json").read_text())
        v = m.get("versiones", {})
        hw = m.get("hardware", "--")
        line = (f"\\renewcommand{{\\HW}}{{{tex(hw)}}}\n\\renewcommand{{\\GPUH}}{{{m.get('gpu_horas', 0):.2f}}}\n"
                f"\\renewcommand{{\\NTRAIN}}{{{m.get('n_train', '--')}}}\n\\renewcommand{{\\NSYN}}{{{m.get('n_sinteticas', '--')}}}\n"
                f"\\renewcommand{{\\NVAL}}{{{m.get('n_val', '--')}}}\n\\renewcommand{{\\NTEST}}{{{m.get('n_test', '--')}}}\n"
                f"\\renewcommand{{\\VTORCH}}{{{tex(v.get('torch', m.get('torch', '--')))}}}\n"
                f"\\renewcommand{{\\VTRANSF}}{{{tex(v.get('transformers', '--'))}}}\n")
        write("numeros", line)

    for f in FIGS:
        src = R / "figures" / f"{f}.png"
        if src.exists():
            shutil.copy(src, PAPER / "figs" / src.name)
            print("  figs/" + src.name)


def _bold_tex(s: str) -> str:
    out = []
    for cell in s.split(" & "):
        if "BOLD" in cell:
            end = cell.find(r" \\")
            body, rest = (cell[:end], cell[end:]) if end >= 0 else (cell, "")
            cell = "\\textbf{" + body.replace("BOLD", "") + "}" + rest
        out.append(cell)
    return " & ".join(out)


if __name__ == "__main__":
    main()
