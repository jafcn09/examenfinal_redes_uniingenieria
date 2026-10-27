from __future__ import annotations

import json
import shutil
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import welch

from .config import CFG, CLASSES, DEVICE, HW
from .data import envelope_spectrum, fault_freqs, load_signal
from .interpret import GradCAM, attention_rollout, dominant_freq, expected_freqs, mel_patch_labels
from .train import AUG_NAMES, load_trained, records_frame

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
MODEL_ORDER = ("CNN1D", "CNN2D", "AST")


def _save(cfg: CFG, name: str):
    plt.tight_layout(); plt.savefig(cfg.out / "figures" / name); plt.show(); plt.close()


def fmt(s: pd.Series) -> str:
    return f"{s.mean():.3f} ± {s.std(ddof=0):.3f}"


def fig_distribution(cfg: CFG, dist: pd.DataFrame):
    dist.to_csv(cfg.out / "tables" / "distribucion_particiones.csv")
    ax = dist.plot.bar(figsize=(9, 3.5), width=0.8); ax.set_ylabel("ventanas"); ax.set_title("Distribución por partición (desbalance inducido solo en train)")
    _save(cfg, "fig01_distribucion.png")


def fig_envelope(cfg: CFG):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ax, c in zip(axes.ravel(), ["Normal", "IR007", "OR007", "B007", "IR021", "OR021"]):
        x, rpm, _ = load_signal(cfg, c, 3)
        f, p = envelope_spectrum(x[:48000], cfg.fs); ff = fault_freqs(rpm)
        ax.semilogy(f, p, lw=0.8); ax.set_xlim(0, 600); ax.set_title(f"{c} @3HP ({rpm:.0f} rpm)")
        for key, col in [("BPFO", "tab:red"), ("BPFI", "tab:green"), ("BSF", "tab:orange")]:
            for h in (1, 2, 3):
                ax.axvline(ff[key] * h, color=col, ls="--", lw=0.9, alpha=0.9, label=key if h == 1 else None)
        ax.legend(fontsize=7); ax.set_xlabel("Hz")
    fig.suptitle("Espectro de envolvente (Hilbert) vs frecuencias teóricas de falla (SKF 6205)")
    _save(cfg, "fig02_envolvente_teorica.png")


def fig_diffusion_loss(cfg: CFG, info: dict):
    plt.figure(figsize=(5, 3)); plt.plot(info["history"]); plt.xlabel("época"); plt.ylabel("MSE(ε)"); plt.title("Entrenamiento del DDPM condicional")
    _save(cfg, "fig03_difusion_loss.png")


def fig_real_vs_synthetic(cfg: CFG, Xtr, ytr, Xsyn, ysyn, show=("IR021", "OR021", "B021")):
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.5))
    for j, c in enumerate(show):
        ci = CLASSES.index(c)
        xr = Xtr[ytr == ci][0].numpy(); xs = Xsyn[ysyn == ci][0].numpy()
        axes[0, j].plot(xr[:600], lw=0.7, label="real"); axes[0, j].plot(xs[:600] - 6, lw=0.7, label="sintética", color="tab:orange")
        axes[0, j].set_title(f"{c}: forma de onda (z-score)"); axes[0, j].legend(fontsize=7)
        fr, pr = welch(Xtr[ytr == ci].numpy(), fs=cfg.fs, nperseg=512, axis=1); fs_, ps = welch(Xsyn[ysyn == ci].numpy(), fs=cfg.fs, nperseg=512, axis=1)
        axes[1, j].semilogy(fr, pr.mean(0), label="real (media)"); axes[1, j].semilogy(fs_, ps.mean(0), label="sintética (media)", color="tab:orange")
        axes[1, j].set_xlabel("Hz"); axes[1, j].set_title("PSD media (Welch)"); axes[1, j].legend(fontsize=7)
    _save(cfg, "fig04_real_vs_sintetico.png")


def summary_tables(cfg: CFG, records: list):
    df = records_frame(records); df.to_csv(cfg.out / "tables" / "resultados_por_corrida.csv", index=False)
    aggs = dict(F1_macro=("test_f1_macro", fmt), **{f"F1_SNR{s}dB": (f"test_f1_snr{s}", fmt) for s in cfg.snr_eval},
                Accuracy=("test_acc", fmt), val_F1=("best_val_f1", fmt), min_GPU=("train_sec", lambda s: f"{s.sum()/60:.1f}"))
    models = [m for m in MODEL_ORDER if m in df.model.unique()]
    summary = df.groupby(["model", "aug"]).agg(**aggs).reindex(pd.MultiIndex.from_product([models, AUG_NAMES], names=["model", "aug"]))
    summary.to_csv(cfg.out / "tables" / "tabla_resumen.csv")
    recall = df.groupby(["model", "aug"])[[f"recall_{c}" for c in CLASSES]].mean().rename(columns=lambda c: c.replace("recall_", "")).round(3)
    recall.to_csv(cfg.out / "tables" / "tabla_recall_por_clase.csv")
    return df, summary, recall


def fig_results(cfg: CFG, df: pd.DataFrame):
    models = [m for m in MODEL_ORDER if m in df.model.unique()]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    mean_t = df.groupby(["model", "aug"]).test_f1_macro.mean().unstack("aug").reindex(index=models, columns=list(AUG_NAMES))
    std_t = df.groupby(["model", "aug"]).test_f1_macro.std(ddof=0).unstack("aug").reindex(index=models, columns=list(AUG_NAMES)).fillna(0)
    mean_t.plot.bar(ax=ax, yerr=std_t.values.T, capsize=3, width=0.75)
    ax.set_ylabel("F1-macro (3 HP)"); ax.set_ylim(0, 1.02); ax.set_title("Efecto del aumento por modelo (media ± std sobre semillas)"); ax.legend(title="aumento")
    _save(cfg, "fig05_f1_por_modelo_aumento.png")


def fig_confusion(cfg: CFG, records: list, model: str = "CNN1D"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, aname in zip(axes, AUG_NAMES):
        r = next((r for r in records if r["model"] == model and r["aug"] == aname and r["seed"] == cfg.seeds[0]), None)
        if r is None:
            ax.set_visible(False); continue
        cm = np.array(r["confusion"]); cmn = cm / cm.sum(1, keepdims=True).clip(1)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); ax.set_title(f"{model} / {aname} (F1 {r['test_f1_macro']:.3f})"); ax.grid(False)
        ax.set_xticks(range(10)); ax.set_yticks(range(10)); ax.set_xticklabels(CLASSES, rotation=90, fontsize=7); ax.set_yticklabels(CLASSES, fontsize=7)
        for i in range(10):
            for j in range(10):
                if cmn[i, j] > 0.05:
                    ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", fontsize=6, color="white" if cmn[i, j] > 0.6 else "black")
        ax.set_xlabel("predicho"); ax.set_ylabel("real")
    _save(cfg, "fig06_matrices_confusion.png")


def fig_curves(cfg: CFG, records: list):
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in records)]
    fig, axes = plt.subplots(1, len(models), figsize=(4.7 * len(models), 3.4), squeeze=False)
    for ax, mname in zip(axes[0], models):
        for aname in AUG_NAMES:
            r = next((r for r in records if r["model"] == mname and r["aug"] == aname and r["seed"] == cfg.seeds[0]), None)
            if r:
                ax.plot([h["val_f1"] for h in r["history"]], label=aname)
        ax.set_title(f"{mname}: F1 validación por época"); ax.set_xlabel("época"); ax.legend(fontsize=8)
    _save(cfg, "fig07_curvas_val.png")


def fig_gradcam2d(cfg: CFG, cam_model, Xte, yte, rpm3: float, show=("Normal", "IR021", "OR021", "B021")):
    cam2d = GradCAM(cam_model, cam_model.cam_layer); exp = expected_freqs(rpm3); hop = cam_model.frontend.hop
    freqs = np.fft.rfftfreq(cam_model.frontend.n_fft, 1 / cfg.fs)
    fig, axes = plt.subplots(2, len(show), figsize=(4 * len(show), 6.5))
    for j, c in enumerate(show):
        ci = CLASSES.index(c); i = int(torch.nonzero(yte == ci)[0]); x = Xte[i:i + 1]
        with torch.no_grad():
            S = cam_model.frontend(x.to(DEVICE)).squeeze().cpu().numpy()
        cam_raw = cam2d(x, torch.tensor([ci], device=DEVICE))
        cam = F.interpolate(cam_raw.unsqueeze(1), size=S.shape, mode="bilinear", align_corners=False).squeeze().numpy()
        t_axis = np.arange(S.shape[1]) * hop / cfg.fs * 1000; ext = [t_axis[0], t_axis[-1], freqs[0], freqs[-1]]
        axes[0, j].imshow(S, aspect="auto", origin="lower", cmap="magma", extent=ext); axes[0, j].set_title(f"{c}: log-STFT"); axes[0, j].grid(False)
        axes[1, j].imshow(S, aspect="auto", origin="lower", cmap="gray", extent=ext)
        axes[1, j].imshow(cam, aspect="auto", origin="lower", cmap="jet", alpha=0.5, extent=ext); axes[1, j].grid(False)
        sub = ""
        if c in exp:
            f_cam = dominant_freq(cam_raw[0].sum(0).numpy(), cam_raw.shape[-1] / (cfg.win / cfg.fs))
            sub = f" | CAM f={f_cam:.0f} Hz vs {exp[c][0]} {exp[c][1]:.0f} Hz"
        axes[1, j].set_title(f"Grad-CAM (CNN2D){sub}", fontsize=9); axes[1, j].set_xlabel("ms"); axes[0, j].set_ylabel("Hz")
    plt.suptitle("Grad-CAM sobre espectrograma: relevancia en bandas de resonancia e impactos periódicos")
    _save(cfg, "fig08_gradcam_2d.png")


def fig_gradcam1d(cfg: CFG, m1, Xte, yte, rpm3: float, show=("IR021", "OR021", "B021")):
    cam1d = GradCAM(m1, m1.cam_layer); exp = expected_freqs(rpm3)
    fig, axes = plt.subplots(1, len(show), figsize=(5 * len(show), 3))
    for ax, c in zip(axes, show):
        ci = CLASSES.index(c); i = int(torch.nonzero(yte == ci)[0]); x = Xte[i:i + 1]
        cam = F.interpolate(cam1d(x, torch.tensor([ci], device=DEVICE)).unsqueeze(1), size=cfg.win, mode="linear", align_corners=False).squeeze().numpy()
        t = np.arange(cfg.win) / cfg.fs * 1000
        ax.plot(t, x.squeeze().numpy(), lw=0.6, color="k"); ax.fill_between(t, -4, 4, where=cam > 0.5, color="tab:red", alpha=0.25)
        per = f", periodo {exp[c][0]} = {1000 / exp[c][1]:.1f} ms" if c in exp else ""
        ax.set_title(f"{c}: Grad-CAM 1D (rojo = relevancia > 0.5){per}", fontsize=9); ax.set_xlabel("ms"); ax.set_ylim(-4, 4)
    _save(cfg, "fig09_gradcam_1d.png")


def fig_ast_attention(cfg: CFG, ast_model, Xte, yte, show=("Normal", "IR021", "OR021", "B021")):
    labels = mel_patch_labels(cfg, ast_model.grid[0])
    fig, axes = plt.subplots(1, len(show), figsize=(4 * len(show), 3.4))
    for ax, c in zip(axes, show):
        ci = CLASSES.index(c); i = int(torch.nonzero(yte == ci)[0])
        A = attention_rollout(ast_model, Xte[i:i + 1])[0]
        ax.imshow(A, aspect="auto", origin="lower", cmap="viridis"); ax.grid(False)
        ax.set_yticks(range(A.shape[0])); ax.set_yticklabels([l.replace(" Hz", "") for l in labels], fontsize=7)
        ax.set_xlabel("parche temporal"); ax.set_ylabel("Hz (centro de parche mel)"); ax.set_title(f"AST rollout CLS: {c}")
    _save(cfg, "fig10_ast_atencion.png")


def fig_tsne(cfg: CFG, Z, lab, src):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ci, c in enumerate(CLASSES):
        for s, mk in (("real", "o"), ("sintético", "x")):
            m = (lab == ci) & (src == s)
            if m.any():
                axes[0].scatter(Z[m, 0], Z[m, 1], s=10, marker=mk, color=plt.cm.tab10(ci), label=f"{c} {s}" if s == "real" else None, alpha=0.7)
    axes[0].legend(fontsize=6, ncol=2); axes[0].set_title("t-SNE de embeddings CNN1D: color = clase, o = real, x = sintético")
    for s, col in (("real", "tab:blue"), ("sintético", "tab:orange")):
        m = src == s; axes[1].scatter(Z[m, 0], Z[m, 1], s=10, color=col, alpha=0.5, label=s)
    axes[1].legend(); axes[1].set_title("t-SNE: real vs sintético")
    _save(cfg, "fig11_tsne.png")


def cost_table(cfg: CFG, df: pd.DataFrame, diff_info: dict) -> pd.DataFrame:
    rows = [{"componente": f"Clasificadores ({len(df)} corridas)", "segundos": df.train_sec.sum()},
            {"componente": "DDPM entrenamiento", "segundos": diff_info["train_sec"]},
            {"componente": "DDPM muestreo", "segundos": diff_info["sample_sec"]}]
    cost = pd.DataFrame(rows); total = cost.segundos.sum()
    cost.loc[len(cost)] = ["TOTAL", total]
    cost["minutos"] = (cost.segundos / 60).round(1); cost["horas_GPU"] = (cost.segundos / 3600).round(3); cost["hardware"] = HW
    cost.to_csv(cfg.out / "tables" / "tabla_costo.csv", index=False)
    return cost


def write_manifest(cfg: CFG, splits: dict, Xsyn, gpu_hours: float) -> dict:
    manifest = {"fecha": time.strftime("%Y-%m-%d %H:%M"), "hardware": HW, "torch": torch.__version__, "perfil": cfg.profile,
                "config": cfg.as_dict(), "n_train": int(len(splits["train"][0])), "n_sinteticas": int(len(Xsyn)),
                "n_val": int(len(splits["val"][0])), "n_test": int(len(splits["test"][0])), "clases": CLASSES, "gpu_horas": gpu_hours,
                "tablas": sorted(p.name for p in (cfg.out / "tables").glob("*.csv")), "figuras": sorted(p.name for p in (cfg.out / "figures").glob("*.png")),
                "pesos": sorted(p.name for p in (cfg.out / "weights").glob("*.pt"))}
    (cfg.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    shutil.make_archive(str(cfg.out.parent / "results_cwru"), "zip", cfg.out)
    return manifest
