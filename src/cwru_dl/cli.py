from __future__ import annotations

import argparse
import json

import matplotlib
import torch

from .config import CLASSES, DEVICE, HW, load_config
from .data import build_split, class_counts, download_cwru, load_meta, load_signal, make_windows, zscore
from .diffusion import run_diffusion_stage
from .interpret import ast_band_table, cam_frequency_table, tsne_real_vs_synthetic, tstr
from .models import CNN1D
from .train import has_run, load_records, load_trained, run_experiments, weight_path
from . import report


def _cfg(args):
    return load_config(args.config, profile=args.profile, root=args.data, out=args.out)


def cmd_data(args):
    cfg = _cfg(args)
    download_cwru(cfg.root)
    meta = load_meta(cfg); splits = build_split(cfg); dist = class_counts(splits)
    print(meta.pivot(index="clase", columns="carga_HP", values="muestras").loc[CLASSES]); print(dist)
    report.fig_distribution(cfg, dist); report.fig_envelope(cfg)


def cmd_diffusion(args):
    cfg = _cfg(args); download_cwru(cfg.root); splits = build_split(cfg)
    Xsyn, ysyn, info = run_diffusion_stage(cfg, splits)
    report.fig_diffusion_loss(cfg, info); report.fig_real_vs_synthetic(cfg, *splits["train"], Xsyn, ysyn)
    print(f"Sintéticas: {tuple(Xsyn.shape)} | entrenamiento {info['train_sec']:.0f} s | muestreo {info['sample_sec']:.0f} s")


def cmd_train(args):
    cfg = _cfg(args); download_cwru(cfg.root); splits = build_split(cfg)
    Xsyn, ysyn, _ = run_diffusion_stage(cfg, splits)
    run_experiments(cfg, splits, Xsyn, ysyn, models=args.models.split(","), augs=args.augs.split(","))


def cmd_report(args):
    cfg = _cfg(args); download_cwru(cfg.root); splits = build_split(cfg)
    Xte, yte = splits["test"]; Xva, yva = splits["val"]; Xtr, ytr = splits["train"]
    records = load_records(cfg)
    df, summary, recall = report.summary_tables(cfg, records); print(summary); print(recall)
    report.fig_results(cfg, df); report.fig_confusion(cfg, records); report.fig_curves(cfg, records)
    rpm3 = load_meta(cfg).query("carga_HP == 3").rpm.mean()
    if has_run(cfg, "CNN2D", "diffusion"):
        cam_model = load_trained(cfg, "CNN2D", "diffusion")
        tab = cam_frequency_table(cfg, cam_model, Xte, yte, rpm3); print(tab); tab.to_csv(cfg.out / "tables" / "tabla_gradcam_frecuencias.csv", index=False)
        report.fig_gradcam2d(cfg, cam_model, Xte, yte, rpm3)
    if has_run(cfg, "CNN1D", "diffusion"):
        report.fig_gradcam1d(cfg, load_trained(cfg, "CNN1D", "diffusion"), Xte, yte, rpm3)
    if has_run(cfg, "AST", "diffusion"):
        ast_model = load_trained(cfg, "AST", "diffusion")
        band = ast_band_table(cfg, ast_model, Xte, yte); band.to_csv(cfg.out / "tables" / "tabla_ast_atencion_bandas.csv")
        report.fig_ast_attention(cfg, ast_model, Xte, yte)
    Xsyn, ysyn, info = run_diffusion_stage(cfg, splits)
    if has_run(cfg, "CNN1D", "none"):
        report.fig_tsne(cfg, *tsne_real_vs_synthetic(cfg, Xtr, ytr, Xsyn, ysyn))
    t = tstr(cfg, Xsyn, ysyn, Xva, yva, Xte, yte); print(t); t.to_csv(cfg.out / "tables" / "tabla_tstr.csv", index=False)
    cost = report.cost_table(cfg, df, info); print(cost)
    report.write_manifest(cfg, splits, Xsyn, float(cost.horas_GPU.iloc[-1]))
    print("results_cwru.zip generado")


def cmd_demo(args):
    cfg = _cfg(args)
    demo = CNN1D(); demo.load_state_dict(torch.load(weight_path(cfg, "CNN1D", args.aug, cfg.seeds[0]), map_location="cpu")); demo.eval()
    x_raw, rpm, name = load_signal(cfg, args.cls, args.load)
    w = torch.from_numpy(zscore(make_windows(x_raw[:cfg.win * 8], cfg.win, cfg.win)))
    with torch.no_grad():
        probs = demo(w).softmax(1)
    print(f"Archivo {name} ({rpm:.0f} rpm) | predicción por ventana:", [CLASSES[i] for i in probs.argmax(1).tolist()])
    print(f"Confianza media: {probs.max(1)[0].mean():.3f}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="cwru_dl", description="CWRU: diagnóstico de fallas con aumento por difusión condicional")
    p.add_argument("--config", default=None, help="YAML en configs/ (se superpone a configs/base.yaml)")
    p.add_argument("--profile", default=None, choices=["smoke", "fast", "full"], help="por defecto: full con GPU, smoke sin GPU")
    p.add_argument("--data", default=None); p.add_argument("--out", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data", help="descarga, particiones y figuras 1-2").set_defaults(fn=cmd_data)
    sub.add_parser("diffusion", help="entrena el DDPM y genera las ventanas sintéticas").set_defaults(fn=cmd_diffusion)
    t = sub.add_parser("train", help="corre la rejilla modelos x aumentos x semillas"); t.set_defaults(fn=cmd_train)
    t.add_argument("--models", default="CNN1D,CNN2D,AST"); t.add_argument("--augs", default="none,classic,diffusion")
    sub.add_parser("report", help="tablas, figuras, interpretabilidad, TSTR, costo y manifest").set_defaults(fn=cmd_report)
    d = sub.add_parser("demo", help="inferencia con pesos guardados"); d.set_defaults(fn=cmd_demo)
    d.add_argument("--cls", default="OR014"); d.add_argument("--load", type=int, default=3); d.add_argument("--aug", default="diffusion")
    args = p.parse_args(argv)
    if args.cmd != "demo":
        matplotlib.use("Agg")
    print(f"device {DEVICE} | hardware {HW}")
    args.fn(args)


if __name__ == "__main__":
    main()
