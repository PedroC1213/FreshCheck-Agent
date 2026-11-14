"""Ajuste fino de ResNet-50 o EfficientNet-B0 (ImageNet) para fresca vs. danada.

Modos:
  estandar          entrena en train, elige la mejor epoca en val y (solo con --evaluar-test) predice test UNA vez.
  fruta_no_vista    entrena sin la fruta indicada y evalua en esa fruta (train+val de esa fruta); sin seleccion de epoca.

Los aumentos (recorte aleatorio, volteos, brillo/contraste) se aplican SOLO a entrenamiento.
La particion de prueba solo se toca con --evaluar-test (corridas finales).
Uso:  python -m src.finetune --arq resnet50 --semilla 42 --lr 1e-4 --descongelar todo
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .config import Rutas
from .metricas import metricas

MEDIA, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


@dataclass
class Cfg:
    arq: str = "resnet50"
    semilla: int = 42
    lr: float = 1e-4
    epocas: int = 6
    descongelar: str = "todo"        # todo | ultimo_bloque
    lote: int = 32
    tam: int = 224                   # recorte de entrada
    lado: int = 256                  # lado de la cache
    modo: str = "estandar"           # estandar | fruta_no_vista
    fruta: str | None = None
    sin_pesos: bool = False          # solo para pruebas (no descarga pesos)
    evaluar_test: bool = False


def fijar_semilla(s: int):
    import torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cargar_datos(rutas: Rutas, lado: int = 256, ruta_cache: Path | None = None):
    import torch
    ruta_cache = ruta_cache or rutas.cache_imgs(lado)
    m = pd.read_csv(rutas.manifiesto).merge(pd.read_csv(rutas.particiones), on="id")
    ids = json.loads(ruta_cache.with_suffix(".ids.json").read_text(encoding="utf-8"))
    if m["id"].tolist() != ids:
        raise SystemExit("La cache no coincide con el manifiesto; vuelva a ejecutar python -m src.cache")
    imgs = torch.from_numpy(np.load(ruta_cache))
    y = (m["clase"] == "danada").astype(int).to_numpy()
    return m, imgs, y


def construir_modelo(arq: str, descongelar: str, pesos: bool = True):
    import torch.nn as nn
    from torchvision import models
    if arq == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pesos else None)
        net.fc = nn.Linear(net.fc.in_features, 2)
        cabeza, ultimo = [net.fc], [net.layer4]
    elif arq == "efficientnet_b0":
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pesos else None)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, 2)
        cabeza, ultimo = [net.classifier], [net.features[7], net.features[8]]
    else:
        raise ValueError(f"arquitectura desconocida: {arq}")
    entrenables = cabeza + ultimo
    if descongelar == "ultimo_bloque":
        for p in net.parameters():
            p.requires_grad = False
        for b in entrenables:
            for p in b.parameters():
                p.requires_grad = True
    return net, cabeza, entrenables


def _poner_modo_train(net, descongelar, entrenables):
    net.train()
    if descongelar == "ultimo_bloque":      # BatchNorm de las capas congeladas se mantiene fija
        net.eval()
        for b in entrenables:
            b.train()


def _normalizar(x, dev):
    import torch
    m = torch.tensor(MEDIA, device=dev).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    return (x - m) / s


def lote_train(imgs, idx, tam, dev, gen):
    import torch
    L = imgs.shape[1]
    x = imgs[idx]
    B = len(idx)
    out = torch.empty((B, 3, tam, tam), dtype=torch.uint8)
    tops = torch.randint(0, L - tam + 1, (B,), generator=gen)
    lefts = torch.randint(0, L - tam + 1, (B,), generator=gen)
    for k in range(B):
        c = x[k, int(tops[k]):int(tops[k]) + tam, int(lefts[k]):int(lefts[k]) + tam]
        if torch.rand(1, generator=gen).item() < 0.5:
            c = c.flip(1)
        if torch.rand(1, generator=gen).item() < 0.5:
            c = c.flip(0)
        out[k] = c.permute(2, 0, 1)
    x = out.to(dev).float() / 255
    b = (1 + torch.rand(B, 1, 1, 1, generator=gen) * 0.4 - 0.2).to(dev)
    c = (1 + torch.rand(B, 1, 1, 1, generator=gen) * 0.4 - 0.2).to(dev)
    mu = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (((x - mu) * c + mu) * b).clamp(0, 1)
    return _normalizar(x, dev)


def lote_eval(imgs, idx, tam, dev):
    L = imgs.shape[1]
    t = (L - tam) // 2
    x = imgs[idx][:, t:t + tam, t:t + tam].permute(0, 3, 1, 2).to(dev).float() / 255
    return _normalizar(x, dev)


def predecir(net, imgs, idx, tam, dev, lote=64):
    import torch
    net.eval()
    probs = []
    with torch.inference_mode():
        for i in range(0, len(idx), lote):
            logits = net(lote_eval(imgs, idx[i:i + lote], tam, dev)).float()
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs) if probs else np.array([])


def _pred_df(m, idx, p_danada):
    d = m.iloc[idx][["id", "fruta"]].copy()
    d["y_real"] = m.iloc[idx]["clase"].to_numpy()
    d["p_danada"] = p_danada
    d["y_pred"] = np.where(p_danada >= 0.5, "danada", "fresca")
    return d[["id", "fruta", "y_real", "y_pred", "p_danada"]].reset_index(drop=True)


def entrenar(cfg: Cfg, datos, dev: str = "cpu", verbose: bool = True) -> dict:
    import torch
    import torch.nn as nn
    m, imgs, y = datos
    fijar_semilla(cfg.semilla)
    part = m["particion"].to_numpy()
    fruta = m["fruta"].to_numpy()
    if cfg.modo == "fruta_no_vista":
        tr = np.nonzero((part == "train") & (fruta != cfg.fruta))[0]
        va = np.nonzero((fruta == cfg.fruta) & np.isin(part, ["train", "val"]))[0]
    else:
        tr, va = np.nonzero(part == "train")[0], np.nonzero(part == "val")[0]
    net, cabeza, entrenables = construir_modelo(cfg.arq, cfg.descongelar, pesos=not cfg.sin_pesos)
    net.to(dev)
    ids_cabeza = {id(p) for b in cabeza for p in b.parameters()}
    base = [p for p in net.parameters() if p.requires_grad and id(p) not in ids_cabeza]
    head = [p for b in cabeza for p in b.parameters()]
    opt = torch.optim.AdamW([{"params": base, "lr": cfg.lr}, {"params": head, "lr": cfg.lr * 10}], weight_decay=1e-4)
    pasos = cfg.epocas * math.ceil(len(tr) / cfg.lote)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, pasos))
    cuenta = np.bincount(y[tr], minlength=2).astype(float)
    pesos_clase = torch.tensor(len(tr) / (2 * np.maximum(cuenta, 1)), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=pesos_clase)
    usa_amp = dev == "cuda"
    escalador = torch.amp.GradScaler("cuda", enabled=usa_amp)
    gen = torch.Generator().manual_seed(cfg.semilla)
    yt = torch.tensor(np.asarray(y), dtype=torch.long)

    historial, mejor, mejor_estado, t0 = [], (-1.0, 0), None, time.time()
    for ep in range(1, cfg.epocas + 1):
        _poner_modo_train(net, cfg.descongelar, entrenables)
        perm = tr[torch.randperm(len(tr), generator=gen).numpy()]
        perdida = 0.0
        for i in range(0, len(perm), cfg.lote):
            idx = perm[i:i + cfg.lote]
            if len(idx) < 2:
                continue
            x = lote_train(imgs, idx, cfg.tam, dev, gen)
            t = yt[idx].to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=usa_amp):
                loss = crit(net(x), t)
            escalador.scale(loss).backward()
            escalador.step(opt)
            escalador.update()
            sched.step()
            perdida += loss.item() * len(idx)
        p_va = predecir(net, imgs, va, cfg.tam, dev)
        r = metricas(m.iloc[va]["clase"].to_numpy(), np.where(p_va >= 0.5, "danada", "fresca"))
        fila = {"epoca": ep, "loss_train": perdida / len(perm), "macro_f1_val": r["macro_f1"],
                "exactitud_val": r["exactitud"], "seg": round(time.time() - t0, 1)}
        historial.append(fila)
        if verbose:
            print(f"  [{cfg.arq} s{cfg.semilla} lr={cfg.lr:g} {cfg.descongelar}] ep {ep}/{cfg.epocas} "
                  f"loss {fila['loss_train']:.4f} | val macro-F1 {fila['macro_f1_val']:.4f} | {fila['seg']}s", flush=True)
        if cfg.modo == "estandar" and fila["macro_f1_val"] > mejor[0]:
            mejor = (fila["macro_f1_val"], ep)
            mejor_estado = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    if cfg.modo == "estandar" and mejor_estado is not None:
        net.load_state_dict(mejor_estado)
    mejor_ep = mejor[1] if cfg.modo == "estandar" else cfg.epocas
    p_va = predecir(net, imgs, va, cfg.tam, dev)
    res = {"config": asdict(cfg), "historial": historial, "mejor_epoca": mejor_ep, "segundos": round(time.time() - t0, 1),
           "n_train": int(len(tr)), "n_eval": int(len(va)),
           "val": metricas(m.iloc[va]["clase"].to_numpy(), np.where(p_va >= 0.5, "danada", "fresca")),
           "pred_val": _pred_df(m, va, p_va), "pred_test": None, "net": net}
    if cfg.evaluar_test and cfg.modo == "estandar":
        te = np.nonzero(part == "test")[0]
        p_te = predecir(net, imgs, te, cfg.tam, dev)
        res["pred_test"] = _pred_df(m, te, p_te)
        res["test"] = metricas(res["pred_test"]["y_real"].to_numpy(), res["pred_test"]["y_pred"].to_numpy())
    return res


def guardar_corrida(res: dict, carpeta: Path, nombre_pred: str = "val"):
    carpeta.mkdir(parents=True, exist_ok=True)
    res["pred_val"].to_csv(carpeta / f"pred_{nombre_pred}.csv", index=False)
    if res.get("pred_test") is not None:
        res["pred_test"].to_csv(carpeta / "pred_test.csv", index=False)
    resumen = {k: v for k, v in res.items() if k not in ("pred_val", "pred_test", "net")}
    (carpeta / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def guardar_modelo(net, ruta: Path, cfg: Cfg):
    import torch
    ruta.parent.mkdir(parents=True, exist_ok=True)
    estado = {k: (v.half() if v.is_floating_point() else v) for k, v in net.state_dict().items()}
    torch.save({"estado": estado, "arq": cfg.arq, "tam": cfg.tam, "clases": list(config.CLASES)}, ruta)


def dispositivo(pedido: str = "auto") -> str:
    import torch
    return "cuda" if pedido == "auto" and torch.cuda.is_available() else ("cpu" if pedido == "auto" else pedido)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", type=Path, default=config.ROOT)
    ap.add_argument("--arq", choices=config.ARQUITECTURAS, default="resnet50")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epocas", type=int, default=6)
    ap.add_argument("--descongelar", choices=("todo", "ultimo_bloque"), default="todo")
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--evaluar-test", action="store_true", help="SOLO corridas finales")
    ap.add_argument("--salida", type=Path, default=None)
    a = ap.parse_args(argv)
    rutas = Rutas(a.raiz)
    cfg = Cfg(arq=a.arq, semilla=a.semilla, lr=a.lr, epocas=a.epocas, descongelar=a.descongelar, lote=a.lote,
              evaluar_test=a.evaluar_test)
    dev = dispositivo()
    print(f"Dispositivo: {dev}")
    res = entrenar(cfg, cargar_datos(rutas, cfg.lado), dev)
    salida = a.salida or rutas.final / "manual" / f"{cfg.arq}_s{cfg.semilla}"
    guardar_corrida(res, salida)
    print(f"val macro-F1 {res['val']['macro_f1']:.4f} (mejor epoca {res['mejor_epoca']}) -> {salida}")


if __name__ == "__main__":
    main()
