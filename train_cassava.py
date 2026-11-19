"""Treino de CNN para classificação de doenças em folhas de mandioca.

Uso básico:
    python3 train_cassava.py --data data_384 --arch efficientnet_v2_s --img 384 --out runs/v2s

--data deve apontar para a pasta cujas SUBPASTAS são as classes
(CBB, CBSD, CGM, CMD, Healthy), no formato ImageFolder.

O que este script faz de importante:
  * split 70/15/15 estratificado SEM separar cópias da mesma foto (ver dedupe_dataset.py);
  * aquecimento + cosseno que de fato chega ao fim (o LR cai ao longo de --epochs);
  * EMA dos pesos (suaviza o treino e costuma dar um ganho extra de acurácia);
  * aumento de dados mais forte (recorte aleatório, apagamento) e CutMix;
  * retomada com --resume se o treino for interrompido (last.pt);
  * grava history.csv, curvas.png, split.json, best.pt e metricas_teste.json em --out.

Para um ensemble (o caminho mais barato para subir a acurácia), treine duas ou três
arquiteturas diferentes (--out diferentes) e combine-as no infer_cassava.py.
"""
import argparse
import copy
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

try:
    from tqdm import tqdm
except ImportError:  # sem tqdm o treino roda igual, só sem a barra
    def tqdm(it, **kw):
        return it

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def pick_device(choice="auto"):
    """'auto' usa a GPU (CUDA) se o PyTorch enxergar uma; 'cuda' exige GPU; 'cpu' força a CPU."""
    disponivel = torch.cuda.is_available()
    auto = choice == "auto"
    if auto:
        choice = "cuda" if disponivel else "cpu"
    if choice.startswith("cuda") and not disponivel:
        raise SystemExit(
            "O PyTorch não enxerga nenhuma GPU CUDA. Confira: (1) `nvidia-smi` mostra a placa? "
            "Se não, falta instalar o driver NVIDIA; (2) `python3 -c \"import torch; "
            "print(torch.version.cuda)\"` mostra um número? Se mostra None, o PyTorch instalado "
            "é a versão só de CPU e precisa ser reinstalado com suporte a CUDA.")
    device = torch.device(choice)
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        print(f"GPU: {p.name} ({p.total_memory / 2**30:.1f} GB) | "
              f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
    else:
        if auto:
            print("ATENÇÃO: o PyTorch não enxergou GPU; rodando na CPU (bem mais lento). "
                  "Use `--device cuda` para exigir a GPU e ver como corrigir.")
        else:
            print("Rodando na CPU, como pedido.")
    return device


def bf16_ok(device):
    """bf16 de verdade só existe em placas Ampere (capacidade 8.x) ou mais novas."""
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8


class SmallCNN(nn.Module):
    """Baseline treinada do zero: 5 blocos conv + BN + ReLU + MaxPool."""

    def __init__(self, n_classes):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256), block(256, 256)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.head(self.features(x))


# nome -> (função do torchvision, enum dos pesos pré-treinados, onde fica a camada final)
TV_ARCHS = {
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights", ("classifier", 1)),
    "efficientnet_b3": ("efficientnet_b3", "EfficientNet_B3_Weights", ("classifier", 1)),
    "efficientnet_v2_s": ("efficientnet_v2_s", "EfficientNet_V2_S_Weights", ("classifier", 1)),
    "efficientnet_v2_m": ("efficientnet_v2_m", "EfficientNet_V2_M_Weights", ("classifier", 1)),
    "convnext_tiny": ("convnext_tiny", "ConvNeXt_Tiny_Weights", ("classifier", 2)),
    "convnext_small": ("convnext_small", "ConvNeXt_Small_Weights", ("classifier", 2)),
    "resnet50": ("resnet50", "ResNet50_Weights", ("fc", None)),
}
ARCHS = ["smallcnn"] + list(TV_ARCHS)


def build_model(arch, n_classes, pretrained=True):
    """pretrained=False não baixa pesos (usado na inferência, que carrega o checkpoint)."""
    if arch == "smallcnn":
        return SmallCNN(n_classes)
    if arch not in TV_ARCHS:
        raise ValueError(f"arquitetura desconhecida: {arch}")
    fn, wname, (attr, pos) = TV_ARCHS[arch]
    w = getattr(models, wname).DEFAULT if pretrained else None
    m = getattr(models, fn)(weights=w)
    head = getattr(m, attr)
    if pos is None:
        setattr(m, attr, nn.Linear(head.in_features, n_classes))
    else:
        head[pos] = nn.Linear(head[pos].in_features, n_classes)
    return m


def cutmix(x, t, alpha=1.0):
    """Cola um retângulo de outra imagem do lote; devolve lote, rótulos trocados e lambda."""
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    h, w = x.shape[2:]
    r = np.sqrt(1 - lam)
    ch, cw = int(h * r), int(w * r)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
    x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)  # proporção real da imagem original
    return x, t[perm], lam


class ModelEMA:
    """Média móvel exponencial dos pesos. É ela que é avaliada e salva no best.pt."""

    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))  # começa "solta"
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:  # contadores inteiros do BatchNorm
                v.copy_(msd[k])


def make_schedule(opt, total_steps, warmup_steps, min_ratio=0.01):
    """Aquecimento linear e depois cosseno até min_ratio * lr, atualizado a cada passo."""
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def class_weights(counts, mode):
    """inverse: pesos inversamente proporcionais à frequência; sqrt: meio-termo; none: sem peso."""
    counts = np.asarray(counts, dtype=np.float64)
    if mode == "none":
        return np.ones(len(counts))
    w = counts.sum() / (len(counts) * counts)
    return np.sqrt(w) if mode == "sqrt" else w


def split_indices(y, groups, seed, n_folds=20):
    """Split 70/15/15 estratificado por classe sem separar imagens do mesmo grupo."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [te for _, te in sgkf.split(np.zeros(len(y)), y, groups)]
    te = np.sort(np.concatenate(folds[:3]))     # 3/20 = 15%
    va = np.sort(np.concatenate(folds[3:6]))    # 3/20 = 15%
    tr = np.sort(np.concatenate(folds[6:]))     # 14/20 = 70%
    for a_, b_ in ((tr, va), (tr, te), (va, te)):
        assert not set(groups[a_]) & set(groups[b_]), "um grupo caiu em dois conjuntos"
    return tr, va, te


def make_loaders(a, out, pin_memory=False):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop((a.img, a.img), scale=(a.min_scale, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation(20)], p=0.5),
        # jitter leve: a cor da folha faz parte do sintoma
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        # apagar pedaços força a rede a olhar várias regiões da folha, não só uma
        transforms.RandomErasing(p=a.erase, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((a.img, a.img)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    full_train = datasets.ImageFolder(a.data, transform=train_tf)
    full_eval = datasets.ImageFolder(a.data, transform=eval_tf)

    y = np.array(full_train.targets)
    if a.no_groups:
        groups = np.arange(len(y))  # cada imagem é seu próprio grupo (comportamento antigo)
        print("AVISO: split sem agrupar duplicatas; o teste pode estar otimista.")
    else:
        from dedupe_dataset import groups_for
        paths = [s[0] for s in full_train.samples]
        cache = Path(a.groups) if a.groups else out / "groups.json"
        groups = groups_for(paths, a.data, cache)
        extra = len(groups) - len(set(groups.tolist()))
        print(f"Duplicatas: {extra} imagens são cópias de outra (ficam no mesmo conjunto).")

    tr, va, te = split_indices(y, groups, a.seed)
    with open(out / "split.json", "w") as f:
        json.dump({"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()}, f)
    print(f"split: treino {len(tr)} | validação {len(va)} | teste {len(te)}")

    kw = dict(batch_size=a.bs, num_workers=a.workers, pin_memory=pin_memory)
    persist = dict(persistent_workers=a.workers > 0)
    loaders = (
        # drop_last: um último lote de 1 imagem quebraria o BatchNorm
        DataLoader(Subset(full_train, tr), shuffle=True, drop_last=True, **kw, **persist),
        DataLoader(Subset(full_eval, va), **kw),
        DataLoader(Subset(full_eval, te), **kw),
    )
    return loaders, full_train.classes, y[tr]


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, amp_dtype=None):
    model.eval()
    fmt = torch.channels_last if device.type == "cuda" else torch.contiguous_format
    total, preds, labels = 0.0, [], []
    for x, t in tqdm(loader, desc="avaliando", leave=False):
        x = x.to(device, non_blocking=True, memory_format=fmt)
        t = t.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(x).float()
        total += loss_fn(out, t).item() * len(t)
        preds += out.argmax(1).tolist()
        labels += t.tolist()
    f1 = f1_score(labels, preds, average="macro")
    return total / len(labels), f1, np.array(preds), np.array(labels)


def save_atomic(obj, path):
    """Grava num arquivo temporário e renomeia, para uma queda de energia não corromper."""
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_history(history, path):
    if not history:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)


def save_curves(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ep = [h["epoca"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(ep, [h["treino"] for h in history], label="treino (com aumento/CutMix)")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="validação")
    ax[0].set_title("Perda")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="acurácia")
    ax[1].plot(ep, [h["val_f1"] for h in history], label="F1-macro")
    ax[1].set_title("Validação")
    ax[2].plot(ep, [h["lr"] for h in history])
    ax[2].set_title("Taxa de aprendizado")
    for x in ax:
        x.set_xlabel("época")
        x.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--arch", default="efficientnet_v2_s", choices=ARCHS)
    p.add_argument("--img", type=int, default=384,
                   help="lado da imagem de entrada; use o mesmo --size do resize_dataset.py")
    p.add_argument("--epochs", type=int, default=30,
                   help="o LR cai ao longo de TODAS as épocas; não use um número exagerado")
    p.add_argument("--bs", type=int, default=32, help="se faltar memória da GPU, tente 16 e reduza o --lr")
    p.add_argument("--lr", type=float, default=3e-4)  # sugestão: 1e-3 para smallcnn
    p.add_argument("--wd", type=float, default=0.02, help="weight decay (não vale p/ bias e normalização)")
    p.add_argument("--warmup", type=float, default=2.0, help="épocas de aquecimento do LR")
    p.add_argument("--clip", type=float, default=1.0, help="limite da norma do gradiente (0 desliga)")
    p.add_argument("--ema", type=float, default=0.999, help="decaimento da média móvel dos pesos (0 desliga)")
    p.add_argument("--patience", type=int, default=0,
                   help="early stopping pelo F1-macro de validação; 0 desliga (recomendado: "
                        "deixa o cosseno chegar ao fim e o best.pt guarda a melhor época)")
    p.add_argument("--ls", type=float, default=0.1,
                   help="label smoothing (limita a confiança máxima da rede: 0.1 com 5 classes ~ 92%%)")
    p.add_argument("--cutmix", type=float, default=0.5,
                   help="probabilidade de aplicar CutMix em cada lote (0 desliga)")
    p.add_argument("--weights", default="inverse", choices=["inverse", "sqrt", "none"],
                   help="peso das classes na perda: inverse = balanceia totalmente (melhor "
                        "F1-macro); sqrt = meio-termo; none = melhor acurácia global")
    p.add_argument("--min-scale", type=float, default=0.6,
                   help="menor fração da folha mantida no recorte aleatório (muito baixo corta o sintoma)")
    p.add_argument("--erase", type=float, default=0.25, help="probabilidade de apagar um retângulo")
    p.add_argument("--groups", default=None,
                   help="groups.json do dedupe_dataset.py (padrão: calcula e salva em <out>/groups.json)")
    p.add_argument("--no-groups", action="store_true", help="não agrupa duplicatas (não recomendado)")
    p.add_argument("--workers", type=int, default=4,
                   help="processos que leem/aumentam as imagens; se a GPU ficar ociosa, aumente")
    p.add_argument("--device", default="auto",
                   help="auto (GPU se houver), cuda, cuda:1 ou cpu")
    p.add_argument("--amp", default="auto", choices=["auto", "fp16", "bf16", "off"],
                   help="precisão mista na GPU; auto = bf16 em placas Ampere ou mais novas, senão fp16")
    p.add_argument("--resume", action="store_true", help="continua do last.pt de --out, se existir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="runs/exp1")
    a = p.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = pick_device(a.device)
    on_gpu = device.type == "cuda"
    fmt = torch.channels_last if on_gpu else torch.contiguous_format
    if on_gpu:
        torch.backends.cudnn.benchmark = True  # entrada de tamanho fixo: usa os kernels mais rápidos
        torch.set_float32_matmul_precision("high")  # permite TF32 nas placas que suportam

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (train_dl, val_dl, test_dl), classes, y_train = make_loaders(a, out, pin_memory=on_gpu)
    counts = np.bincount(y_train, minlength=len(classes))
    print("classes:", classes, "| amostras de treino por classe:", counts.tolist())

    model = build_model(a.arch, len(classes)).to(device).to(memory_format=fmt)
    ema = ModelEMA(model, a.ema) if a.ema > 0 else None

    w = class_weights(counts, a.weights)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(w, dtype=torch.float32, device=device), label_smoothing=a.ls
    )
    # sem weight decay em bias e normalização (parâmetros de 1 dimensão)
    decay = [q for q in model.parameters() if q.ndim > 1]
    no_decay = [q for q in model.parameters() if q.ndim <= 1]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=a.lr, fused=on_gpu)
    steps_ep = len(train_dl)
    sched = make_schedule(opt, a.epochs * steps_ep, int(a.warmup * steps_ep))

    amp_dtype = None
    if on_gpu and a.amp != "off":
        if a.amp == "auto":
            amp_dtype = torch.bfloat16 if bf16_ok(device) else torch.float16
        elif a.amp == "bf16":
            amp_dtype = torch.bfloat16
            if not bf16_ok(device):
                print("Esta GPU não tem bf16 nativo; usando fp16.")
                amp_dtype = torch.float16
        else:
            amp_dtype = torch.float16
    use_amp = amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)
    print(f"Precisão mista: {str(amp_dtype).replace('torch.', '') if use_amp else 'desligada'} | "
          f"EMA: {a.ema if ema else 'desligada'} | pesos de classe: {a.weights}")

    best, wait, start_ep, history = -1.0, 0, 1, []
    last_path = out / "last.pt"
    if a.resume and last_path.exists():
        st = torch.load(last_path, map_location=device, weights_only=False)
        if st["args"]["arch"] != a.arch:
            raise SystemExit(f"last.pt é de {st['args']['arch']}, mas --arch é {a.arch}.")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        if ema and st.get("ema") is not None:
            ema.ema.load_state_dict(st["ema"])
            ema.updates = st["ema_updates"]
        best, wait, history = st["best"], st["wait"], st["history"]
        start_ep = st["epoch"] + 1
        print(f"Retomando da época {start_ep} (melhor F1 até aqui: {best:.4f})")
    elif a.resume:
        print("--resume pedido, mas não há last.pt em", out, "- começando do zero.")

    t_ini = time.time()
    for ep in range(start_ep, a.epochs + 1):
        model.train()
        run, seen = 0.0, 0
        t0 = time.time()
        bar = tqdm(train_dl, desc=f"ep {ep:02d}", leave=False)
        for x, t in bar:
            x = x.to(device, non_blocking=True, memory_format=fmt)
            t = t.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if a.cutmix > 0 and random.random() < a.cutmix:
                    x, t2, lam = cutmix(x, t)
                    out_ = model(x)
                    loss = lam * loss_fn(out_, t) + (1 - lam) * loss_fn(out_, t2)
                else:
                    loss = loss_fn(model(x), t)
            scaler.scale(loss).backward()
            if a.clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema:
                ema.update(model)
            run += loss.item() * len(t)
            seen += len(t)
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(loss=f"{loss.item():.3f}")

        eval_model = ema.ema if ema else model
        vl, vf1, vpred, vtrue = evaluate(eval_model, val_dl, loss_fn, device, amp_dtype)
        vacc = float((vpred == vtrue).mean())

        if vf1 > best:  # melhor época pelo F1-macro de validação
            best, wait = float(vf1), 0
            save_atomic({"model": eval_model.state_dict(), "classes": classes,
                         "arch": a.arch, "img": a.img, "ema": ema is not None,
                         "epoch": ep, "val_f1": float(vf1), "args": vars(a)}, out / "best.pt")
            marca = "* melhor"
        else:
            wait += 1
            marca = f"sem melhora {wait}" + (f"/{a.patience}" if a.patience else "")

        lr_now = opt.param_groups[0]["lr"]
        history.append({"epoca": ep, "treino": round(run / seen, 5), "val_loss": round(vl, 5),
                        "val_acc": round(vacc, 5), "val_f1": round(float(vf1), 5),
                        "lr": lr_now, "segundos": round(time.time() - t0, 1)})
        write_history(history, out / "history.csv")
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(),
                     "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                     "ema": ema.ema.state_dict() if ema else None,
                     "ema_updates": ema.updates if ema else 0,
                     "epoch": ep, "best": best, "wait": wait, "history": history,
                     "args": vars(a)}, last_path)

        print(f"ep {ep:02d} | treino {run / seen:.4f} | val {vl:.4f} | val acc {vacc:.4f} "
              f"| val F1-macro {vf1:.4f} | lr {lr_now:.2e} | {time.time() - t0:.0f}s | {marca}",
              flush=True)
        if a.patience > 0 and wait >= a.patience:
            print("early stopping")
            break

    save_curves(history, out / "curvas.png")
    print(f"Treino concluído em {(time.time() - t_ini) / 60:.1f} min")

    # avaliação final no teste, com o melhor checkpoint
    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    _, f1, pred, true = evaluate(model, test_dl, loss_fn, device, amp_dtype)
    acc = float((pred == true).mean())
    print(f"\nTESTE (época {ckpt['epoch']}) | acurácia {acc:.4f} | F1-macro {f1:.4f}")
    print(classification_report(true, pred, target_names=classes, digits=4))
    cm = confusion_matrix(true, pred)
    np.save(out / "confusion_matrix.npy", cm)
    print(cm)
    rel = classification_report(true, pred, target_names=classes, digits=4, output_dict=True)
    (out / "metricas_teste.json").write_text(json.dumps(
        {"acuracia": acc, "f1_macro": float(f1), "epoca": int(ckpt["epoch"]), "por_classe": rel},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nPróximo passo: python3 infer_cassava.py report --ckpt "
          f"{out / 'best.pt'} --data {a.data} --split test --calibrate --tta")


if __name__ == "__main__":
    main()
