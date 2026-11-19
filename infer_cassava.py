import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from train_cassava import MEAN, STD, build_model, pick_device, tqdm

EXTS = {".jpg", ".jpeg", ".png"}
MIN_ACEITAS = 50  # mínimo de imagens aceitas para confiar num limiar escolhido na validação


# ----------------------------------------------------------------- modelo ---
def get_device(choice="auto"):
    return pick_device(choice)


def load_model(ckpt_path, device):
    if not Path(ckpt_path).is_file():
        raise SystemExit(f"Checkpoint '{ckpt_path}' não encontrado. Use o caminho do arquivo "
                         f"best.pt (ex.: runs/v2s/best.pt), sem nada depois dele.")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt["classes"], ckpt["img"], ckpt["arch"]


def eval_transform(img):
    # idêntico ao eval_tf do treino
    return transforms.Compose([
        transforms.Resize((img, img)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_logits(model, x, tta=False):
    """Logits (B, K). Com TTA, tira a média sobre a imagem original e 3 espelhamentos."""
    views = [x]
    if tta:
        views += [x.flip(3), x.flip(2), x.flip((2, 3))]
    return torch.stack([model(v) for v in views]).mean(0)


@torch.no_grad()
def collect_logits(model, dl, device, tta):
    return torch.cat([get_logits(model, x.to(device, non_blocking=True), tta).cpu()
                      for x, _ in tqdm(dl, desc="inferência", leave=False)])


# ------------------------------------------------- calibração e confiança ---
def fit_temperature(logits, labels):
    """Temperature scaling: acha o T que minimiza a NLL na validação (não muda a classe prevista)."""
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def pick_threshold(conf, ok, target, min_n=MIN_ACEITAS):
    """Limiar de confiança que aceita o maior número de imagens mantendo acurácia >= target."""
    order = np.argsort(-conf)
    n_aceitas = np.arange(1, len(conf) + 1)
    cum_acc = np.cumsum(ok[order]) / n_aceitas
    good = np.where((cum_acc >= target) & (n_aceitas >= min_n))[0]
    return float(conf[order][good.max()]) if len(good) else None


def ece(conf, ok, bins=10):
    """Expected Calibration Error: diferença média entre confiança e acurácia real."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(total)


def wilson(k, n, z=1.96):
    """Intervalo de confiança de 95% para uma proporção."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def load_calib(ckpt_path):
    f = Path(ckpt_path).parent / "calibracao.json"
    return json.loads(f.read_text()) if f.exists() else {}


# --------------------------------------------------------------- Grad-CAM ---
def target_layer(model, arch):
    if arch == "resnet50":
        return model.layer4[-1]
    return model.features[-1]  # efficientnet_b0/v2_s, convnext_tiny e smallcnn


def gradcam(model, arch, x):
    """x: tensor (1,3,H,W). Mapa 0-1 (H x W) de onde a rede olhou para a classe prevista."""
    store = {}

    def fwd(_, __, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    handle = target_layer(model, arch).register_forward_hook(fwd)
    try:
        with torch.enable_grad():
            logits = model(x)
            c = int(logits.argmax(1))
            model.zero_grad(set_to_none=True)
            logits[0, c].backward()
    finally:
        handle.remove()

    a, g = store["a"].detach(), store["g"]
    w = g.mean(dim=(2, 3), keepdim=True)  # importância de cada canal
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.cpu().numpy()


def save_overlay(pil_img, cam, path, title):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4.3))
    ax[0].imshow(pil_img)
    ax[1].imshow(pil_img)
    ax[1].imshow(cam, cmap="jet", alpha=0.45)
    for x in ax:
        x.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- predict ---
def list_images(path):
    if path.is_file():
        return [path]
    return sorted(f for f in path.rglob("*") if f.suffix.lower() in EXTS)


def cmd_predict(a):
    if not Path(a.input).exists():
        raise SystemExit(f"O caminho '{a.input}' não existe. Confira o nome do arquivo ou da "
                         f"pasta e o diretório em que você está (comando pwd).")
    device = get_device(a.device)
    model, classes, img, arch = load_model(a.ckpt, device)
    tf = eval_transform(img)
    paths = list_images(Path(a.input))
    if not paths:
        raise SystemExit(f"Nenhuma imagem encontrada em {a.input}")

    calib = load_calib(a.ckpt)
    temp = calib.get("T", 1.0)
    use_tta = a.tta or calib.get("tta", False)
    thr = a.conf_min if a.conf_min is not None else (calib.get("limiar") or 0.6)
    print(f"Temperatura {temp:.3f} | TTA {'sim' if use_tta else 'não'} | "
          f"limiar de confiança {thr:.1%}\n")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, p in enumerate(paths):
        pil = Image.open(p).convert("RGB")
        x = tf(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            pr = F.softmax(get_logits(model, x, use_tta) / temp, dim=1)[0].cpu().numpy()
        order = pr.argsort()[::-1]
        top, second = order[0], order[1]
        conf = float(pr[top])
        status = "ok" if conf >= thr else "incerta (revisar)"
        print(f"{p.name}: {classes[top]} ({conf:.1%}) | 2ª opção: "
              f"{classes[second]} ({pr[second]:.1%}) | {status}")

        if a.gradcam:
            cam = gradcam(model, arch, x)
            save_overlay(pil.resize((img, img)), cam, out / f"{i:04d}_{p.stem}_gradcam.png",
                         f"{classes[top]} ({conf:.1%})")
        rows.append([str(p), classes[top], f"{conf:.4f}", status] + [f"{v:.4f}" for v in pr])

    with open(out / "predicoes.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "previsao", "confianca", "status"] + [f"prob_{c}" for c in classes])
        w.writerows(rows)
    print(f"\nTabela salva em {out / 'predicoes.csv'}")


# ----------------------------------------------------------------- report ---
def categoria(ok, conf, thr):
    if ok:
        return "acerto confiante" if conf >= thr else "acerto incerto"
    return "erro confiante" if conf >= thr else "erro incerto"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def plot_cm(cm, classes, path):
    k = len(classes)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, m, title, fmt in ((axes[0], cm, "Contagem", "d"),
                              (axes[1], norm, "Normalizada por classe real", ".2f")):
        ax.imshow(m, cmap="Blues")
        ax.set_xticks(range(k))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(k))
        ax.set_yticklabels(classes)
        ax.set_xlabel("Prevista")
        ax.set_ylabel("Real")
        ax.set_title(title)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, format(m[i, j], fmt), ha="center", va="center",
                        color="white" if m[i, j] > m.max() / 2 else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cmd_report(a):
    device = get_device(a.device)
    model, classes, img, arch = load_model(a.ckpt, device)
    tf = eval_transform(img)
    ds = datasets.ImageFolder(a.data, transform=tf)
    if ds.classes != classes:
        raise SystemExit(f"Classes da pasta {ds.classes} diferem das do checkpoint {classes}.")

    ckpt_dir = Path(a.ckpt).parent
    split_file = Path(a.split_file) if a.split_file else ckpt_dir / "split.json"
    splits = json.loads(split_file.read_text()) if split_file.exists() else {}
    if a.split == "all":
        idx = list(range(len(ds)))
    else:
        if a.split not in splits:
            raise SystemExit(f"{split_file} não encontrado ou sem a chave '{a.split}'.")
        idx = splits[a.split]
        if max(idx) >= len(ds):
            raise SystemExit("split.json não bate com --data: use a mesma pasta do treino.")

    out = Path(a.out) if a.out else ckpt_dir / f"relatorio_{a.split}"
    out.mkdir(parents=True, exist_ok=True)

    def logits_of(indices):
        dl = DataLoader(Subset(ds, indices), batch_size=a.bs, num_workers=a.workers,
                        pin_memory=device.type == "cuda")
        return collect_logits(model, dl, device, a.tta)

    # --- inferência no conjunto avaliado
    logits = logits_of(idx)
    true = np.array([ds.targets[i] for i in idx])
    paths = [ds.samples[i][0] for i in idx]
    rel = [os.path.relpath(p, a.data) for p in paths]

    # --- calibração: SEMPRE ajustada na validação, nunca no conjunto avaliado
    temp, limiar = 1.0, None
    if a.calibrate:
        if "val" not in splits:
            raise SystemExit("--calibrate precisa do split 'val' no split.json.")
        vlog = logits_of(splits["val"])
        vtrue = np.array([ds.targets[i] for i in splits["val"]])
        temp = fit_temperature(vlog, vtrue)
        vp = F.softmax(vlog / temp, dim=1).numpy()
        limiar = pick_threshold(vp.max(1), vp.argmax(1) == vtrue, a.target_acc)
        (ckpt_dir / "calibracao.json").write_text(json.dumps(
            {"T": temp, "tta": a.tta, "limiar": limiar, "meta_acc": a.target_acc}, indent=2))

    probs_raw = F.softmax(logits, dim=1).numpy()
    probs = F.softmax(logits / temp, dim=1).numpy()
    pred, conf = probs.argmax(1), probs.max(1)
    ok = pred == true

    if a.conf_min is not None:
        thr, thr_origem = a.conf_min, "definido por --conf-min"
    elif limiar is not None:
        thr, thr_origem = limiar, f"escolhido na validação para ≥{a.target_acc:.0%} de acurácia"
    else:
        thr, thr_origem = 0.6, "padrão"
    cats = [categoria(o, c, thr) for o, c in zip(ok, conf)]
    k, n = len(classes), len(idx)

    # --- tabela com uma linha por imagem
    with open(out / "predicoes.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "real", "prevista", "confianca", "resultado", "categoria"]
                   + [f"prob_{c}" for c in classes])
        for i in range(n):
            w.writerow([rel[i], classes[true[i]], classes[pred[i]], f"{conf[i]:.4f}",
                        "acerto" if ok[i] else "erro", cats[i]]
                       + [f"{v:.4f}" for v in probs[i]])

    # --- métricas
    acc = accuracy_score(true, pred)
    acc_lo, acc_hi = wilson(int(ok.sum()), n)
    f1m = f1_score(true, pred, average="macro", zero_division=0)
    f1w = f1_score(true, pred, average="weighted", zero_division=0)
    prec, rec, f1, sup = precision_recall_fscore_support(
        true, pred, labels=list(range(k)), zero_division=0)
    cm = confusion_matrix(true, pred, labels=list(range(k)))
    plot_cm(cm, classes, out / "matriz_confusao.png")

    # --- imagens separadas em pastas (opcional)
    if a.copy_images:
        for i in range(n):
            sub = (Path("acertos") / classes[true[i]] if ok[i]
                   else Path("erros") / f"{classes[true[i]]}_como_{classes[pred[i]]}")
            dst = out / sub
            dst.mkdir(parents=True, exist_ok=True)
            # prefixo com a confiança, para ordenar por ela no explorador de arquivos
            shutil.copy2(paths[i], dst / f"{conf[i]:.2f}_{Path(paths[i]).name}")

    # --- Grad-CAM dos erros mais confiantes (opcional)
    if a.gradcam:
        gdir = out / "erros_gradcam"
        gdir.mkdir(exist_ok=True)
        errs = sorted(np.where(~ok)[0], key=lambda i: -conf[i])[: a.max_gradcam]
        for j, i in enumerate(errs):
            pil = Image.open(paths[i]).convert("RGB")
            x = tf(pil).unsqueeze(0).to(device)
            cam = gradcam(model, arch, x)  # sempre para a classe prevista
            save_overlay(pil.resize((img, img)), cam, gdir / f"{j:03d}_{Path(paths[i]).stem}.png",
                         f"real: {classes[true[i]]} | prevista: {classes[pred[i]]} ({conf[i]:.1%})")

    # --- relatório em markdown
    cnt = Counter(cats)
    m_ok = f"{conf[ok].mean():.1%}" if ok.any() else "—"
    m_err = f"{conf[~ok].mean():.1%}" if (~ok).any() else "—"
    md = [f"# Relatório de classificação — conjunto `{a.split}`\n"]
    if a.split == "all":
        md.append("> Atenção: `all` inclui imagens usadas no treino; o resultado tende a ser "
                  "otimista. Para o artigo, use `--split test`.\n")
    if a.calibrate and a.split == "val":
        md.append("> Atenção: a calibração foi ajustada nesta mesma validação; o resultado "
                  "tende a ser otimista. Para o artigo, use `--split test`.\n")
    md += [
        f"- Modelo: `{arch}` (entrada {img}x{img}), checkpoint `{a.ckpt}`",
        f"- TTA: {'sim (original + 3 espelhamentos)' if a.tta else 'não'} | "
        f"temperatura de calibração: {temp:.3f}",
        f"- Imagens avaliadas: **{n}**",
        f"- Acertos: **{int(ok.sum())}** | Erros: **{int((~ok).sum())}**",
        f"- Acurácia global: **{acc:.2%}** (IC 95%: {acc_lo:.1%} a {acc_hi:.1%}) | "
        f"F1-macro: **{f1m:.4f}** | F1-ponderado: {f1w:.4f}",
        f"- Confiança média nos acertos: {m_ok} | nos erros: {m_err}",
        f"- Erro de calibração (ECE): {ece(probs_raw.max(1), ok):.3f} sem calibrar"
        + (f" → {ece(conf, ok):.3f} calibrado" if a.calibrate else ""),
        f"- Limiar de confiança usado nas categorias abaixo: {thr:.1%} ({thr_origem})",
        "",
        "## Acurácia por nível de certeza\n",
        "A acurácia global mistura imagens fáceis e difíceis. Aqui a rede só \"responde\" quando a "
        "confiança passa do limiar, e a cobertura mostra quantas imagens ela aceita responder.\n",
    ]
    if a.calibrate and limiar is None:
        md.append(f"> Na validação, a rede não chegou a {a.target_acc:.0%} de acurácia em nenhum "
                  f"limiar com pelo menos {MIN_ACEITAS} imagens aceitas.\n")

    rows_sel = []
    for t in sorted(set([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, round(thr, 4)])):
        m = conf >= t
        acc_m = f"{ok[m].mean():.1%}" if m.any() else "—"
        rows_sel.append([f"{t:.1%}" + (" ← usado" if abs(t - thr) < 1e-4 else ""),
                         f"{m.mean():.1%}", int(m.sum()), acc_m, int(m.sum() - ok[m].sum())])
    md.append(md_table(["Limiar", "Cobertura", "Imagens aceitas",
                        "Acurácia nas aceitas", "Erros entre as aceitas"], rows_sel))

    m_thr = conf >= thr
    if m_thr.any():
        lo, hi = wilson(int(ok[m_thr].sum()), int(m_thr.sum()))
        md.append(f"\nNo limiar usado, a rede responde {m_thr.mean():.1%} das imagens com acurácia "
                  f"de {ok[m_thr].mean():.2%} (IC 95%: {lo:.1%} a {hi:.1%}). As demais "
                  f"{(~m_thr).mean():.1%} ficam para revisão humana.\n")
    if not a.calibrate and probs_raw.max() < 0.95:
        md.append("> Nota: a confiança bruta desta rede nunca chegou a 95% (comum com label "
                  "smoothing). Use `--calibrate` para que o limiar faça sentido.\n")

    md += [
        "## Acertos e erros por categoria\n",
        md_table(["Categoria", "Quantidade", "% do total"],
                 [[c, cnt[c], f"{cnt[c] / n:.1%}"] for c in
                  ("acerto confiante", "acerto incerto", "erro confiante", "erro incerto")]),
        "\nErros confiantes são os mais perigosos: a rede errou e mesmo assim demonstrou "
        "segurança. Acertos incertos acertaram, mas com pouca margem.\n",
        "## Desempenho por classe\n",
        md_table(["Classe", "Suporte", "Acertos", "Erros", "Precisão", "Recall", "F1"],
                 [[classes[c], int(sup[c]), int(cm[c, c]), int(sup[c] - cm[c, c]),
                   f"{prec[c]:.3f}", f"{rec[c]:.3f}", f"{f1[c]:.3f}"] for c in range(k)]),
        "\n## Matriz de confusão (linhas: real, colunas: prevista)\n",
        md_table(["real \\ prevista"] + classes,
                 [[classes[i]] + [int(v) for v in cm[i]] for i in range(k)]),
        "\n![matriz de confusão](matriz_confusao.png)\n",
    ]

    conf_pairs = Counter((classes[t], classes[p]) for t, p in zip(true[~ok], pred[~ok]))
    if conf_pairs:
        md += ["## Confusões mais frequentes\n",
               md_table(["Real", "Prevista", "Ocorrências"],
                        [[t, p, c] for (t, p), c in conf_pairs.most_common(10)]), ""]

    err_idx = sorted(np.where(~ok)[0], key=lambda i: -conf[i])[: a.top]
    if len(err_idx):
        md += [f"## Erros mais confiantes (top {len(err_idx)})\n",
               md_table(["Arquivo", "Real", "Prevista", "Confiança"],
                        [[rel[i], classes[true[i]], classes[pred[i]], f"{conf[i]:.1%}"]
                         for i in err_idx]),
               "\nSe a confiança for alta e a imagem for claramente da classe prevista, vale "
               "revisar o rótulo: este dataset tem ruído de anotação.\n"]

    unc_idx = sorted(np.where(ok)[0], key=lambda i: conf[i])[: a.top]
    if len(unc_idx):
        md += [f"## Acertos mais incertos (top {len(unc_idx)})\n",
               md_table(["Arquivo", "Classe", "Confiança"],
                        [[rel[i], classes[true[i]], f"{conf[i]:.1%}"] for i in unc_idx]), ""]

    md += ["## Arquivos gerados\n",
           "- `predicoes.csv`: uma linha por imagem, com probabilidades de todas as classes",
           "- `matriz_confusao.png`: contagem e versão normalizada por classe"]
    if a.calibrate:
        md.append(f"- `{ckpt_dir / 'calibracao.json'}`: temperatura, TTA e limiar, lidos pelo `predict`")
    if a.copy_images:
        md.append("- `acertos/` e `erros/`: imagens separadas (nome começa com a confiança)")
    if a.gradcam:
        md.append("- `erros_gradcam/`: mapas de atenção dos erros mais confiantes")

    (out / "relatorio.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Acurácia global {acc:.2%} | F1-macro {f1m:.4f} | acertos {int(ok.sum())} | "
          f"erros {int((~ok).sum())}")
    if m_thr.any():
        print(f"Limiar {thr:.1%}: cobertura {m_thr.mean():.1%} | "
              f"acurácia nas aceitas {ok[m_thr].mean():.2%}")
    print(f"Relatório salvo em {out / 'relatorio.md'}")


# ------------------------------------------------------------------- main ---
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("report", help="relatório de acertos e erros em imagens rotuladas")
    pr.add_argument("--ckpt", required=True)
    pr.add_argument("--data", required=True, help="mesma pasta usada no treino")
    pr.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    pr.add_argument("--split-file", default=None, help="padrão: split.json ao lado do checkpoint")
    pr.add_argument("--out", default=None, help="padrão: <pasta do ckpt>/relatorio_<split>")
    pr.add_argument("--calibrate", action="store_true",
                    help="ajusta temperatura e limiar na validação e salva calibracao.json")
    pr.add_argument("--target-acc", type=float, default=0.95,
                    help="acurácia-alvo nas previsões aceitas (usada com --calibrate)")
    pr.add_argument("--conf-min", type=float, default=None,
                    help="força um limiar de confiança (padrão: o escolhido na calibração, ou 0.6)")
    pr.add_argument("--tta", action="store_true", help="média sobre original + 3 espelhamentos")
    pr.add_argument("--top", type=int, default=20, help="linhas nas tabelas de erros/incertos")
    pr.add_argument("--copy-images", action="store_true", help="copia imagens para acertos/ e erros/")
    pr.add_argument("--gradcam", action="store_true", help="Grad-CAM dos erros mais confiantes")
    pr.add_argument("--max-gradcam", type=int, default=50)
    pr.add_argument("--device", default="auto", help="auto (GPU se houver), cuda, cuda:1 ou cpu")
    pr.add_argument("--bs", type=int, default=32)
    pr.add_argument("--workers", type=int, default=4)
    pr.set_defaults(func=cmd_report)

    pp = sub.add_parser("predict", help="classifica imagens novas, uma a uma")
    pp.add_argument("--ckpt", required=True, help="best.pt gerado pelo treino")
    pp.add_argument("--input", required=True, help="uma imagem ou uma pasta de imagens")
    pp.add_argument("--out", default="predicoes")
    pp.add_argument("--conf-min", type=float, default=None,
                    help="abaixo disso a previsão é marcada como incerta "
                         "(padrão: limiar do calibracao.json, ou 0.6)")
    pp.add_argument("--tta", action="store_true", help="média sobre original + 3 espelhamentos")
    pp.add_argument("--gradcam", action="store_true", help="salva o mapa de atenção")
    pp.add_argument("--device", default="auto", help="auto (GPU se houver), cuda, cuda:1 ou cpu")
    pp.set_defaults(func=cmd_predict)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
