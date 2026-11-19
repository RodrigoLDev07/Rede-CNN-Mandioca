"""Auditoria de duplicatas: acha imagens idênticas ou quase idênticas e as agrupa.

Por que isso importa: se a mesma foto (ou uma cópia recomprimida, redimensionada ou espelhada)
cair no treino e no teste, a rede "decora" a imagem e a acurácia do artigo fica inflada. Este
script agrupa as cópias; o train_cassava.py usa os grupos para que cada grupo inteiro vá para um
único conjunto (treino, validação ou teste).

Uso:
    python3 dedupe_dataset.py --data data_384
    python3 train_cassava.py --data data_384 --groups groups.json ...

Também revela ruído de rótulo: se a mesma foto aparece em duas pastas de classes diferentes,
ela é listada como "conflito" no duplicados.csv.

Método: miniatura 16x16 em tons de cinza de cada imagem, comparada por correlação com todas as
outras nas 8 orientações (4 rotações de 90° x espelho). Só precisa de OpenCV e numpy.
Detecta: cópias exatas, recompressão JPEG, redimensionamento, mudança de brilho/contraste,
espelhamentos e rotações de 90°. Não detecta: recortes, zoom e rotações de ângulo livre.
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png"}
S = 16  # lado da miniatura em tons de cinza usada na comparação
BLOCO = 1024  # imagens comparadas por vez (limita a memória: ~90 MB)


def load_thumb(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    return cv2.resize(g, (S, S), interpolation=cv2.INTER_AREA)


def dihedral(t):
    """As 8 versões de uma imagem quadrada: 4 rotações de 90° da original e do espelho."""
    m = t[:, ::-1]
    return [np.rot90(t, k) for k in range(4)] + [np.rot90(m, k) for k in range(4)]


# Rotacionar/espelhar a miniatura é só permutar seus pixels, e permutar não muda média nem
# norma. Então basta guardar as 8 permutações e aplicá-las ao vetor já normalizado.
PERMS = [np.ascontiguousarray(v).ravel() for v in dihedral(np.arange(S * S).reshape(S, S))]


def normalizar(t):
    """Vetor de média zero e norma 1: o produto escalar de dois vetores é a correlação de Pearson."""
    v = t.astype(np.float32).ravel()
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros_like(v)  # imagem totalmente lisa: nunca casa


def compute_groups(paths, min_corr=0.97, workers=8):
    """Agrupa imagens duplicadas. Devolve (grupos, pares).

    grupos[i] é o id do grupo da imagem i (imagens sem cópia ficam sozinhas no grupo);
    pares é a lista de (i, j) confirmados como cópias.

    Compara a miniatura de cada imagem (nas 8 orientações) com as demais por correlação, em
    blocos de matrizes. O custo é fixo (~n² · 8 · 256 operações) e não depende do conteúdo.
    """
    n = len(paths)
    with ThreadPoolExecutor(workers) as ex:
        thumbs = list(ex.map(load_thumb, paths))
    ilegiveis = sum(t is None for t in thumbs)
    if ilegiveis:
        print(f"Aviso: {ilegiveis} arquivo(s) ilegível(is); ficam em grupos próprios.")
    O = np.stack([normalizar(t) if t is not None else np.zeros(S * S, np.float32)
                  for t in thumbs])

    pares = []
    for s in range(0, n, BLOCO):
        e = min(s + BLOCO, n)
        q, resto = O[s:e], O[s:]  # só compara com j >= s: a simetria cobre o resto
        for perm in PERMS:
            sim = q[:, perm] @ resto.T  # (e-s, n-s)
            r, c = np.nonzero(sim >= min_corr)
            i, j = s + r, s + c
            m = j > i
            pares.extend(zip(i[m].tolist(), j[m].tolist()))
    pares = sorted(set(pares))

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pares:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    ids, grupos = {}, np.zeros(n, dtype=np.int64)
    for i in range(n):
        grupos[i] = ids.setdefault(find(i), len(ids))
    return grupos, pares


def _rel(p, root):
    return os.path.relpath(p, root).replace(os.sep, "/")


def save_groups(paths, grupos, root, out):
    Path(out).write_text(json.dumps(
        {"root": str(root), "n": len(paths),
         "groups": {_rel(p, root): int(g) for p, g in zip(paths, grupos)}}), encoding="utf-8")


def groups_for(paths, root, cache, **kw):
    """Lê os grupos do arquivo `cache` se ele cobrir todas as imagens; senão calcula e salva."""
    cache = Path(cache)
    if cache.exists():
        mapa = json.loads(cache.read_text(encoding="utf-8")).get("groups", {})
        rels = [_rel(p, root) for p in paths]
        if all(r in mapa for r in rels):
            print(f"Grupos de duplicatas lidos de {cache}")
            return np.array([mapa[r] for r in rels], dtype=np.int64)
        print(f"{cache} não corresponde a esta pasta; recalculando.")
    print("Procurando imagens duplicadas (uma vez só; o resultado fica salvo)...")
    grupos, _ = compute_groups(paths, **kw)
    cache.parent.mkdir(parents=True, exist_ok=True)
    save_groups(paths, grupos, root, cache)
    return grupos


def write_report(paths, grupos, root, out):
    por_grupo = defaultdict(list)
    for i, g in enumerate(grupos):
        por_grupo[int(g)].append(i)
    multi = {g: v for g, v in por_grupo.items() if len(v) > 1}
    conflitos = 0
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["grupo", "tamanho", "classes", "conflito_de_rotulo", "arquivos"])
        for g, v in sorted(multi.items(), key=lambda kv: -len(kv[1])):
            rels = [_rel(paths[i], root) for i in v]
            classes = sorted({r.split("/")[0] for r in rels})
            conflito = len(classes) > 1
            conflitos += conflito
            w.writerow([g, len(v), "+".join(classes), "sim" if conflito else "não",
                        " | ".join(rels)])
    n_extra = sum(len(v) - 1 for v in multi.values())
    print(f"{len(paths)} imagens | {len(por_grupo)} grupos distintos")
    print(f"{len(multi)} grupos com cópias ({n_extra} imagens a mais que o necessário)")
    print(f"{conflitos} grupos com a MESMA foto em classes diferentes (ruído de rótulo)")
    print(f"Detalhes em {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="pasta no formato ImageFolder")
    p.add_argument("--out", default="groups.json")
    p.add_argument("--report", default="duplicados.csv")
    p.add_argument("--min-corr", type=float, default=0.97,
                   help="correlação mínima entre as miniaturas para considerar cópia "
                        "(0.97 é conservador; baixe para 0.95 para pegar cópias mais editadas)")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()

    root = Path(a.data)
    paths = sorted(str(f) for f in root.rglob("*") if f.suffix.lower() in EXTS)
    if not paths:
        raise SystemExit(f"Nenhuma imagem encontrada em {a.data}")
    print(f"{len(paths)} imagens encontradas")
    grupos, _ = compute_groups(paths, a.min_corr, a.workers)
    save_groups(paths, grupos, root, a.out)
    write_report(paths, grupos, root, a.report)
    print(f"Grupos salvos em {a.out}")


if __name__ == "__main__":
    main()
