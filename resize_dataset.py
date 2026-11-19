"""Redimensiona o dataset uma única vez (OpenCV), mantendo a estrutura de pastas.

Uso:
    python3 resize_dataset.py --src data --dst data_384 --size 384

Depois, treine apontando --data para a pasta reduzida e use o mesmo --img:
    python3 train_cassava.py --data data_384 --img 384

Dica: o tamanho define o teto de detalhe que a rede enxerga. Sintomas sutis (CBSD, CGM) pedem
resolução: 384 ou mais costuma render bem mais que 224. Use 224 só para testes rápidos.
As imagens são deformadas para um quadrado (size x size), igual ao que o treino faz.
"""
import argparse
import os
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2

try:
    from tqdm import tqdm
except ImportError:  # sem tqdm roda igual, só sem a barra
    def tqdm(it, **kw):
        return it

cv2.setNumThreads(1)  # evita brigar com o Pool por CPU
EXTS = {".jpg", ".jpeg", ".png"}


def process(job):
    src, dst, size, quality = job
    if dst.exists():  # permite retomar se for interrompido
        return "pulado", str(src)
    img = cv2.imread(str(src))  # o OpenCV já aplica a orientação EXIF (foto de celular)
    if img is None:  # arquivo corrompido
        return "erro", str(src)
    interp = cv2.INTER_AREA if max(img.shape[:2]) > size else cv2.INTER_CUBIC
    img = cv2.resize(img, (size, size), interpolation=interp)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # grava num arquivo temporário e renomeia: uma interrupção não deixa JPEG pela metade
    tmp = dst.with_name(dst.stem + ".part.jpg")
    if not cv2.imwrite(str(tmp), img, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        return "erro", str(src)
    os.replace(tmp, dst)
    return "ok", str(src)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--quality", type=int, default=95, help="qualidade do JPEG de saída")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    a = p.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    if not src.is_dir():
        raise SystemExit(f"Pasta de origem '{a.src}' não existe (confira com pwd/ls).")
    rs, rd = src.resolve(), dst.resolve()
    if rd == rs or rs in rd.parents:
        raise SystemExit("--dst não pode ser a própria --src nem ficar dentro dela.")

    for sobra in dst.rglob("*.part.jpg"):  # restos de uma execução interrompida
        sobra.unlink()

    jobs, usados = [], set()
    for f in sorted(src.rglob("*")):
        if f.suffix.lower() not in EXTS:
            continue
        d = dst / f.relative_to(src).with_suffix(".jpg")
        if d in usados:  # a.jpg e a.png na mesma pasta: não deixa um sobrescrever o outro
            d = d.with_name(f"{d.stem}_{f.suffix[1:].lower()}.jpg")
        usados.add(d)
        jobs.append((f, d, a.size, a.quality))
    print(f"{len(jobs)} imagens encontradas")

    cont, erros = Counter(), []
    with Pool(a.workers) as pool:
        for status, arq in tqdm(pool.imap_unordered(process, jobs, chunksize=64),
                                total=len(jobs)):
            cont[status] += 1
            if status == "erro":
                erros.append(arq)

    for r in ("ok", "pulado", "erro"):
        print(r, cont[r])
    if erros:
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "_erros.txt").write_text("\n".join(erros), encoding="utf-8")
        print(f"Arquivos com erro listados em {dst / '_erros.txt'}")


if __name__ == "__main__":
    main()
