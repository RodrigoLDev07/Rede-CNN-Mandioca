#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Menu para o pipeline de classificação de doenças em folhas de mandioca.

Uso (na pasta do projeto, onde ficam os scripts e a pasta "data"):
    python3 menu_cassava.py
    python3 menu_cassava.py --diagnostico      # só mostra o que encontrou e sai
    python3 menu_cassava.py --pular-gpu        # não testa o PyTorch/GPU na partida (mais rápido)
    python3 menu_cassava.py --root /outra/pasta --scripts-dir /pasta/dos/scripts

O que o menu faz sozinho:
  * localiza os scripts (resize_dataset.py, dedupe_dataset.py, train_cassava.py, infer_cassava.py);
  * confere as bibliotecas e se o PyTorch enxerga a GPU;
  * procura pastas de imagens no formato ImageFolder (uma subpasta por classe), diz quantas
    imagens e classes há em cada uma e se parecem originais ou já redimensionadas;
  * procura pastas com imagens soltas (para classificar fotos novas);
  * procura treinos já feitos (best.pt) e o que existe ao lado deles (relatório, calibração...);
  * sugere o próximo passo, monta o comando de cada operação, mostra-o e só então executa;
  * grava cada comando executado em menu_historico.txt (útil para a seção de métodos do artigo).

Para incluir um script novo no futuro, basta acrescentá-lo ao dicionário SCRIPTS abaixo.
Este arquivo usa só a biblioteca padrão: não importa PyTorch nem OpenCV (por isso abre rápido).
"""
import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ----------------------------------------------------------------------------- configuração
EXTS = {".jpg", ".jpeg", ".png"}
CLASSES_ARTIGO = {"cbb", "cbsd", "cgm", "cmd", "healthy"}
CLASSES_CASSAVA_FULL = {
    "cassava___bacterial_blight",
    "cassava___brown_streak_disease",
    "cassava___green_mottle",
    "cassava___healthy",
    "cassava___mosaic_disease",
}
MAX_PROF = 150            # profundidade máxima da busca por imagens
MAX_ENTRADAS = 400_000_000  # trava de segurança para o caso de rodar numa pasta gigante
HISTORICO = "menu_historico.txt"

# chave -> (arquivo, descrição, módulos Python que o script exige)
SCRIPTS = {
    "resize": ("resize_dataset.py", "redimensiona o dataset (OpenCV)", ["cv2"]),
    "dedupe": ("dedupe_dataset.py", "acha fotos duplicadas e agrupa", ["cv2", "numpy"]),
    "treino": ("train_cassava.py", "treina a CNN",
               ["torch", "torchvision", "sklearn", "numpy"]),
    "infer": ("infer_cassava.py", "relatório, classificação de fotos novas e Grad-CAM",
              ["torch", "torchvision", "sklearn", "numpy", "matplotlib", "PIL"]),
}
# módulo -> nomes possíveis do pacote no pip (para mostrar a versão e a dica de instalação)
PACOTES = {
    "cv2": ["opencv-python", "opencv-python-headless", "opencv-contrib-python"],
    "sklearn": ["scikit-learn"],
    "PIL": ["pillow", "Pillow"],
    "torch": ["torch"],
    "torchvision": ["torchvision"],
    "numpy": ["numpy"],
    "matplotlib": ["matplotlib"],
    "tqdm": ["tqdm"],
}
PIP_NOME = {"cv2": "opencv-python", "sklearn": "scikit-learn", "PIL": "pillow"}
OPCIONAIS = {"tqdm"}  # só melhoram a experiência

# (nome, comentário) das arquiteturas; só aparecem as que o train_cassava.py realmente tiver
ARQUITETURAS = [
    ("efficientnet_v2_s", "recomendada: boa acurácia com custo moderado"),
    ("convnext_tiny", "ótima segunda opção (boa para ensemble)"),
    ("efficientnet_b0", "leve e rápida; bom ponto de partida"),
    ("efficientnet_b3", "intermediária"),
    ("convnext_small", "maior; pede mais memória de GPU"),
    ("efficientnet_v2_m", "a mais pesada; pede bastante memória de GPU"),
    ("resnet50", "clássica; útil para comparar no artigo"),
    ("smallcnn", "baseline treinada do zero (para comparação)"),
]

IGNORAR_EXATOS = {"__pycache__", "node_modules", "site-packages", "runs", "venv", "env",
                  "logs", "acertos", "erros"}
IGNORAR_PREFIXOS = (".", "relatorio_", "explicacao_", "erros_", "predicoes")

UTF = "utf" in (getattr(sys.stdout, "encoding", "") or "").lower()
OK, NAO, SETA = ("✓", "✗", "→") if UTF else ("[ok]", "[--]", "->")


# ----------------------------------------------------------------------------- utilidades
def fmt_n(n):
    return f"{n:,}".replace(",", ".")


def ler_json(caminho):
    try:
        return json.loads(Path(caminho).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def sim_nao(txt, padrao=True):
    suf = "S/n" if padrao else "s/N"
    while True:
        r = input(f"{txt} [{suf}]: ").strip().lower()
        if not r:
            return padrao
        if r in ("s", "sim", "y", "yes"):
            return True
        if r in ("n", "nao", "não", "no"):
            return False
        print("Responda s ou n.")


def perguntar(txt, padrao=None):
    suf = f" [{padrao}]" if padrao not in (None, "") else ""
    r = input(f"{txt}{suf}: ").strip()
    return r if r else ("" if padrao is None else str(padrao))


def perguntar_caminho(txt, padrao=None):
    return os.path.expanduser(perguntar(txt, padrao).strip("'\""))


def perguntar_numero(txt, padrao, tipo=int, minimo=None, maximo=None):
    while True:
        r = perguntar(txt, padrao).replace(",", ".")
        try:
            v = tipo(r)
        except ValueError:
            print("Digite um número.")
            continue
        if (minimo is not None and v < minimo) or (maximo is not None and v > maximo):
            print(f"Use um valor entre {minimo} e {maximo}.")
            continue
        return v


def escolher(titulo, opcoes, padrao=1, multi=False, voltar=True):
    """Mostra opções numeradas. Devolve o índice (ou lista de índices se multi); None se voltar."""
    print(f"\n{titulo}")
    for i, o in enumerate(opcoes, 1):
        print(f"  {i}) {o}")
    if voltar:
        print("  0) Voltar / cancelar")
    dica = " (vários: 1,2)" if multi else ""
    while True:
        r = input(f"Escolha{dica} [{padrao}]: ").strip() or str(padrao)
        if r == "0" and voltar:
            return None
        try:
            nums = [int(x) for x in r.replace(",", " ").split()]
        except ValueError:
            nums = []
        if nums and all(1 <= n <= len(opcoes) for n in nums) and (multi or len(nums) == 1):
            return [n - 1 for n in nums] if multi else nums[0] - 1
        print("Opção inválida.")


# ----------------------------------------------------------------------------- varredura de imagens
@dataclass
class No:
    caminho: Path
    diretas: int = 0                       # imagens diretamente nesta pasta
    total: int = 0                         # imagens nesta pasta e abaixo
    filhos: list = field(default_factory=list)   # só subpastas que contêm imagens
    amostras: list = field(default_factory=list)


@dataclass
class Dataset:
    caminho: Path
    classes: dict
    total: int
    tamanho: tuple = None
    reduzido: bool = None   # True: quadradas e pequenas (parecem redimensionadas); None: não deu para saber


@dataclass
class Treino:
    pasta: Path
    best: Path
    mtime: float
    split_total: int = None
    metricas: dict = None
    calib: dict = None
    epocas: int = None
    tem_last: bool = False
    relatorios: list = field(default_factory=list)
    explicacoes: list = field(default_factory=list)


@dataclass
class Estado:
    scripts: dict
    modulos: dict
    gpu: dict
    datasets: list
    soltas: list
    treinos: list
    groups: dict
    truncado: bool


def deve_ignorar(nome):
    n = nome.lower()
    if n == "data" or n.startswith("cassava___"):
        return False
    return n in IGNORAR_EXATOS or n.startswith(IGNORAR_PREFIXOS)


def varrer(caminho, prof, orc):
    no = No(Path(caminho))
    try:
        with os.scandir(caminho) as it:
            entradas = sorted(it, key=lambda e: e.name)
    except OSError:
        return no
    for e in entradas:
        orc["n"] += 1
        if orc["n"] > MAX_ENTRADAS:
            orc["truncado"] = True
            break
        try:
            if e.is_dir():
                if (prof < MAX_PROF and not deve_ignorar(e.name)
                        and not os.path.exists(os.path.join(e.path, "pyvenv.cfg"))):
                    f = varrer(e.path, prof + 1, orc)
                    if f.total:
                        no.filhos.append(f)
                        no.total += f.total
            elif os.path.splitext(e.name)[1].lower() in EXTS:
                no.diretas += 1
                no.total += 1
                if len(no.amostras) < 2:
                    no.amostras.append(Path(e.path))
        except OSError:
            continue
    return no


def _tem_subpastas_de_classe(no):
    """Verifica se os filhos diretos parecem classes (ex.: Cassava___* ou subpastas com imagens)."""
    if no.diretas != 0 or len(no.filhos) < 2:
        return False
    nomes = {f.caminho.name.lower() for f in no.filhos}
    if nomes & CLASSES_CASSAVA_FULL or nomes & CLASSES_ARTIGO:
        return True
    return all(f.total > 0 for f in no.filhos)


def eh_dataset(no, memo):
    """Pasta cujas subpastas são classes: >= 2 subpastas com imagens, sem imagens soltas e sem
    nenhum dataset mais abaixo (evita tomar a pasta do projeto por um dataset)."""
    chave = ("ds", no.caminho)
    if chave not in memo:
        memo[chave] = (
            no.diretas == 0
            and len(no.filhos) >= 2
            and not any(contem_dataset(f, memo) for f in no.filhos)
            and _tem_subpastas_de_classe(no)
        )
    return memo[chave]


def contem_dataset(no, memo):
    chave = ("contem", no.caminho)
    if chave not in memo:
        memo[chave] = eh_dataset(no, memo) or any(contem_dataset(f, memo) for f in no.filhos)
    return memo[chave]


def coletar(no, memo, datasets, soltas):
    if eh_dataset(no, memo):
        datasets.append(no)
        return
    if no.diretas > 0:
        soltas.append(no)
    for f in no.filhos:
        coletar(f, memo, datasets, soltas)


def tamanho_imagem(p):
    try:
        from PIL import Image
        with Image.open(p) as im:
            return im.size
    except ImportError:
        pass
    except Exception:
        return None
    try:
        import cv2
        im = cv2.imread(str(p))
        if im is not None:
            return (im.shape[1], im.shape[0])
    except ImportError:
        pass
    return None


def montar_dataset(no):
    tams = []
    for f in no.filhos[:5]:
        for p in f.amostras:
            t = tamanho_imagem(p)
            if t:
                tams.append(t)
    reduzido = None
    if tams:
        reduzido = all(w == h for w, h in tams) and max(max(t) for t in tams) <= 1024
    return Dataset(
        caminho=no.caminho,
        classes={f.caminho.name: f.total for f in no.filhos},
        total=no.total,
        tamanho=tams[0] if tams else None,
        reduzido=reduzido,
    )


def dataset_em(caminho):
    """Varre um caminho digitado e devolve o Dataset que houver nele (ou None)."""
    p = Path(caminho)
    if not p.exists():
        return None
    no = varrer(p, 0, {"n": 0, "truncado": False})
    if _tem_subpastas_de_classe(no):
        return montar_dataset(no)
    memo, ds, _ = {}, [], []
    coletar(no, memo, ds, _)
    return montar_dataset(ds[0]) if len(ds) == 1 else None

# ----------------------------------------------------------------------------- ambiente
def localizar_scripts(scripts_dir, cwd):
    achados = {}
    for chave, (nome, _, _) in SCRIPTS.items():
        cands = [scripts_dir / nome, cwd / nome] + sorted(cwd.glob(f"*/{nome}"))
        achados[chave] = next((c for c in cands if c.is_file()), None)
    return achados


def versao_modulo(mod):
    try:
        if importlib.util.find_spec(mod) is None:
            return None
    except (ImportError, ValueError):
        return None
    for dist in PACOTES.get(mod, [mod]):
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    return "?"


def verificar_gpu(torch_ok):
    """Pergunta ao nvidia-smi e (num subprocesso) ao PyTorch, sem importar nada aqui."""
    info = {"placa": None, "torch_cuda": None, "cuda": None, "verificado": True}
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            r = subprocess.run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=15)
            if r.returncode == 0 and r.stdout.strip():
                info["placa"] = r.stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError):
            pass
    if torch_ok:
        try:
            r = subprocess.run(
                [sys.executable, "-c",
                 "import torch;print(int(torch.cuda.is_available()));print(torch.version.cuda)"],
                capture_output=True, text=True, timeout=180)
            linhas = r.stdout.split()
            if r.returncode == 0 and linhas:
                info["torch_cuda"] = linhas[0] == "1"
                info["cuda"] = linhas[1] if len(linhas) > 1 else None
        except (OSError, subprocess.SubprocessError):
            pass
    return info


def achar_treinos(cwd):
    vistos, res = set(), []
    for padrao in ("best.pt", "*/best.pt", "*/*/best.pt"):
        for b in cwd.glob(padrao):
            rel_partes = b.relative_to(cwd).parts
            if any(p.startswith(".") for p in rel_partes) or b.resolve() in vistos:
                continue
            vistos.add(b.resolve())
            d = b.parent
            t = Treino(pasta=d, best=b, mtime=b.stat().st_mtime, tem_last=(d / "last.pt").exists())
            sp = ler_json(d / "split.json")
            if isinstance(sp, dict):
                t.split_total = sum(len(v) for v in sp.values() if isinstance(v, list))
            t.metricas = ler_json(d / "metricas_teste.json")
            t.calib = ler_json(d / "calibracao.json")
            try:
                t.epocas = max(0, len((d / "history.csv").read_text(encoding="utf-8")
                                      .strip().splitlines()) - 1)
            except OSError:
                pass
            t.relatorios = sorted(p.name for p in d.glob("relatorio_*") if p.is_dir())
            t.explicacoes = sorted(p.name for p in d.glob("explicacao_*") if p.is_dir())
            res.append(t)
    return sorted(res, key=lambda t: t.mtime, reverse=True)


# ----------------------------------------------------------------------------- o menu
class Menu:
    def __init__(self, scripts_dir, pular_gpu):
        self.cwd = Path.cwd()
        self.scripts_dir = Path(scripts_dir)
        self.ctx = {}            # o que o fluxo guiado acabou de produzir (dataset, treino)
        self.gpu = {"placa": None, "torch_cuda": None, "cuda": None, "verificado": False}
        self.pular_gpu = pular_gpu
        self.estado = None

    # ---- varredura -------------------------------------------------------------------------
    def varrer_tudo(self, verificar_gpu_agora=False):
        scripts = localizar_scripts(self.scripts_dir, self.cwd)
        modulos = {m: versao_modulo(m) for m in PACOTES}
        if verificar_gpu_agora and not self.pular_gpu:
            print("Verificando PyTorch e GPU (pode levar alguns segundos)...")
            self.gpu = verificar_gpu(modulos.get("torch") is not None)

        orc = {"n": 0, "truncado": False}
        memo, nos_ds, soltas = {}, [], []
        vistos_ds = set()

        raizes = [self.cwd]
        for cand in (
            self.scripts_dir / "data",
            Path("/home/iarapp/Área de trabalho/Rede CNN V1/data"),
        ):
            if cand.is_dir() and cand.parent.resolve() not in {r.resolve() for r in raizes}:
                raizes.append(cand.parent)

        for r_path in raizes:
            raiz = varrer(r_path, 0, orc)
            ds_local, soltas_local = [], []
            coletar(raiz, memo, ds_local, soltas_local)
            for n in ds_local:
                res_p = n.caminho.resolve()
                if res_p not in vistos_ds:
                    vistos_ds.add(res_p)
                    nos_ds.append(n)
            for s in soltas_local:
                if s.caminho.resolve() not in vistos_ds:
                    soltas.append(s)

        datasets = [montar_dataset(n) for n in nos_ds]
        groups = None
        gj = ler_json(self.cwd / "groups.json")
        if isinstance(gj, dict) and "n" in gj:
            groups = {"caminho": self.cwd / "groups.json", "n": gj["n"], "root": gj.get("root")}
        self.estado = Estado(scripts, modulos, self.gpu, datasets, soltas,
                             achar_treinos(self.cwd), groups, orc["truncado"])
        return self.estado

    # ---- exibição --------------------------------------------------------------------------
    def rel(self, p):
        p = Path(p)
        try:
            return str(p.resolve().relative_to(self.cwd.resolve()))
        except ValueError:
            return str(p)

    def desc_dataset(self, d):
        if d.reduzido is True:
            tam = f"quadradas {d.tamanho[0]}x{d.tamanho[1]} (parecem redimensionadas)"
        elif d.reduzido is False and d.tamanho:
            tam = f"~{d.tamanho[0]}x{d.tamanho[1]} (parecem originais)"
        else:
            tam = "tamanho não verificado"
        return f"{self.rel(d.caminho)} — {len(d.classes)} classes, {fmt_n(d.total)} imagens, {tam}"

    def desc_treino(self, t):
        partes = []
        if t.epocas:
            partes.append(f"{t.epocas} épocas")
        if t.metricas:
            try:
                partes.append(f"teste: acc {t.metricas['acuracia']:.1%}, "
                              f"F1 {t.metricas['f1_macro']:.3f} (melhor época {t.metricas['epoca']})")
            except (KeyError, TypeError, ValueError):
                pass
        if t.calib:
            partes.append("calibrado")
        if t.relatorios:
            partes.append("relatório")
        if t.explicacoes:
            partes.append("explicabilidade")
        if t.tem_last and not t.metricas:
            partes.append("treino possivelmente interrompido (dá para retomar)")
        return f"{self.rel(t.pasta)} — " + (", ".join(partes) if partes else "só o best.pt")

    def mostrar_diagnostico(self):
        e = self.estado
        print(f"\nPasta de trabalho: {self.cwd}")
        print("\nScripts")
        for chave, (nome, desc, _) in SCRIPTS.items():
            p = e.scripts[chave]
            print(f"  {OK if p else NAO} {nome:<20} {desc}" + (f"  [{self.rel(p)}]" if p else "  [não encontrado]"))
        pastas = {p.parent for p in e.scripts.values() if p}
        if len(pastas) > 1:
            print("  AVISO: os scripts estão em pastas diferentes; o train_cassava.py e o "
                  "infer_cassava.py importam uns aos outros e precisam ficar na mesma pasta.")

        print("\nBibliotecas (para o python que está rodando o menu: " + sys.executable + ")")
        linha = []
        for m, v in e.modulos.items():
            if v is None:
                linha.append(f"{NAO} {m}" + (" (opcional)" if m in OPCIONAIS else ""))
            else:
                linha.append(f"{OK} {m} {v if v != '?' else ''}".strip())
        print("  " + " | ".join(linha))

        print("\nGPU")
        print("  " + self.texto_gpu())

        print("\nImagens rotuladas (uma subpasta por classe — o formato do treino)")
        if not e.datasets:
            print(f"  {NAO} nenhuma encontrada em {self.cwd}")
            data_cand = self.cwd / "data"
            if data_cand.exists():
                sub = [p.name for p in data_cand.iterdir() if p.is_dir()]
                print(f"      Pasta 'data' encontrada com subpastas: {sub or '[vazia]'}")
        for d in e.datasets:
            print(f"  {OK} {self.desc_dataset(d)}")
            cls = " | ".join(f"{k} {fmt_n(v)}" for k, v in sorted(d.classes.items()))
            print(f"      {cls}")
            nomes = {k.lower() for k in d.classes}
            if nomes == CLASSES_ARTIGO or nomes == CLASSES_CASSAVA_FULL:
                print("      5 classes válidas de mandioca reconhecidas (CBB, CBSD, CGM, CMD, Healthy)")
            else:
                print("      ATENÇÃO: nomes de classe diferentes do padrão esperado")
            if d.classes and max(d.classes.values()) > 20 * max(1, min(d.classes.values())):
                print("      desbalanceamento forte (o treino já compensa com pesos de classe)")
                
        print("\nPastas com imagens soltas (servem para classificar fotos novas)")
        if not e.soltas:
            print("  nenhuma encontrada (você também pode apontar para qualquer pasta ou arquivo)")
        for s in e.soltas:
            extra = " (e subpastas)" if s.filhos else ""
            print(f"  {OK} {self.rel(s.caminho)} — {fmt_n(s.total if s.filhos else s.diretas)} imagens{extra}")

        print("\nTreinos já feitos (best.pt)")
        if not e.treinos:
            print("  nenhum encontrado")
        for t in e.treinos:
            print(f"  {OK} {self.desc_treino(t)}")

        if e.groups:
            print(f"\nAgrupamento de duplicatas: groups.json cobre {fmt_n(e.groups['n'])} imagens")
        if e.truncado:
            print(f"\nAVISO: a busca por imagens parou após {fmt_n(MAX_ENTRADAS)} entradas; "
                  "rode o menu dentro da pasta do projeto.")
        print(f"\n{SETA} Próximo passo sugerido: {self.sugerir()}")

    def texto_gpu(self):
        g, e = self.gpu, self.estado
        if not g.get("verificado"):
            return "não verificada (a opção --pular-gpu foi usada)"
        if e.modulos.get("torch") is None:
            base = "PyTorch não instalado"
            return base + (f"; placa NVIDIA detectada: {g['placa']}" if g["placa"] else
                           "; nvidia-smi não encontrado")
        if g["torch_cuda"]:
            return f"{OK} PyTorch enxerga a GPU (CUDA {g['cuda']}) — {g['placa'] or 'placa NVIDIA'}"
        if g["placa"]:
            return (f"{NAO} há placa ({g['placa']}), mas o PyTorch instalado NÃO a enxerga: "
                    "provavelmente é a versão só de CPU — reinstale com suporte a CUDA "
                    "(https://pytorch.org/get-started/locally/)")
        return f"{NAO} sem GPU utilizável (nvidia-smi não encontrado): o treino rodaria na CPU, muito lento"

    def resumo(self):
        e = self.estado
        n_scripts = sum(1 for p in e.scripts.values() if p)
        faltam = [m for m, v in e.modulos.items() if v is None and m not in OPCIONAIS]
        print(f"\n{'=' * 64}\n  Cassava CNN — menu   |   pasta: {self.cwd}\n{'=' * 64}")
        print(f"  Scripts: {n_scripts}/{len(SCRIPTS)} {OK if n_scripts == len(SCRIPTS) else NAO}"
              f"   Bibliotecas: {OK + ' ok' if not faltam else NAO + ' faltam ' + ', '.join(faltam)}")
        print("  GPU: " + self.texto_gpu().split(" — ")[0].split(":")[0][:90])
        if e.datasets:
            for d in e.datasets[:4]:
                print(f"  Imagens: {self.desc_dataset(d)}")
        else:
            print("  Imagens: nenhuma pasta rotulada encontrada")
        print(f"  Treinos: {len(e.treinos)}" + (f" (mais recente: {self.rel(e.treinos[0].pasta)})"
                                                if e.treinos else ""))
        print(f"  {SETA} Sugestão: {self.sugerir()}")

    def sugerir(self):
        e = self.estado
        ausentes = [SCRIPTS[k][0] for k, p in e.scripts.items() if p is None]
        if ausentes:
            return f"coloque {', '.join(ausentes)} na pasta do menu."
        if not e.datasets:
            return ("nenhuma pasta de imagens rotuladas. Coloque o dataset numa pasta (ex.: data/) "
                    "com uma subpasta por classe (CBB, CBSD, CGM, CMD, Healthy).")
        faltam = [m for m in ("torch", "torchvision", "sklearn") if e.modulos.get(m) is None]
        if faltam:
            return f"instale {', '.join(faltam)} para treinar (veja a opção 1)."
        if not any(d.reduzido for d in e.datasets) and any(d.reduzido is False for d in e.datasets):
            return "opção 2: redimensionar o dataset (treinar nas fotos originais é lento)."
        if not e.treinos:
            return "opção 4: treinar a rede (faça primeiro o teste rápido de 2 épocas)."
        t = e.treinos[0]
        if not t.metricas:
            return f"o treino {self.rel(t.pasta)} não terminou: opção 4 e escolha retomar."
        if not t.relatorios:
            return f"opção 5: gerar o relatório de {self.rel(t.pasta)} (calibra o limiar de 95%)."
        if not t.explicacoes:
            return f"opção 7: explicabilidade de {self.rel(t.pasta)} (Grad-CAM e fidelidade)."
        return "opção 6: classificar fotos novas, ou treinar outra arquitetura para um ensemble."

    # ---- execução --------------------------------------------------------------------------
    def exigir(self, chave, extras=()):
        nome, _, mods = SCRIPTS[chave]
        p = self.estado.scripts[chave]
        if p is None:
            print(f"\n{NAO} {nome} não foi encontrado (procurei em {self.scripts_dir} e em {self.cwd}).")
            return None
        faltam = [m for m in list(mods) + list(extras) if self.estado.modulos.get(m) is None]
        if faltam:
            print(f"\n{NAO} {nome} precisa de: {', '.join(faltam)}, que não estão instalados "
                  f"para este python ({sys.executable}).")
            pip = [PIP_NOME.get(m, m) for m in faltam if m not in ("torch", "torchvision")]
            if pip:
                print(f"   pip install {' '.join(pip)}")
            if any(m in ("torch", "torchvision") for m in faltam):
                print("   PyTorch/torchvision: use o comando de https://pytorch.org/get-started/locally/ "
                      "(escolha a sua versão de CUDA para usar a GPU).")
            return None
        return p

    def comando(self, script, *args):
        return [sys.executable, str(script), *map(str, args)]

    def exibir(self, cmd):
        return " ".join(shlex.quote(x) for x in ["python3", self.rel(cmd[1])] + cmd[2:])

    def rodar(self, cmd):
        print(f"\nComando:\n  {self.exibir(cmd)}\n")
        if not sim_nao("Executar agora?", True):
            print("Cancelado.")
            return False
        ini = time.time()
        try:
            codigo = subprocess.run(cmd, cwd=self.cwd).returncode
        except KeyboardInterrupt:
            codigo = 130
            print("\nInterrompido pelo usuário (Ctrl+C).")
        except OSError as ex:
            codigo = 127
            print(f"Não consegui executar: {ex}")
        dur = time.time() - ini
        try:
            with open(self.cwd / HISTORICO, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | saída {codigo} | {dur / 60:.1f} min | "
                        f"{self.exibir(cmd)}\n")
        except OSError:
            pass
        print(f"\n{OK + ' Concluído' if codigo == 0 else NAO + ' Terminou com erro (código ' + str(codigo) + ')'}"
              f" em {dur / 60:.1f} min.")
        self.varrer_tudo()
        return codigo == 0

    # ---- escolhas reutilizáveis --------------------------------------------------------------
    def escolher_dataset(self, titulo, preferir=None, padrao_caminho=None):
        ds = self.estado.datasets
        padrao = 1
        alvo = padrao_caminho or self.ctx.get("data")
        for i, d in enumerate(ds, 1):
            if alvo and Path(alvo).resolve() == d.caminho.resolve():
                padrao = i
                break
        else:
            for i, d in enumerate(ds, 1):
                if (preferir == "reduzido" and d.reduzido) or (preferir == "original" and d.reduzido is False):
                    padrao = i
                    break
        opcoes = [self.desc_dataset(d) for d in ds] + ["Digitar outro caminho"]
        i = escolher(titulo, opcoes, padrao=padrao if ds else len(opcoes))
        if i is None:
            return None
        if i < len(ds):
            return ds[i]
        while True:
            c = perguntar_caminho("Caminho da pasta (uma subpasta por classe; vazio cancela)")
            if not c:
                return None
            d = dataset_em(c)
            if d:
                return d
            print("Não encontrei, nesse caminho, uma pasta com >= 2 subpastas de classe contendo imagens.")

    def dataset_do_treino(self, t):
        if t.split_total is None:
            return None
        for d in self.estado.datasets:
            if d.total == t.split_total:
                return d
        return None

    def escolher_treinos(self, titulo, multi=True):
        ts = self.estado.treinos
        if not ts:
            print(f"\n{NAO} Nenhum treino (best.pt) encontrado. Treine primeiro (opção 4).")
            return None
        padrao = 1
        if self.ctx.get("run"):
            for i, t in enumerate(ts, 1):
                if t.pasta.resolve() == Path(self.ctx["run"]).resolve():
                    padrao = i
        r = escolher(titulo, [self.desc_treino(t) for t in ts], padrao=padrao, multi=multi)
        if r is None:
            return None
        return [ts[i] for i in r] if multi else [ts[r]]

    # ---- ações -----------------------------------------------------------------------------
    def acao_resize(self):
        script = self.exigir("resize")
        if not script:
            return False
        d = self.escolher_dataset("Qual pasta de imagens redimensionar?", preferir="original")
        if not d:
            return False
        if d.reduzido:
            print(f"\nObs.: {self.rel(d.caminho)} já parece redimensionada ({d.tamanho[0]}px).")
        lado = perguntar_numero("Lado final em pixels (384 = recomendado; 224 só p/ testes rápidos)",
                                384, minimo=32)
        dst = perguntar_caminho("Pasta de destino",
                                self.rel(d.caminho.with_name(f"{d.caminho.name}_{lado}")))
        if Path(dst).resolve() == d.caminho.resolve():
            print(f"{NAO} O destino não pode ser a própria pasta de origem.")
            return False
        if Path(dst).exists() and any(Path(dst).rglob("*.jpg")):
            print("Obs.: o destino já tem imagens; as que já existem são puladas (retoma de onde parou).")
        ok = self.rodar(self.comando(script, "--src", self.rel(d.caminho), "--dst", dst,
                                     "--size", lado))
        if ok:
            self.ctx["data"] = Path(dst)
        return ok

    def acao_dedupe(self):
        script = self.exigir("dedupe")
        if not script:
            return False
        d = self.escolher_dataset("Em qual pasta procurar duplicatas?", preferir="reduzido")
        if not d:
            return False
        if self.estado.groups and self.estado.groups["n"] == d.total:
            print(f"\nJá existe groups.json cobrindo {fmt_n(d.total)} imagens.")
            if not sim_nao("Refazer mesmo assim?", False):
                return False
        print("\nAviso: compara todas as fotos entre si (8 orientações); com ~21 mil imagens pode levar minutos.")
        corr = perguntar_numero("Correlação mínima para considerar duplicata (0,97 = conservador)",
                                0.97, tipo=float, minimo=0.5, maximo=1.0)
        args = ["--data", self.rel(d.caminho), "--out", "groups.json", "--report", "duplicados.csv"]
        if corr != 0.97:
            args += ["--min-corr", corr]
        return self.rodar(self.comando(script, *args))

    def acao_treinar(self):
        script = self.exigir("treino")
        if not script:
            return False
        args_extra = []
        if self.estado.scripts["dedupe"] is None or self.estado.modulos.get("cv2") is None:
            print(f"\n{NAO} O agrupamento de duplicatas (dedupe_dataset.py + OpenCV) não está disponível.")
            if not sim_nao("Treinar sem agrupar duplicatas (--no-groups, não recomendado)?", False):
                return False
            args_extra.append("--no-groups")
        g = self.estado.scripts
        gpu = self.gpu
        if gpu.get("verificado") and gpu.get("torch_cuda") is False:
            print(f"\n{NAO} {self.texto_gpu()}")
            if not sim_nao("Treinar mesmo assim, na CPU (muito lento)?", False):
                return False

        d = self.escolher_dataset("Qual pasta de imagens usar no treino?", preferir="reduzido")
        if not d:
            return False
        if d.reduzido is False:
            print(f"\nAtenção: {self.rel(d.caminho)} parece ter as fotos originais; o treino fica bem "
                  "mais lento e a GPU espera pelo disco. Recomendo a opção 2 (redimensionar) antes.")
            if not sim_nao("Continuar mesmo assim?", False):
                return False

        modo = escolher("Modo de treino", [
            "Completo (30 épocas — para o resultado do artigo)",
            "Teste rápido (2 épocas a 224 px — só para ver se tudo funciona)"], padrao=1, voltar=True)
        if modo is None:
            return False
        rapido = modo == 1

        nomes_ok = []
        try:
            fonte = Path(g["treino"]).read_text(encoding="utf-8")
        except OSError:
            fonte = ""
        for nome, com in ARQUITETURAS:
            if not fonte or f'"{nome}"' in fonte:
                nomes_ok.append((nome, com))
        i = escolher("Arquitetura", [f"{n} — {c}" for n, c in nomes_ok], padrao=1)
        if i is None:
            return False
        arch = nomes_ok[i][0]

        if rapido:
            img = 224
        else:
            sug = d.tamanho[0] if d.reduzido and d.tamanho else 384
            img = perguntar_numero("Tamanho da imagem de entrada em pixels", sug, minimo=64)
        out = perguntar_caminho("Pasta de saída do treino", f"runs/{arch}{'_teste' if rapido else ''}")

        args = ["--data", self.rel(d.caminho), "--arch", arch, "--img", img, "--out", out]
        if rapido:
            args += ["--epochs", 2]

        saida = Path(out)
        if (saida / "last.pt").exists() or (saida / "best.pt").exists():
            r = escolher(f"Já existe um treino em {out}", [
                "Retomar de onde parou (--resume; use os mesmos parâmetros do treino original)",
                "Usar outro nome de pasta",
                "Sobrescrever (o best.pt anterior será substituído quando a validação melhorar)"],
                padrao=1)
            if r is None:
                return False
            if r == 0:
                args.append("--resume")
            elif r == 1:
                out = perguntar_caminho("Novo nome da pasta de saída", f"{out}_2")
                args[args.index("--out") + 1] = out

        if "--resume" not in args and not rapido and sim_nao("\nAjustar opções avançadas (épocas, lote, taxa...)?", False):
            ep = perguntar_numero("Épocas", 30, minimo=1)
            bs = perguntar_numero("Tamanho do lote (se faltar memória da GPU: 16)", 32, minimo=1)
            lr = perguntar_numero("Taxa de aprendizado", 0.0003, tipo=float, minimo=1e-7, maximo=1.0)
            pat = perguntar_numero("Paciência do early stopping (0 = desligado)", 0, minimo=0)
            cm = perguntar_numero("Probabilidade de CutMix por lote (0 = desliga)", 0.5, tipo=float,
                                  minimo=0.0, maximo=1.0)
            peso = ["inverse", "sqrt", "none"][escolher(
                "Pesos de classe na perda", ["inverse (padrão)", "sqrt (mais suave)", "none"],
                padrao=1, voltar=False)]
            wk = perguntar_numero("Processos de leitura de imagens (workers)", 4, minimo=0)
            dev = perguntar("Dispositivo (auto, cuda, cuda:1 ou cpu)", "auto")
            for flag, v, padrao in (("--epochs", ep, 30), ("--bs", bs, 32), ("--lr", lr, 0.0003),
                                    ("--patience", pat, 0), ("--cutmix", cm, 0.5),
                                    ("--weights", peso, "inverse"), ("--workers", wk, 4),
                                    ("--device", dev, "auto")):
                if v != padrao:
                    args += [flag, v]

        if "--no-groups" not in args_extra:
            gp = self.estado.groups
            if gp and gp["n"] == d.total:
                print(f"\nVou reutilizar o groups.json (cobre as {fmt_n(d.total)} imagens), "
                      "assim todos os modelos usam o mesmo split.")
                args += ["--groups", "groups.json"]
            else:
                print("\nO treino vai agrupar as duplicatas sozinho (pode levar alguns minutos na primeira vez).")
        args += args_extra

        ok = self.rodar(self.comando(script, *args))
        if ok:
            self.ctx["run"] = Path(out)
            self.ctx["data"] = d.caminho
        return ok

    def _opcoes_inferencia(self, ts):
        """Pergunta o que é comum ao relatório e à explicabilidade: o dataset do treino."""
        casado = self.dataset_do_treino(ts[0])
        if casado:
            print(f"\nA pasta de imagens deste treino parece ser {self.rel(casado.caminho)} "
                  f"({fmt_n(casado.total)} imagens, igual ao split.json).")
        elif ts[0].split_total:
            print(f"\nAtenção: o split.json do treino cobre {fmt_n(ts[0].split_total)} imagens e nenhuma "
                  "pasta encontrada tem esse total. Use exatamente a mesma pasta do treino.")
        return self.escolher_dataset("Qual pasta de imagens usar?",
                                     padrao_caminho=casado.caminho if casado else None)

    def acao_report(self):
        script = self.exigir("infer")
        if not script:
            return False
        ts = self.escolher_treinos("Qual treino avaliar? (vários = ensemble)", multi=True)
        if not ts:
            return False
        d = self._opcoes_inferencia(ts)
        if not d:
            return False
        split = ["test", "val", "train", "all"][escolher(
            "Conjunto a avaliar", ["test — dados nunca vistos (use este para o artigo)",
                                   "val — validação", "train — treino", "all — tudo"],
            padrao=1, voltar=False)]
        args = ["report", "--ckpt", *[self.rel(t.best) for t in ts], "--data", self.rel(d.caminho),
                "--split", split]
        calib_existente = ts[0].calib is not None
        if split == "test":
            if sim_nao("Calibrar a confiança e escolher o limiar de 95% na validação?"
                       + (" (já existe calibração; refaz)" if calib_existente else ""), True):
                args.append("--calibrate")
                meta = perguntar_numero("Meta de acurácia nas previsões aceitas", 0.95, tipo=float,
                                        minimo=0.5, maximo=0.999)
                if meta != 0.95:
                    args += ["--target-acc", meta]
        if sim_nao("Usar TTA (média sobre a imagem e 3 espelhamentos; um pouco mais lento)?", True):
            args.append("--tta")
        if sim_nao("Gerar Grad-CAM dos erros mais confiantes?", False):
            args.append("--gradcam")
        if sim_nao("Copiar as imagens para pastas acertos/ e erros/?", False):
            args.append("--copy-images")
        ok = self.rodar(self.comando(script, *args))
        if ok:
            self.ctx["run"] = ts[0].pasta
        return ok

    def acao_predict(self):
        script = self.exigir("infer")
        if not script:
            return False
        ts = self.escolher_treinos("Qual treino usar para classificar? (vários = ensemble)", multi=True)
        if not ts:
            return False
        ids = [f"{t.pasta.name}/best.pt" for t in ts]
        if ts[0].calib is None:
            print(f"\nAtenção: este treino ainda não foi calibrado (opção 5 com calibração). "
                  "Sem isso o limiar de confiança padrão é 60%, e não o de 95%.")
        elif ts[0].calib.get("ckpts") and ts[0].calib["ckpts"] != ids:
            print(f"\nAtenção: a calibração existente é de {ts[0].calib['ckpts']}, não desta combinação. "
                  "Rode a opção 5 com calibração para estes modelos.")
        else:
            lim = ts[0].calib.get("limiar")
            print(f"\nCalibração encontrada (temperatura {ts[0].calib.get('T', 1):.2f}"
                  + (f", limiar de confiança {lim:.1%}" if lim else "") + ").")

        soltas = self.estado.soltas
        opcoes = [f"{self.rel(s.caminho)} — {fmt_n(s.total if s.filhos else s.diretas)} imagens"
                  for s in soltas] + ["Digitar outro caminho (imagem ou pasta)"]
        i = escolher("Quais imagens classificar?", opcoes, padrao=1 if soltas else len(opcoes))
        if i is None:
            return False
        if i < len(soltas):
            entrada = self.rel(soltas[i].caminho)
        else:
            while True:
                entrada = perguntar_caminho("Caminho da imagem ou da pasta (vazio cancela)")
                if not entrada:
                    return False
                p = Path(entrada)
                if p.is_file() and p.suffix.lower() in EXTS:
                    break
                if p.is_dir() and varrer(p, 0, {"n": 0, "truncado": False}).total:
                    break
                print("Não achei imagens (.jpg/.jpeg/.png) nesse caminho.")
        args = ["predict", "--ckpt", *[self.rel(t.best) for t in ts], "--input", entrada]
        out = perguntar_caminho("Pasta de saída (CSV e mapas)", "predicoes")
        if out != "predicoes":
            args += ["--out", out]
        if sim_nao("Usar TTA?", bool(ts[0].calib and ts[0].calib.get("tta"))):
            args.append("--tta")
        if sim_nao("Salvar o Grad-CAM de cada imagem?", False):
            args.append("--gradcam")
        return self.rodar(self.comando(script, *args))

    def acao_explain(self):
        script = self.exigir("infer")
        if not script:
            return False
        ts = self.escolher_treinos("Qual treino explicar? (um só)", multi=False)
        if not ts:
            return False
        d = self._opcoes_inferencia(ts)
        if not d:
            return False
        faith = perguntar_numero("Imagens no teste de fidelidade (200 = bom; 50 = rápido)", 200, minimo=5)
        args = ["explain", "--ckpt", self.rel(ts[0].best), "--data", self.rel(d.caminho), "--split", "test"]
        if faith != 200:
            args += ["--faith-n", faith]
        ok = self.rodar(self.comando(script, *args))
        if ok:
            self.ctx["run"] = ts[0].pasta
        return ok

    def fluxo_completo(self):
        print("\nFluxo guiado: redimensionar → treinar → relatório → explicabilidade.\n"
              "Cada passo mostra o comando e pede confirmação; dá para pular qualquer um.")
        tem_reduzido = any(d.reduzido for d in self.estado.datasets)
        passos = [("1/4 Redimensionar o dataset", self.acao_resize, tem_reduzido),
                  ("2/4 Treinar a rede", self.acao_treinar, False),
                  ("3/4 Relatório (acertos, erros, calibração)", self.acao_report, False),
                  ("4/4 Explicabilidade (Grad-CAM e fidelidade)", self.acao_explain, False)]
        for titulo, fn, ja_feito in passos:
            print(f"\n--- {titulo} ---")
            if ja_feito:
                if not sim_nao("Já existe um dataset redimensionado. Redimensionar de novo?", False):
                    continue
            elif not sim_nao("Fazer este passo agora?", True):
                continue
            if not fn() and not sim_nao("Esse passo não terminou. Continuar para o próximo?", False):
                break
        print("\nFim do fluxo guiado.")

    def mostrar_historico(self):
        p = self.cwd / HISTORICO
        if not p.exists():
            print("\nAinda não há comandos registrados.")
            return
        linhas = p.read_text(encoding="utf-8").splitlines()
        print(f"\nÚltimos comandos ({HISTORICO}):")
        for ln in linhas[-15:]:
            print("  " + ln)

    def ajuda(self):
        print("""
Ordem usual do trabalho:
  2) Redimensionar   reduz o dataset uma vez (384 px) para o treino não esperar pelo disco.
  3) Duplicatas      opcional aqui: o treino também agrupa sozinho; rode se quiser ver o
                     duplicados.csv ou reutilizar o mesmo split em vários modelos.
  4) Treinar         comece pelo teste rápido (2 épocas) para validar o ambiente.
  5) Relatório       avalia no conjunto de teste, calibra a confiança e acha o limiar para
                     a meta de 95% nas previsões aceitas. Gera acertos, erros e métricas.
  7) Explicabilidade Grad-CAM por classe + teste de fidelidade dos mapas (para o artigo).
  6) Classificar     usa o modelo treinado e calibrado em fotos novas, uma a uma.
  8) Fluxo completo  conduz 2 → 4 → 5 → 7 em sequência.

Dicas: Ctrl+C cancela uma pergunta ou interrompe um comando; um treino interrompido pode ser
retomado na opção 4. Todos os comandos ficam em menu_historico.txt.""")

    # ---- laço principal -------------------------------------------------------------------
    def loop(self):
        acoes = {"1": self.mostrar_diagnostico, "2": self.acao_resize, "3": self.acao_dedupe,
                 "4": self.acao_treinar, "5": self.acao_report, "6": self.acao_predict,
                 "7": self.acao_explain, "8": self.fluxo_completo, "9": self.mostrar_historico,
                 "h": self.ajuda, "?": self.ajuda}
        while True:
            self.resumo()
            print("""
  1) Diagnóstico completo         5) Relatório (acertos/erros/calibração)
  2) Redimensionar dataset        6) Classificar imagens novas
  3) Procurar duplicatas          7) Explicabilidade (Grad-CAM + fidelidade)
  4) Treinar a rede               8) Fluxo completo guiado
  9) Histórico de comandos        h) Ajuda        0) Sair""")
            r = input("Escolha: ").strip().lower()
            if r in ("0", "q", "sair"):
                print("Até mais!")
                return
            fn = acoes.get(r)
            if not fn:
                print("Opção inválida.")
                continue
            try:
                fn()
            except KeyboardInterrupt:
                print("\nCancelado.")
            if r in ("1", "9", "h", "?"):
                input("\n(Enter para voltar ao menu) ")


def main():
    ap = argparse.ArgumentParser(description="Menu do pipeline de doenças da mandioca")
    ap.add_argument("--root", default=None, help="pasta de trabalho (padrão: a pasta atual)")
    ap.add_argument("--scripts-dir", default=None,
                    help="pasta dos scripts (padrão: a pasta deste arquivo)")
    ap.add_argument("--diagnostico", action="store_true", help="mostra o que foi encontrado e sai")
    ap.add_argument("--pular-gpu", action="store_true",
                    help="não testa o PyTorch/GPU na partida (a partida fica mais rápida)")
    a = ap.parse_args()

    scripts_dir = Path(a.scripts_dir).resolve() if a.scripts_dir else Path(__file__).resolve().parent
    if a.root:
        os.chdir(a.root)
    m = Menu(scripts_dir, a.pular_gpu)
    m.varrer_tudo(verificar_gpu_agora=True)
    if a.diagnostico:
        m.mostrar_diagnostico()
        return
    try:
        m.loop()
    except (EOFError, KeyboardInterrupt):
        print("\nSaindo.")


if __name__ == "__main__":
    main()
