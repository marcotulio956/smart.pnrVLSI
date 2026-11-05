# -*- coding: utf-8 -*-
"""
cts_heuristicas.py
==================

Heurísticas propostas (H1, H2, H3) para **Clock Tree Synthesis (CTS)** com
**Deferred-Merge Embedding (DME)** como sub-rotina, mais *assinaturas* das
heurísticas de controle (C1, C2, C3), que devem ser implementadas depois.

Este módulo acompanha o pré-projeto: dados de entrada (fonte r, sinks S,
posições, limite de skew δ), decisão (topologia binária T), função objetivo
(minimizar o comprimento de fio WL*(T) = min_{E,W} WL(T,E,W) s.a. skew ≤ δ).
O estilo (dataclass de histórico, runner de comparação, Metropolis, gráficos
importando matplotlib dentro das funções) segue `sa_vlsi_placement.py` e
`aco_vlsi_routing.py`, mas o módulo é **autônomo** (não importa nenhum dos dois):
aqui o espaço é contínuo (plano retilíneo) e não uma grade de sítios/G-cells.

Modelo
------
- Plano retilíneo (Manhattan). Atraso linear: t(r, s) = soma dos comprimentos
  das arestas do caminho r → s. O atraso interno dos sinks é 0; a carga (capacitância)
  dos sinks só entra no custo de fusão (balanceamento), não no atraso.
- Coordenadas rotacionadas u = x + y, v = x − y: a distância de Manhattan vira
  distância L∞ e "regiões de fusão" (TRR) viram **retângulos** alinhados aos eixos.

DME-Embed (sub-rotina)
----------------------
Fase ascendente: cada nó guarda (região de fusão, intervalo de atraso [lo, hi],
comprimento de fio da subárvore). Ao fundir A e B escolhe-se (e_a, e_b) com
e_a + e_b mínimo tal que (i) as regiões se tocam (e_a + e_b ≥ d(A,B)) e
(ii) o intervalo de atraso resultante tem largura ≤ δ; se necessário há
*snaking* (e_a + e_b > d). A região do pai é TRR(A, e_a) ∩ TRR(B, e_b).
Fase descendente: a raiz r é ligada ao ponto mais próximo da região do topo, e
cada filho é posicionado no ponto da sua região mais próximo do pai.

Para δ = 0 é o DME clássico de skew nulo (Boese/Kahng/Edahiro). Para δ > 0 é uma
versão **simplificada** do BST/DME de Cong et al.: as regiões de fusão continuam
retangulares (em u,v) e o folga de skew é usada para evitar *snaking* e para
escolher a divisão (e_a, e_b); não são exploradas as regiões poligonais do
BST/DME completo. O skew realizado é exato (as larguras de intervalo são
somas de comprimentos de aresta fixos) e sempre ≤ δ.

Avaliação incremental: WL e skew dependem só da fase ascendente. As buscas locais
trocam subárvores e recalculam **apenas os ancestrais** dos nós afetados
(O(profundidade) por movimento, em vez de O(n)), com desfazer barato.

Heurísticas de controle
-----------------------
`heuristica_c1`, `heuristica_c2`, `heuristica_c3` são apenas **assinaturas**
(levantam NotImplementedError). O contrato de retorno está documentado em
cada uma; o runner de experimentos as detecta e as ignora enquanto não
estiverem implementadas.

ISPD 2009/2010: não há leitor de benchmark aqui. Para usá-lo, basta construir
`InstanciaCTS` (xy, cargas, raiz) a partir dos arquivos.
"""

from __future__ import annotations

import heapq
import math
import os
import pickle
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

TOL = 1e-9

HEURISTICAS_PROPOSTAS = ("H1", "H2", "H3")
CONTROLES = ("C1", "C2", "C3")
ESTOCASTICAS = ("H1", "H2", "H3")

PARAMS_PADRAO: Dict[str, object] = dict(
    # construção semi-gulosa (H1)
    k_rcl=3,              # tamanho da RCL (k pares de menor custo)
    alpha=0.3,            # peso de balanceamento (em "espaçamentos médios entre sinks" por sink de diferença)
    k_viz=6,              # vizinhos candidatos por cluster (aceleração espacial)
    # busca local (H2, H3)
    vizinhos_busca=6,     # vizinhos geométricos (centróides) por nó
    raio_arvore=3,        # vizinhança por "distância de níveis" (arestas na árvore)
    incluir_folhas=False, # False: só nós internos (como no pré-projeto)
    # H3
    theta0_rel=1e-3,      # θ0 = theta0_rel * WL inicial
    theta_final_rel=1e-2, # θ_final = theta_final_rel * θ0 (resfriamento por tempo)
    taxa_resfriamento=None,  # None: resfria geometricamente em função do tempo; float: θ ← ρ·θ por iteração
    frac_fragmento=(0.02, 0.10),
    max_fusoes=200,
    busca_focada=True,
    max_foco=64,
    max_passadas_foco=2,
)


# --------------------------------------------------------------------------- #
# 1. Instâncias
# --------------------------------------------------------------------------- #

@dataclass
class InstanciaCTS:
    """Instância de CTS: sinks (posição + carga) e fonte de clock."""
    nome: str
    xy: np.ndarray                 # (n, 2), em µm
    cargas: np.ndarray             # (n,)
    raiz: Tuple[float, float]
    distribuicao: str = "uniforme"
    semente: Optional[int] = None

    @property
    def n(self) -> int:
        return int(len(self.xy))

    def escala_distancia(self) -> float:
        """Espaçamento típico entre sinks, sqrt(área_bbox / n)."""
        w = float(np.ptp(self.xy[:, 0]))
        h = float(np.ptp(self.xy[:, 1]))
        return max(math.sqrt(w * h / self.n), max(w, h) / self.n, 1e-9)


def gerar_instancia_sintetica(
    n: int,
    distribuicao: str = "uniforme",
    semente: int = 0,
    lado: float = 10_000.0,
    n_aglomerados: int = 8,
    desvio_rel: float = 0.03,
    concentracao: float = 0.7,
    nome: Optional[str] = None,
) -> InstanciaCTS:
    """
    Instância sintética com `n` sinks em um chip quadrado de `lado` µm.

    - "uniforme": posições uniformes na área.
    - "clusterizada": mistura de `n_aglomerados` gaussianas (desvio = desvio_rel*lado)
      com pesos assimétricos (Dirichlet(concentracao)), imitando blocos de lógica.
    A fonte de clock fica no centro do chip; as cargas são U(0,5; 1,5).
    """
    rng = np.random.default_rng(semente)
    if distribuicao == "uniforme":
        xy = rng.uniform(0.0, lado, size=(n, 2))
    elif distribuicao == "clusterizada":
        centros = rng.uniform(0.1 * lado, 0.9 * lado, size=(n_aglomerados, 2))
        pesos = rng.dirichlet(np.full(n_aglomerados, concentracao))
        comp = rng.choice(n_aglomerados, size=n, p=pesos)
        xy = centros[comp] + rng.normal(0.0, desvio_rel * lado, size=(n, 2))
        xy = np.clip(xy, 0.0, lado)
    else:
        raise ValueError("distribuicao deve ser 'uniforme' ou 'clusterizada'")
    cargas = rng.uniform(0.5, 1.5, size=n)
    return InstanciaCTS(
        nome=nome or f"{distribuicao[:3]}_{n}_{semente}",
        xy=xy, cargas=cargas, raiz=(lado / 2.0, lado / 2.0),
        distribuicao=distribuicao, semente=semente,
    )


def gerar_conjunto_sintetico(
    tamanhos: Sequence[int] = (100, 500, 1000, 5000, 10000),
    distribuicoes: Sequence[str] = ("uniforme", "clusterizada"),
    n_instancias: int = 10,
    semente_base: int = 2024,
    lado: float = 10_000.0,
) -> List[InstanciaCTS]:
    """`n_instancias` instâncias independentes por combinação (n, distribuição)."""
    inst = []
    contador = 0
    for n in tamanhos:
        for d in distribuicoes:
            for k in range(n_instancias):
                s = semente_base + contador
                contador += 1
                inst.append(gerar_instancia_sintetica(
                    n, d, s, lado=lado, nome=f"{d[:3]}_{n}_{k}"))
    return inst


# --------------------------------------------------------------------------- #
# 2. Árvore + DME-Embed (ascendente incremental, descendente)
# --------------------------------------------------------------------------- #
# Estado por nó (tupla imutável, o que torna desfazer movimentos trivial):
#   0 reg=(u0,u1,v0,v1)  1 lo  2 hi  3 wl_sub  4 e_esq  5 e_dir
#   6 cnt  7 soma_x  8 soma_y  9 carga

class ArvoreCTS:
    """
    Topologia binária sobre os sinks, com o estado do DME-Embed por nó.

    Nós 0..n-1 são os sinks; nós n..2n-2 são pontos de Steiner. `ordem` é a
    *sequência de fusões* (ordem topológica dos nós internos), usada por H3.
    """

    def __init__(self, inst: InstanciaCTS, delta: float):
        self.inst = inst
        self.n = inst.n
        self.delta = float(delta)
        N = 2 * self.n - 1
        self.esq = [-1] * N
        self.dir = [-1] * N
        self.pai = [-1] * N
        self.est: List[Optional[tuple]] = [None] * N
        self.top = -1
        self.ordem: List[int] = []
        rx, ry = float(inst.raiz[0]), float(inst.raiz[1])
        self._ur, self._vr = rx + ry, rx - ry
        for i, ((x, y), c) in enumerate(zip(inst.xy.tolist(), inst.cargas.tolist())):
            u, v = x + y, x - y
            self.est[i] = ((u, u, v, v), 0.0, 0.0, 0.0, 0.0, 0.0, 1, x, y, c)

    # -- construção ------------------------------------------------------ #
    @classmethod
    def de_fusoes(cls, inst: InstanciaCTS, delta: float, fusoes: Sequence[Tuple[int, int]]) -> "ArvoreCTS":
        """Constrói a árvore a partir da sequência de fusões. A fusão j cria o
        nó n+j a partir dos nós (a, b) (sinks < n ou fusões anteriores)."""
        arv = cls(inst, delta)
        n = inst.n
        for j, (a, b) in enumerate(fusoes):
            v = n + j
            arv.esq[v], arv.dir[v] = a, b
            arv.pai[a] = v
            arv.pai[b] = v
            arv.ordem.append(v)
            arv.est[v] = arv._fundir(arv.est[a], arv.est[b])
        arv.top = n + len(fusoes) - 1
        return arv

    def copiar(self) -> "ArvoreCTS":
        novo = ArvoreCTS.__new__(ArvoreCTS)
        novo.inst, novo.n, novo.delta = self.inst, self.n, self.delta
        novo.esq, novo.dir, novo.pai = self.esq[:], self.dir[:], self.pai[:]
        novo.est = self.est[:]
        novo.top, novo.ordem = self.top, self.ordem[:]
        novo._ur, novo._vr = self._ur, self._vr
        return novo

    # -- DME: fase ascendente -------------------------------------------- #
    def _fundir(self, A: tuple, B: tuple) -> tuple:
        (au0, au1, av0, av1), loa, hia, wla, _, _, ca, sxa, sya, qa = A
        (bu0, bu1, bv0, bv1), lob, hib, wlb, _, _, cb, sxb, syb, qb = B
        delta = self.delta

        gu = max(0.0, bu0 - au1, au0 - bu1)
        gv = max(0.0, bv0 - av1, av0 - bv1)
        d = gu if gu > gv else gv                      # distância L∞ entre regiões

        # x = e_a - e_b deve satisfazer  Lw <= x <= U  (skew do pai <= delta)
        Lw = hib - loa - delta
        U = delta + lob - hia
        if Lw > U:                                      # ruído numérico
            Lw = U = 0.5 * (Lw + U)
        x0 = 0.5 * ((hib + lob) - (hia + loa))          # equilibra os pontos médios
        lx = max(Lw, -d)
        hx = min(U, d)
        if lx <= hx:                                    # sem snaking
            x = min(max(x0, lx), hx)
            L = d
        elif U < -d:                                    # snaking no lado de B
            x = U
            L = -U
        else:                                           # snaking no lado de A (Lw > d)
            x = Lw
            L = Lw
        ea = max(0.0, 0.5 * (L + x))
        eb = max(0.0, 0.5 * (L - x))

        u0 = max(au0 - ea, bu0 - eb)
        u1 = min(au1 + ea, bu1 + eb)
        if u0 > u1:
            u0 = u1 = 0.5 * (u0 + u1)
        v0 = max(av0 - ea, bv0 - eb)
        v1 = min(av1 + ea, bv1 + eb)
        if v0 > v1:
            v0 = v1 = 0.5 * (v0 + v1)

        lo = min(loa + ea, lob + eb)
        hi = max(hia + ea, hib + eb)
        return ((u0, u1, v0, v1), lo, hi, wla + wlb + ea + eb, ea, eb,
                ca + cb, sxa + sxb, sya + syb, qa + qb)

    def _recomputar(self, v: int):
        self.est[v] = self._fundir(self.est[self.esq[v]], self.est[self.dir[v]])

    def recomputar_tudo(self):
        """Recalcula a fase ascendente inteira. Regenera antes a sequência de
        fusões, pois trocas de subárvores deixam `ordem` não topológica."""
        self.reordenar_sequencia()
        for v in self.ordem:
            self._recomputar(v)

    def _recomputar_conjunto(self, S: Set[int]):
        """Recalcula os nós de S (filhos antes dos pais; S fechado nos ancestrais
        relevantes) em tempo O(|S|)."""
        esq, dir_, pai = self.esq, self.dir, self.pai
        pend = {}
        pilha = []
        for v in S:
            c = (1 if esq[v] in S else 0) + (1 if dir_[v] in S else 0)
            pend[v] = c
            if c == 0:
                pilha.append(v)
        while pilha:
            v = pilha.pop()
            self._recomputar(v)
            p = pai[v]
            if p in S:
                pend[p] -= 1
                if pend[p] == 0:
                    pilha.append(p)

    # -- métricas --------------------------------------------------------- #
    def dist_raiz(self) -> float:
        u0, u1, v0, v1 = self.est[self.top][0]
        du = max(u0 - self._ur, 0.0, self._ur - u1)
        dv = max(v0 - self._vr, 0.0, self._vr - v1)
        return du if du > dv else dv

    def wl(self) -> float:
        """Comprimento total de fio WL(T) = fio da árvore + aresta r → topo."""
        return self.est[self.top][3] + self.dist_raiz()

    def skew(self) -> float:
        e = self.est[self.top]
        return e[2] - e[1]

    def skew_ok(self) -> bool:
        s = self.skew()
        return s <= self.delta + 1e-7 * (1.0 + abs(s))

    # -- movimentos (troca de subárvores) -------------------------------- #
    def troca_valida(self, u: int, v: int) -> bool:
        """Nós não-irmãos, nenhum ancestral do outro, nenhum é o topo."""
        if u == v or u == self.top or v == self.top:
            return False
        if self.pai[u] == self.pai[v]:
            return False
        w = self.pai[u]
        while w != -1:
            if w == v:
                return False
            w = self.pai[w]
        w = self.pai[v]
        while w != -1:
            if w == u:
                return False
            w = self.pai[w]
        return True

    def _trocar_ponteiros(self, u: int, v: int):
        pu, pv = self.pai[u], self.pai[v]
        if self.esq[pu] == u:
            self.esq[pu] = v
        else:
            self.dir[pu] = v
        if self.esq[pv] == v:
            self.esq[pv] = u
        else:
            self.dir[pv] = u
        self.pai[u], self.pai[v] = pv, pu

    def aplicar_troca(self, u: int, v: int):
        """Troca as subárvores u e v e recalcula só os ancestrais afetados.
        Devolve (S, antigo) para `desfazer_troca`."""
        pu, pv = self.pai[u], self.pai[v]
        self._trocar_ponteiros(u, v)
        S: Set[int] = set()
        for w in (pu, pv):
            while w != -1 and w not in S:
                S.add(w)
                w = self.pai[w]
        antigo = {w: self.est[w] for w in S}
        self._recomputar_conjunto(S)
        return S, antigo

    def desfazer_troca(self, u: int, v: int, antigo: Dict[int, tuple]):
        self._trocar_ponteiros(u, v)
        for w, e in antigo.items():
            self.est[w] = e

    def reordenar_sequencia(self):
        """Regenera `ordem` como ordem topológica válida, preservando ao máximo a
        ordem relativa anterior (necessário após trocas de subárvores)."""
        n = self.n
        pos = {v: i for i, v in enumerate(self.ordem)}
        pend = {}
        heap = []
        for v in range(n, 2 * n - 1):
            c = (1 if self.esq[v] >= n else 0) + (1 if self.dir[v] >= n else 0)
            pend[v] = c
            if c == 0:
                heap.append((pos[v], v))
        heapq.heapify(heap)
        nova = []
        while heap:
            _, v = heapq.heappop(heap)
            nova.append(v)
            p = self.pai[v]
            if p != -1:
                pend[p] -= 1
                if pend[p] == 0:
                    heapq.heappush(heap, (pos[p], p))
        self.ordem = nova


@dataclass
class ResultadoEmbedding:
    wl: float
    skew: float
    pos: np.ndarray            # (2n-1, 2): posição (x, y) de cada nó
    comp_aresta: np.ndarray    # comprimento da aresta de cada nó ao pai (topo: aresta r→topo)
    folga_snaking: np.ndarray  # comp_aresta − distância geométrica (>0: fio serpenteado)
    atrasos: np.ndarray        # atraso raiz→sink, (n,)


def dme_embed(arv: ArvoreCTS, recalcular: bool = True) -> ResultadoEmbedding:
    """
    **DME-Embed(T)**: fase ascendente (se `recalcular`) + fase descendente.
    Devolve WL, skew realizado, posição dos nós e comprimentos das arestas.
    """
    if recalcular:
        arv.recomputar_tudo()
    n = arv.n
    N = 2 * n - 1
    U = [0.0] * N
    V = [0.0] * N
    comp = [0.0] * N
    folga = [0.0] * N
    atraso = [0.0] * N

    top = arv.top
    r = arv.est[top][0]
    U[top] = min(max(arv._ur, r[0]), r[1])
    V[top] = min(max(arv._vr, r[2]), r[3])
    comp[top] = arv.dist_raiz()
    atraso[top] = comp[top]
    pilha = [top]
    while pilha:
        w = pilha.pop()
        e = arv.est[w]
        for filho, el in ((arv.esq[w], e[4]), (arv.dir[w], e[5])):
            rg = arv.est[filho][0]
            qu = min(max(U[w], rg[0]), rg[1])
            qv = min(max(V[w], rg[2]), rg[3])
            U[filho], V[filho] = qu, qv
            comp[filho] = el
            folga[filho] = el - max(abs(U[w] - qu), abs(V[w] - qv))
            atraso[filho] = atraso[w] + el
            if filho >= n:
                pilha.append(filho)

    Ua, Va = np.array(U), np.array(V)
    pos = np.column_stack([(Ua + Va) / 2.0, (Ua - Va) / 2.0])
    at = np.array(atraso[:n])
    comp_a = np.array(comp)
    return ResultadoEmbedding(
        wl=float(comp_a.sum()), skew=float(at.max() - at.min()), pos=pos,
        comp_aresta=comp_a, folga_snaking=np.array(folga), atrasos=at,
    )


def atrasos_sinks(arv: ArvoreCTS) -> np.ndarray:
    """Atraso raiz→sink de cada sink (só depende dos comprimentos de aresta)."""
    n = arv.n
    atr = [0.0] * (2 * n - 1)
    atr[arv.top] = arv.dist_raiz()
    pilha = [arv.top]
    while pilha:
        w = pilha.pop()
        e = arv.est[w]
        for filho, el in ((arv.esq[w], e[4]), (arv.dir[w], e[5])):
            atr[filho] = atr[w] + el
            if filho >= n:
                pilha.append(filho)
    return np.array(atr[:n])


def verificar_arvore(arv: ArvoreCTS) -> Dict[str, object]:
    """Valida uma solução: conexão de todos os sinks, fio realizável
    (comprimento ≥ distância geométrica), skew ≤ δ e consistência WL/skew da
    avaliação incremental com o embedding completo."""
    wl_inc, skew_inc = arv.wl(), arv.skew()   # valores da avaliação incremental, antes de recalcular
    emb = dme_embed(arv, recalcular=True)
    n = arv.n
    # todos os sinks alcançáveis a partir do topo
    vistos, pilha = 0, [arv.top]
    while pilha:
        w = pilha.pop()
        for f in (arv.esq[w], arv.dir[w]):
            if f < n:
                vistos += 1
            else:
                pilha.append(f)
    escala = 1.0 + abs(emb.wl)
    return {
        "sinks_conectados": vistos == n,
        "fio_realizavel": bool((emb.folga_snaking >= -1e-6 * escala).all()),
        "skew": emb.skew,
        "skew_ok": emb.skew <= arv.delta + 1e-6 * (1.0 + emb.skew),
        "wl": emb.wl,
        "wl_consistente": abs(emb.wl - wl_inc) <= 1e-6 * escala,
        "skew_consistente": abs(emb.skew - skew_inc) <= 1e-6 * escala,
        "snaking_total": float(np.maximum(emb.folga_snaking, 0.0).sum()),
    }


# --------------------------------------------------------------------------- #
# 3. Histórico
# --------------------------------------------------------------------------- #

@dataclass
class HistoricoCTS:
    nome: str = ""
    wl: float = float("inf")
    skew: float = float("nan")
    tempo_total: float = 0.0
    tempo_melhor: float = 0.0
    n_iteracoes: int = 0
    eventos: List[Tuple[float, float]] = field(default_factory=list)  # (t, melhor WL até t)


# --------------------------------------------------------------------------- #
# 4. Construção semi-gulosa (núcleo de H1; reaproveitado em H3)
# --------------------------------------------------------------------------- #

def _tamanho_celula(xs, ys, m):
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    c = 1.5 * max(math.sqrt(w * h / m), max(w, h) / m)
    return c if c > 0 else 1.0


def fusoes_semi_gulosas(px, py, cnt, carga, n_fusoes, k_rcl, alpha, rng=None, k_viz=6):
    """
    Constrói `n_fusoes` fusões bottom-up sobre clusters iniciais (centróides
    px,py; nº de sinks cnt; carga). A cada passo sorteia uniformemente o par
    a fundir dentre os `k_rcl` pares de menor custo

        custo(A,B) = |centróide(A) − centróide(B)|_1 + alpha * |carga(A) − carga(B)|.

    Aceleração: cada cluster mantém seus `k_viz` vizinhos mais próximos (grade
    espacial dinâmica) em um heap com invalidação preguiçosa; os candidatos de
    um cluster são recalculados quando seus parceiros morrem. Assim a RCL é
    *aproximadamente* os k melhores pares globais (exata se k_viz ≥ nº de clusters).

    Devolve (fusoes, raizes, px, py): fusões como pares de índices locais (os
    clusters iniciais são 0..i-1 e a fusão j cria o índice i+j), os índices
    ainda vivos ao final e as listas de centróides estendidas.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    px, py, cnt, carga = list(px), list(py), list(cnt), list(carga)
    i0 = len(px)
    vivo = [True] * i0
    morte_em = [0] * i0
    refresh_em = [-1] * i0
    est = {"mortes": 0, "n_vivos": i0, "m_grade": i0, "x0": 0.0, "y0": 0.0, "cel": 1.0, "maxr": 1}
    grade: Dict[Tuple[int, int], List[int]] = {}
    heap: list = []
    fusoes: List[Tuple[int, int]] = []

    def celula(j):
        return (int((px[j] - est["x0"]) // est["cel"]), int((py[j] - est["y0"]) // est["cel"]))

    def construir_grade():
        ids = [j for j in range(len(px)) if vivo[j]]
        xs = [px[j] for j in ids]
        ys = [py[j] for j in ids]
        est["x0"], est["y0"] = min(xs), min(ys)
        est["cel"] = _tamanho_celula(xs, ys, len(ids))
        est["maxr"] = int(max(max(xs) - min(xs), max(ys) - min(ys)) / est["cel"]) + 2
        est["m_grade"] = len(ids)
        grade.clear()
        for j in ids:
            grade.setdefault(celula(j), []).append(j)

    def vizinhos(a):
        x, y = px[a], py[a]
        cel = est["cel"]
        gx, gy = celula(a)
        cand = []
        r = 0
        while True:
            if r == 0:
                celulas = [(gx, gy)]
            else:
                celulas = ([(gx + dx, gy + r) for dx in range(-r, r + 1)]
                           + [(gx + dx, gy - r) for dx in range(-r, r + 1)]
                           + [(gx + r, gy + dy) for dy in range(-r + 1, r)]
                           + [(gx - r, gy + dy) for dy in range(-r + 1, r)])
            for c in celulas:
                lst = grade.get(c)
                if lst:
                    for j in lst:
                        if j != a:
                            cand.append((abs(px[j] - x) + abs(py[j] - y), j))
            if len(cand) >= k_viz:
                cand.sort()
                if cand[k_viz - 1][0] <= r * cel:
                    break
            if r > est["maxr"]:
                break
            r += 1
        cand.sort()
        return cand[:k_viz]

    def empurrar(a):
        refresh_em[a] = est["mortes"]
        for d, j in vizinhos(a):
            c = d + alpha * abs(carga[a] - carga[j])
            heapq.heappush(heap, (c, a, j) if a < j else (c, j, a))

    construir_grade()
    for a in range(i0):
        empurrar(a)

    for passo in range(n_fusoes):
        rcl = []
        vistos = set()
        while len(rcl) < k_rcl:
            if not heap:
                if rcl:
                    break
                for j in range(len(px)):
                    if vivo[j]:
                        empurrar(j)
                if not heap:
                    break
            c, a, b = heapq.heappop(heap)
            if vivo[a] and vivo[b]:
                if (a, b) in vistos:
                    continue
                vistos.add((a, b))
                rcl.append((c, a, b))
            elif vivo[a]:
                if morte_em[b] > refresh_em[a]:
                    empurrar(a)
            elif vivo[b]:
                if morte_em[a] > refresh_em[b]:
                    empurrar(b)
        if not rcl:
            break
        k = 0 if len(rcl) == 1 else int(rng.integers(len(rcl)))
        _, a, b = rcl[k]
        for idx, e in enumerate(rcl):
            if idx != k:
                heapq.heappush(heap, e)

        cn = cnt[a] + cnt[b]
        px.append((px[a] * cnt[a] + px[b] * cnt[b]) / cn)
        py.append((py[a] * cnt[a] + py[b] * cnt[b]) / cn)
        cnt.append(cn)
        carga.append(carga[a] + carga[b])
        grade[celula(a)].remove(a)
        grade[celula(b)].remove(b)
        vivo[a] = vivo[b] = False
        vivo.append(True)
        est["mortes"] += 2
        morte_em[a] = morte_em[b] = est["mortes"]
        morte_em.append(0)
        refresh_em.append(-1)
        est["n_vivos"] -= 1
        fusoes.append((a, b))
        if passo == n_fusoes - 1 or est["n_vivos"] < 2:
            break
        novo = len(px) - 1
        if est["n_vivos"] < est["m_grade"] / 2:
            construir_grade()
        else:
            grade.setdefault(celula(novo), []).append(novo)
        empurrar(novo)

    raizes = [j for j in range(len(px)) if vivo[j]]
    return fusoes, raizes, px, py


def _alpha_efetivo(inst: InstanciaCTS, alpha: float) -> float:
    """Converte `alpha` (adimensional) para unidades de distância por unidade de carga."""
    return alpha * inst.escala_distancia() / float(np.mean(inst.cargas))


def construir_h1(inst: InstanciaCTS, delta: float, k_rcl: int = 3, alpha: float = 0.3,
                 k_viz: int = 6, rng: Optional[np.random.Generator] = None) -> ArvoreCTS:
    """Construção semi-gulosa balanceada (Algoritmo 1) + DME-Embed."""
    rng = rng if rng is not None else np.random.default_rng()
    n = inst.n
    fusoes, _, _, _ = fusoes_semi_gulosas(
        inst.xy[:, 0].tolist(), inst.xy[:, 1].tolist(), [1] * n, inst.cargas.tolist(),
        n - 1, k_rcl, _alpha_efetivo(inst, alpha), rng, k_viz)
    return ArvoreCTS.de_fusoes(inst, delta, fusoes)


# --------------------------------------------------------------------------- #
# 5. Busca local de troca de subárvores (H2 e H3)
# --------------------------------------------------------------------------- #

def _gerar_vizinhanca(arv, rng, foco, k_viz, raio, incluir_folhas, max_foco):
    """Pares (u, v) candidatos: vizinhos geométricos (KNN de centróides) ∪
    vizinhos a distância de níveis ≤ `raio` na árvore. Se `foco` é dado, só
    pares que contêm um nó de `foco` (amostrado em até `max_foco` nós)."""
    from scipy.spatial import cKDTree

    n = arv.n
    lo = 0 if incluir_folhas else n
    nos = np.array([v for v in range(lo, 2 * n - 1) if v != arv.top], dtype=int)
    if len(nos) < 3:
        return []
    est = arv.est
    pts = np.empty((len(nos), 2))
    for i, v in enumerate(nos.tolist()):
        e = est[v]
        pts[i, 0] = e[7] / e[6]
        pts[i, 1] = e[8] / e[6]
    tree = cKDTree(pts)

    if foco is None:
        alvo = np.arange(len(nos))
    else:
        mapa = {v: i for i, v in enumerate(nos.tolist())}
        alvo = np.array([mapa[v] for v in foco if v in mapa], dtype=int)
        if max_foco and len(alvo) > max_foco:
            alvo = rng.choice(alvo, size=max_foco, replace=False)

    pares: Set[Tuple[int, int]] = set()
    if len(alvo):
        kq = min(k_viz + 1, len(nos))
        _, ix = tree.query(pts[alvo], k=kq)
        for a, linha in zip(alvo.tolist(), np.atleast_2d(ix).tolist()):
            u = int(nos[a])
            for j in linha:
                v = int(nos[j])
                if v != u:
                    pares.add((u, v) if u < v else (v, u))
        if raio > 0:
            conj = set(nos.tolist())
            for a in alvo.tolist():
                u = int(nos[a])
                visto = {u}
                fronteira = [u]
                for _ in range(raio):
                    nova = []
                    for w in fronteira:
                        for x in (arv.pai[w], arv.esq[w], arv.dir[w]):
                            if x >= 0 and x not in visto:
                                visto.add(x)
                                nova.append(x)
                    fronteira = nova
                for x in visto:
                    if x != u and x in conj:
                        pares.add((u, x) if u < x else (x, u))
    lista = sorted(pares)
    return [lista[i] for i in rng.permutation(len(lista))]


def busca_local(
    arv: ArvoreCTS,
    rng: np.random.Generator,
    t_fim: float,
    foco: Optional[Iterable[int]] = None,
    k_vizinhos: int = 6,
    raio_arvore: int = 3,
    incluir_folhas: bool = False,
    max_passadas: Optional[int] = None,
    max_foco: Optional[int] = None,
    callback: Optional[Callable[[float], None]] = None,
) -> int:
    """
    **BuscaLocal** (Algoritmo 2): troca de subárvores entre nós não-irmãos, sem
    relação de ancestralidade, reavaliada via DME-Embed (incremental) e aceita
    se reduz WL sem violar δ (primeira melhoria). Repete passadas até um ótimo
    local (ou até `t_fim` / `max_passadas`). Devolve o nº de movimentos aceitos.
    A árvore é modificada no lugar.
    """
    foco = None if foco is None else set(foco)
    aceitos = 0
    passada = 0
    while True:
        passada += 1
        pares = _gerar_vizinhanca(arv, rng, foco, k_vizinhos, raio_arvore, incluir_folhas, max_foco)
        if not pares:
            break
        wl0 = arv.wl()
        melhorou = False
        for u, v in pares:
            if time.perf_counter() >= t_fim:
                return aceitos
            if not arv.troca_valida(u, v):
                continue
            _, antigo = arv.aplicar_troca(u, v)
            wl1 = arv.wl()
            if wl1 < wl0 - TOL * max(1.0, abs(wl0)) and arv.skew_ok():
                wl0 = wl1
                aceitos += 1
                melhorou = True
                if callback is not None:
                    callback(wl1)
            else:
                arv.desfazer_troca(u, v, antigo)
        if not melhorou or (max_passadas is not None and passada >= max_passadas):
            break
    return aceitos


# --------------------------------------------------------------------------- #
# 6. Heurísticas propostas: H1, H2, H3
# --------------------------------------------------------------------------- #

def heuristica_h1(inst: InstanciaCTS, delta: float, semente: Optional[int] = None,
                  k_rcl: int = 3, alpha: float = 0.3, k_viz: int = 6
                  ) -> Tuple[ArvoreCTS, HistoricoCTS]:
    """**H1 – BSG-DME**: construção semi-gulosa balanceada (GRASP) + DME-Embed."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(semente)
    arv = construir_h1(inst, delta, k_rcl, alpha, k_viz, rng)
    dt = time.perf_counter() - t0
    h = HistoricoCTS(nome="H1", wl=arv.wl(), skew=arv.skew(), tempo_total=dt,
                     tempo_melhor=dt, n_iteracoes=1)
    h.eventos.append((dt, h.wl))
    return arv, h


def heuristica_h2(inst: InstanciaCTS, delta: float, t_max: float, semente: Optional[int] = None,
                  k_rcl: int = 3, alpha: float = 0.3, k_viz: int = 6,
                  vizinhos_busca: int = 6, raio_arvore: int = 3,
                  incluir_folhas: bool = False) -> Tuple[ArvoreCTS, HistoricoCTS]:
    """
    **H2 – GRASP-CTS** (Algoritmo 2): repete [H1 + BuscaLocal] até `t_max` segundos
    e devolve a melhor árvore. A busca local é interrompida ao fim do orçamento
    (a árvore parcialmente melhorada continua válida).
    """
    rng = np.random.default_rng(semente)
    t0 = time.perf_counter()
    t_fim = t0 + t_max
    h = HistoricoCTS(nome="H2")
    melhor: Optional[ArvoreCTS] = None
    estado = {"melhor_wl": math.inf}

    def registrar(wl):
        if wl < estado["melhor_wl"]:
            estado["melhor_wl"] = wl
            t = time.perf_counter() - t0
            h.eventos.append((t, wl))
            h.tempo_melhor = t

    while True:
        T = construir_h1(inst, delta, k_rcl, alpha, k_viz, rng)
        registrar(T.wl())
        busca_local(T, rng, t_fim, None, vizinhos_busca, raio_arvore, incluir_folhas,
                    callback=registrar)
        h.n_iteracoes += 1
        if melhor is None or T.wl() < melhor.wl():
            melhor = T
        if time.perf_counter() >= t_fim:
            break
    h.tempo_total = time.perf_counter() - t0
    h.wl, h.skew = melhor.wl(), melhor.skew()
    melhor.reordenar_sequencia()   # as trocas da busca local desordenam a sequência de fusões
    return melhor, h


def _destruir_e_reparar(T: ArvoreCTS, rng, frac, max_fusoes, k_rcl, alpha_ef, k_viz) -> Set[int]:
    """
    Destroy/repair de H3. Remove um fragmento contíguo da sequência de fusões e o
    reconstrói por construção semi-gulosa restrita às entradas do fragmento. As
    novas fusões reaproveitam os índices removidos, na ordem de criação (a
    sequência continua topológica). As raízes do fragmento reconstruído são
    religadas aos pontos de conexão antigos por atribuição ótima de centróides.
    Devolve o conjunto de nós novos (foco da busca local).
    """
    from scipy.optimize import linear_sum_assignment

    m = len(T.ordem)
    f = int(rng.uniform(*frac) * m)
    f = max(2, min(f, max_fusoes, m))
    k0 = int(rng.integers(0, m - f + 1))
    janela = T.ordem[k0:k0 + f]
    W = set(janela)

    entradas, raizes = [], []
    for v in janela:
        for c in (T.esq[v], T.dir[v]):
            if c not in W:
                entradas.append(c)
        if T.pai[v] not in W:
            raizes.append(v)

    # pontos de conexão antigos e seus centróides (antes de qualquer modificação)
    slots = []
    for v in raizes:
        p = T.pai[v]
        slots.append((p, (T.esq[p] == v) if p != -1 else True))
    ox = [T.est[v][7] / T.est[v][6] for v in raizes]
    oy = [T.est[v][8] / T.est[v][6] for v in raizes]

    px = [T.est[v][7] / T.est[v][6] for v in entradas]
    py = [T.est[v][8] / T.est[v][6] for v in entradas]
    cnt = [T.est[v][6] for v in entradas]
    carga = [T.est[v][9] for v in entradas]
    i = len(entradas)
    fusoes, raizes_loc, px, py = fusoes_semi_gulosas(px, py, cnt, carga, f, k_rcl, alpha_ef, rng, k_viz)

    def mapa(idx):
        return entradas[idx] if idx < i else janela[idx - i]

    for j, (a, b) in enumerate(fusoes):
        v = janela[j]
        na, nb = mapa(a), mapa(b)
        T.esq[v], T.dir[v] = na, nb
        T.pai[na] = v
        T.pai[nb] = v

    novas = [mapa(r) for r in raizes_loc]
    custo = np.array([[abs(px[r] - ox[s]) + abs(py[r] - oy[s]) for s in range(len(slots))]
                      for r in raizes_loc])
    linhas, colunas = linear_sum_assignment(custo)
    S: Set[int] = set(janela)
    for li, co in zip(linhas.tolist(), colunas.tolist()):
        nr = novas[li]
        p, eh_esq = slots[co]
        T.pai[nr] = p
        if p == -1:
            T.top = nr
        else:
            if eh_esq:
                T.esq[p] = nr
            else:
                T.dir[p] = nr
            w = p
            while w != -1 and w not in S:
                S.add(w)
                w = T.pai[w]
    T._recomputar_conjunto(S)
    return set(janela)


def heuristica_h3(
    inst: InstanciaCTS, delta: float, t_max: float, semente: Optional[int] = None,
    k_rcl: int = 3, alpha: float = 0.3, k_viz: int = 6,
    theta0_rel: float = 1e-3, theta_final_rel: float = 1e-2,
    taxa_resfriamento: Optional[float] = None,
    frac_fragmento: Tuple[float, float] = (0.02, 0.10), max_fusoes: int = 200,
    busca_focada: bool = True, max_foco: int = 64, max_passadas_foco: int = 2,
    vizinhos_busca: int = 6, raio_arvore: int = 3, incluir_folhas: bool = False,
) -> Tuple[ArvoreCTS, HistoricoCTS]:
    """
    **H3 – ILS/SA-CTS** (Algoritmo 3). Parte de uma árvore H1; a cada iteração:
    destroy (fragmento contíguo da sequência de fusões) → repair (semi-guloso
    restrito, reaproveitando índices) → BuscaLocal → aceitação de Metropolis
    sobre WL. Guarda a melhor árvore vista.

    Pontos de escala/implementação (ver docstring do módulo):
    - `busca_focada=True`: a BuscaLocal examina só pares que envolvem os nós
      reconstruídos (até `max_foco`), em vez da vizinhança da árvore inteira.
    - `taxa_resfriamento=None`: θ decai geometricamente em função do tempo
      decorrido, de θ0 = theta0_rel·WL(T0) até θ0·theta_final_rel em `t_max`
      (equivale a θ ← ρθ com ρ ajustado ao orçamento). Um float ativa θ ← ρ·θ por iteração.
    """
    rng = np.random.default_rng(semente)
    t0 = time.perf_counter()
    t_fim = t0 + t_max
    alpha_ef = _alpha_efetivo(inst, alpha)
    h = HistoricoCTS(nome="H3")

    T = construir_h1(inst, delta, k_rcl, alpha, k_viz, rng)
    wl_T = T.wl()
    melhor, melhor_wl = T, wl_T
    h.eventos.append((time.perf_counter() - t0, wl_T))
    theta0 = max(theta0_rel * wl_T, 1e-12)
    theta = theta0

    while time.perf_counter() < t_fim and inst.n >= 4:
        agora = time.perf_counter() - t0
        if taxa_resfriamento is None:
            theta = theta0 * theta_final_rel ** min(1.0, agora / max(t_max, 1e-9))
        Tn = T.copiar()
        foco = _destruir_e_reparar(Tn, rng, frac_fragmento, max_fusoes, k_rcl, alpha_ef, k_viz)
        moveu = busca_local(
            Tn, rng, t_fim, foco if busca_focada else None, vizinhos_busca, raio_arvore,
            incluir_folhas, max_passadas=max_passadas_foco if busca_focada else None,
            max_foco=max_foco if busca_focada else None)
        h.n_iteracoes += 1
        wl_n = Tn.wl()
        d = wl_n - wl_T
        if Tn.skew_ok() and (d < 0 or rng.random() < math.exp(-d / theta)):
            if moveu:
                Tn.reordenar_sequencia()
            T, wl_T = Tn, wl_n
            if wl_T < melhor_wl - TOL * max(1.0, melhor_wl):
                melhor, melhor_wl = T, wl_T
                t = time.perf_counter() - t0
                h.eventos.append((t, melhor_wl))
                h.tempo_melhor = t
        if taxa_resfriamento is not None:
            theta *= taxa_resfriamento
    h.tempo_total = time.perf_counter() - t0
    h.wl, h.skew = melhor.wl(), melhor.skew()
    return melhor, h


# --------------------------------------------------------------------------- #
# 7. Heurísticas de controle — APENAS ASSINATURAS (a implementar depois)
# --------------------------------------------------------------------------- #
# Contrato comum (para o runner e as células do notebook funcionarem sem mudanças):
#
#   def heuristica_cX(inst: InstanciaCTS, delta: float, semente: Optional[int] = None,
#                     **kwargs) -> Tuple[Optional[ArvoreCTS], HistoricoCTS]
#
# - `inst`: instância; `delta`: limite de skew δ (pode ser math.inf: "sem restrição").
# - Devolve (arvore, hist). Se a heurística gera uma árvore binária (C2, C3),
#   `arvore` é uma `ArvoreCTS` (use `ArvoreCTS.de_fusoes`) e `hist.wl`/`hist.skew`
#   são `arvore.wl()`/`arvore.skew()`. Se a estrutura não é binária (C1), `arvore`
#   é None e `hist.wl`/`hist.skew` vêm da geometria.
# - `hist` é um `HistoricoCTS` com `nome`, `wl`, `skew` (realizado), `tempo_total`
#   e `tempo_melhor` preenchidos. Heurísticas determinísticas ignoram `semente`.
# - O skew devolvido deve ser ≤ δ para a comparação ser justa nas hipóteses.
#   Se a implementação não garante isso por construção (caso do C1), deve
#   compensar (ex.: snaking nos fios finais) ou o skew violado deve aparecer
#   em `hist.skew` — o runner marca `viavel=False`.
# - Parâmetros próprios podem ser passados em `PARAMS["C1"]`, `PARAMS["C2"]`, ...
#   (dicionário repassado como **kwargs).

def heuristica_c1(inst: InstanciaCTS, delta: float, semente: Optional[int] = None,
                  **kwargs) -> Tuple[Optional[ArvoreCTS], HistoricoCTS]:
    """
    **C1 – H-Tree/X-Tree ajustada** (Algoritmo C1 do pré-projeto): subdivisão
    recursiva do retângulo envolvente dos sinks em 4 quadrantes, ligação dos
    centros em padrão "H" até profundidade D (ou região com ≤ 1 sink), sinks
    ligados ao centro da região-folha; WL e skew calculados pela geometria fixa
    (sem DME-Embed). Retorna `(None, hist)`.

    Parâmetros sugeridos em kwargs: `profundidade` (D) e se o skew deve ser
    compensado até ≤ δ por snaking. NÃO IMPLEMENTADA.
    """
    raise NotImplementedError("C1 (H-Tree/X-Tree ajustada) ainda não implementada: ver contrato acima.")


def heuristica_c2(inst: InstanciaCTS, delta: float, semente: Optional[int] = None,
                  **kwargs) -> Tuple[Optional[ArvoreCTS], HistoricoCTS]:
    """
    **C2 – NN+DME**: topologia puramente gulosa e determinística (sempre funde os
    dois clusters de centróides mais próximos, sem balanceamento de carga nem
    aleatoriedade) + DME-Embed. Retorna `(ArvoreCTS, hist)`.

    Também é usada por `calibrar_delta` com `delta = math.inf` ("sem restrição de
    skew") para definir δ de cada instância; nesse caso deve devolver uma árvore.
    Dica: `fusoes_semi_gulosas(..., k_rcl=1, alpha=0.0)` + `ArvoreCTS.de_fusoes`
    já fornecem a construção; a decisão de reutilizá-las é sua. NÃO IMPLEMENTADA.
    """
    raise NotImplementedError("C2 (NN+DME) ainda não implementada: ver contrato acima.")


def heuristica_c3(inst: InstanciaCTS, delta: float, semente: Optional[int] = None,
                  **kwargs) -> Tuple[Optional[ArvoreCTS], HistoricoCTS]:
    """
    **C3 – MMM+DME** (Jackson, Srinivasan e Kuh): topologia top-down por
    bipartição recursiva na mediana, alternando eixos x/y, + DME-Embed.
    Retorna `(ArvoreCTS, hist)`. NÃO IMPLEMENTADA.
    """
    raise NotImplementedError("C3 (MMM+DME) ainda não implementada: ver contrato acima.")


FUNCOES_CONTROLE = {"C1": heuristica_c1, "C2": heuristica_c2, "C3": heuristica_c3}


def controle_disponivel(nome: str) -> bool:
    """True se a heurística de controle `nome` já foi implementada (não levanta
    NotImplementedError numa instância minúscula)."""
    inst = gerar_instancia_sintetica(8, "uniforme", 0)
    try:
        FUNCOES_CONTROLE[nome](inst, 1.0, semente=0)
    except NotImplementedError:
        return False
    except Exception:
        return True   # implementada (o erro aparecerá na execução real)
    return True


# --------------------------------------------------------------------------- #
# 8. Experimentos
# --------------------------------------------------------------------------- #

def orcamento_tempo(n: int, tempo_por_sink: float = 0.02, tempo_minimo: float = 2.0) -> float:
    """Orçamento de tempo (s) de H2 e H3, proporcional a n, igual para as duas."""
    return max(tempo_minimo, tempo_por_sink * n)


def estimar_tempo_total(tamanhos, n_instancias, R, tempo_por_sink=0.02, tempo_minimo=2.0,
                        n_distribuicoes=2, n_processos=1) -> float:
    """Estimativa (horas) do tempo gasto por H2 + H3 (domina o experimento)."""
    s = sum(2 * R * n_instancias * n_distribuicoes * orcamento_tempo(n, tempo_por_sink, tempo_minimo)
            for n in tamanhos)
    return s / max(n_processos, 1) / 3600.0


def calibrar_delta(inst: InstanciaCTS, percentual: float = 0.05) -> float:
    """
    δ da instância = `percentual` × atraso médio raiz-sink da **C2 sem restrição de
    skew** (δ = ∞). Enquanto C2 não estiver implementada usa-se, de forma
    **provisória**, uma árvore H1 sem restrição de skew (semente 0), com aviso.
    """
    try:
        arv, _ = heuristica_c2(inst, math.inf, semente=0)
    except NotImplementedError:
        warnings.warn("C2 não implementada: δ calibrado provisoriamente com H1 (δ=∞). "
                      "Recalibre quando C2 existir.", stacklevel=2)
        arv, _ = heuristica_h1(inst, math.inf, semente=0)
    if arv is None:
        raise ValueError("C2 com δ=∞ deve devolver uma ArvoreCTS para calibrar δ.")
    return percentual * float(atrasos_sinks(arv).mean())


def executar_heuristica(nome: str, inst: InstanciaCTS, delta: float, semente: int,
                        t_max: float, params: Optional[dict] = None):
    """Executa uma heurística e devolve (linha_resultado: dict, historico)."""
    p = dict(PARAMS_PADRAO)
    p.update({k: v for k, v in (params or {}).items() if k not in FUNCOES_CONTROLE})
    if nome == "H1":
        arv, h = heuristica_h1(inst, delta, semente, p["k_rcl"], p["alpha"], p["k_viz"])
    elif nome == "H2":
        arv, h = heuristica_h2(inst, delta, t_max, semente, p["k_rcl"], p["alpha"], p["k_viz"],
                               p["vizinhos_busca"], p["raio_arvore"], p["incluir_folhas"])
    elif nome == "H3":
        arv, h = heuristica_h3(
            inst, delta, t_max, semente, p["k_rcl"], p["alpha"], p["k_viz"], p["theta0_rel"],
            p["theta_final_rel"], p["taxa_resfriamento"], p["frac_fragmento"], p["max_fusoes"],
            p["busca_focada"], p["max_foco"], p["max_passadas_foco"], p["vizinhos_busca"],
            p["raio_arvore"], p["incluir_folhas"])
    elif nome in FUNCOES_CONTROLE:
        kw = (params or {}).get(nome, {})
        arv, h = FUNCOES_CONTROLE[nome](inst, delta, semente=semente, **kw)
    else:
        raise ValueError(f"heurística desconhecida: {nome}")
    linha = {
        "instancia": inst.nome, "n": inst.n, "distribuicao": inst.distribuicao,
        "heuristica": nome, "semente": semente, "delta": delta,
        "wl": h.wl, "skew": h.skew,
        "viavel": bool(h.skew <= delta * (1 + 1e-6) + 1e-6),
        "tempo_s": h.tempo_total, "tempo_melhor_s": h.tempo_melhor, "n_iter": h.n_iteracoes,
    }
    return linha, h


def _tarefa(args):
    inst, nome, semente, delta, t_max, params = args
    linha, h = executar_heuristica(nome, inst, delta, semente, t_max, params)
    return linha, (inst.nome, nome, semente), h


def executar_experimento(
    instancias: Sequence[InstanciaCTS],
    percentual_skew: float = 0.05,
    R: int = 20,
    tempo_por_sink: float = 0.02,
    tempo_minimo: float = 2.0,
    heuristicas: Sequence[str] = ("H1", "H2", "H3", "C1", "C2", "C3"),
    params: Optional[dict] = None,
    arquivo_cache: Optional[str] = None,
    n_processos: int = 1,
    semente_base: int = 1000,
    salvar_a_cada: int = 25,
    verbose: bool = True,
):
    """
    Executa cada heurística em cada instância (estocásticas: R sementes;
    controles determinísticos: uma execução), com o mesmo δ por instância.
    Controles ainda não implementados são ignorados (com aviso). Se
    `arquivo_cache` existe, retoma de onde parou (pula (instância, heurística,
    semente) já feitas).

    Devolve (df, historicos, deltas): `df` com uma linha por execução,
    `historicos[(instancia, heuristica, semente)]` com as curvas de convergência
    e `deltas[instancia]`.
    """
    ativas = []
    for h in heuristicas:
        if h in FUNCOES_CONTROLE and not controle_disponivel(h):
            if verbose:
                print(f"[aviso] {h} não implementada: ignorada neste experimento.")
            continue
        ativas.append(h)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deltas = {inst.nome: calibrar_delta(inst, percentual_skew) for inst in instancias}

    linhas: List[dict] = []
    historicos: Dict[tuple, HistoricoCTS] = {}
    if arquivo_cache and os.path.exists(arquivo_cache):
        with open(arquivo_cache, "rb") as f:
            cache = pickle.load(f)
        linhas, historicos = cache["linhas"], cache["historicos"]
        if verbose:
            print(f"[cache] {len(linhas)} execuções carregadas de {arquivo_cache}")
    feitas = {(l["instancia"], l["heuristica"], l["semente"]) for l in linhas}

    tarefas = []
    for inst in instancias:
        t_max = orcamento_tempo(inst.n, tempo_por_sink, tempo_minimo)
        for h in ativas:
            sementes = range(R) if h in ESTOCASTICAS else [0]
            for s in sementes:
                semente = semente_base + int(s)
                if (inst.nome, h, semente) not in feitas:
                    tarefas.append((inst, h, semente, deltas[inst.nome], t_max, params))

    def salvar():
        if arquivo_cache:
            tmp = arquivo_cache + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump({"linhas": linhas, "historicos": historicos}, f)
            os.replace(tmp, arquivo_cache)

    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover
        tqdm = None

    def consumir(resultados):
        it = tqdm(resultados, total=len(tarefas), desc="CTS") if (tqdm and verbose) else resultados
        for k, (linha, chave, hist) in enumerate(it, 1):
            linhas.append(linha)
            historicos[chave] = hist
            if k % salvar_a_cada == 0:
                salvar()

    if n_processos > 1 and tarefas:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_processos) as ex:
            consumir(ex.map(_tarefa, tarefas, chunksize=1))
    else:
        consumir(map(_tarefa, tarefas))
    salvar()
    return pd.DataFrame(linhas), historicos, deltas


# --------------------------------------------------------------------------- #
# 9. Análise: tabelas, gap e testes das hipóteses
# --------------------------------------------------------------------------- #

def adicionar_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta `wl_melhor` (melhor WL da instância entre todas as execuções
    disponíveis) e `gap_pct` = 100·(WL/WL_melhor − 1)."""
    d = df.copy()
    d["wl_melhor"] = d.groupby("instancia")["wl"].transform("min")
    d["gap_pct"] = 100.0 * (d["wl"] / d["wl_melhor"] - 1.0)
    return d


def tabela_resultados(df: pd.DataFrame) -> pd.DataFrame:
    """[Instância, n, Heurística, WL, Skew, Tempo (s), Gap (%)] (média sobre sementes)."""
    d = adicionar_gap(df)
    t = d.groupby(["instancia", "n", "heuristica"], sort=False).agg(
        WL=("wl", "mean"), Skew=("skew", "max"), Tempo_s=("tempo_melhor_s", "mean"),
        Gap_pct=("gap_pct", "mean")).reset_index()
    t.columns = ["Instância", "n", "Heurística", "WL (µm)", "Skew", "Tempo (s)", "Gap (%)"]
    return t.round(3)


def resumir_resultados(df: pd.DataFrame) -> pd.DataFrame:
    """Resumo por heurística (e por n): gap médio, WL médio, skew máximo, % viável, tempo."""
    d = adicionar_gap(df)
    g = d.groupby(["n", "heuristica"], sort=True)
    return pd.DataFrame({
        "wl_medio": g["wl"].mean(), "wl_dp": g["wl"].std(ddof=0),
        "gap_medio_pct": g["gap_pct"].mean(), "skew_max": g["skew"].max(),
        "taxa_viavel": g["viavel"].mean(), "tempo_melhor_s": g["tempo_melhor_s"].mean(),
    }).round(3)


def _wl_por_instancia(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz instância × heurística com o WL médio sobre as sementes."""
    return df.groupby(["instancia", "heuristica"])["wl"].mean().unstack()


def testar_hipotese_1(df: pd.DataFrame, n_max: int = 5000, nivel: float = 0.05) -> Dict[str, object]:
    """
    H1 (qualidade frente aos baselines): confirmada se ≥ 2 heurísticas propostas
    têm WL médio estatisticamente menor (Wilcoxon signed-rank unilateral,
    pareado por instância, p < nivel) que ≥ 2 controles, com 100% das soluções
    **das propostas** com skew ≤ δ. Requer ≥ 2 controles implementados; caso
    contrário `veredito=None` (pendente, não refutada).
    """
    from scipy.stats import wilcoxon

    d = df[df["n"] <= n_max]
    piv = _wl_por_instancia(d)
    props = [h for h in HEURISTICAS_PROPOSTAS if h in piv.columns]
    ctrls = [c for c in CONTROLES if c in piv.columns]
    linhas = []
    for h in props:
        for c in ctrls:
            x, y = piv[h].values, piv[c].values
            try:
                p = float(wilcoxon(x, y, alternative="less").pvalue)
            except ValueError:
                p = float("nan")
            linhas.append({
                "proposta": h, "controle": c, "wl_proposta": x.mean(), "wl_controle": y.mean(),
                "reducao_pct": 100 * (1 - x.mean() / y.mean()), "p_valor": p,
                "significativo": bool(p < nivel and x.mean() < y.mean()),
            })
    tabela = pd.DataFrame(linhas)
    skew_ok = bool(d[d["heuristica"].isin(props)]["viavel"].all()) if props else False
    skew_ctrl_ok = {c: bool(d[d["heuristica"] == c]["viavel"].all()) for c in ctrls}
    if len(ctrls) < 2 or tabela.empty:
        return {"veredito": None, "tabela": tabela, "skew_ok": skew_ok, "skew_controles_ok": skew_ctrl_ok,
                "motivo": f"pendente: requer ≥ 2 controles implementados (disponíveis: {ctrls})"}
    vitorias = tabela.groupby("proposta")["significativo"].sum().to_dict()
    props_ok = [h for h, v in vitorias.items() if v >= 2]
    return {
        "veredito": bool(len(props_ok) >= 2 and skew_ok), "tabela": tabela,
        "vitorias": vitorias, "propostas_que_vencem_2_controles": props_ok,
        "skew_ok": skew_ok, "skew_controles_ok": skew_ctrl_ok,
        "controles_completos": len(ctrls) == len(CONTROLES),
        "motivo": "",
    }


def testar_hipotese_2(df: pd.DataFrame, razao_minima: float = 1.5) -> Dict[str, object]:
    """
    H2 (robustez a distribuições assimétricas): gap de cada proposta em relação a
    C1, gap = 100·(WL_C1 − WL_H)/WL_C1 por instância (positivo: a proposta é
    melhor). Confirmada se, para as três propostas, gap médio (clusterizada) ≥
    `razao_minima` × gap médio (uniforme) e > 0. Requer C1 implementada.
    """
    piv = _wl_por_instancia(df)
    if "C1" not in piv.columns:
        return {"veredito": None, "tabela": pd.DataFrame(), "tabela_por_n": pd.DataFrame(),
                "motivo": "pendente: C1 não implementada"}
    info = df.drop_duplicates("instancia").set_index("instancia")[["n", "distribuicao"]]
    linhas = []
    for h in [x for x in HEURISTICAS_PROPOSTAS if x in piv.columns]:
        gap = 100.0 * (piv["C1"] - piv[h]) / piv["C1"]
        tmp = info.join(gap.rename("gap_vs_c1")).assign(heuristica=h)
        linhas.append(tmp.reset_index())
    inst_gap = pd.concat(linhas, ignore_index=True)
    por_n = inst_gap.groupby(["heuristica", "n", "distribuicao"])["gap_vs_c1"].mean().unstack("distribuicao")
    agg = inst_gap.groupby(["heuristica", "distribuicao"])["gap_vs_c1"].mean().unstack("distribuicao")
    res = []
    for h in agg.index:
        gu, gc = float(agg.loc[h, "uniforme"]), float(agg.loc[h, "clusterizada"])
        res.append({
            "heuristica": h, "gap_uniforme_pct": gu, "gap_clusterizada_pct": gc,
            "razao": (gc / gu) if gu > 0 else float("inf") if gc > 0 else float("nan"),
            "cumpre": bool(gc > 0 and gc >= razao_minima * gu),
        })
    tabela = pd.DataFrame(res).round(3)
    sk = df[df["heuristica"].isin(HEURISTICAS_PROPOSTAS) | (df["heuristica"] == "C1")]
    skew_tab = sk.groupby(["heuristica", "distribuicao"])["skew"].agg(["mean", "max"]).round(3)
    return {"veredito": bool(tabela["cumpre"].all()) and len(tabela) == 3, "tabela": tabela,
            "tabela_por_n": por_n.round(3), "skew": skew_tab, "inst_gap": inst_gap, "motivo": ""}


def testar_hipotese_3(df: pd.DataFrame, fracao_minima: float = 0.8, margem: float = 0.01) -> Dict[str, object]:
    """
    H3 (efeito da busca local e da perturbação ILS/SA): para cada n, fração de
    instâncias com WL(H3) ≤ WL(H2)·(1+margem) e WL(H2) ≤ WL(H1)·(1+margem)
    (WL médio sobre as sementes). Confirmada se a fração ≥ `fracao_minima` em
    todos os n testados. Não depende dos controles.
    """
    piv = _wl_por_instancia(df)
    falta = [h for h in HEURISTICAS_PROPOSTAS if h not in piv.columns]
    if falta:
        return {"veredito": None, "tabela": pd.DataFrame(), "motivo": f"faltam resultados de {falta}"}
    info = df.drop_duplicates("instancia").set_index("instancia")[["n"]]
    p = piv[list(HEURISTICAS_PROPOSTAS)].join(info)
    p["h3_le_h2"] = p["H3"] <= p["H2"] * (1 + margem)
    p["h2_le_h1"] = p["H2"] <= p["H1"] * (1 + margem)
    p["ordem_ok"] = p["h3_le_h2"] & p["h2_le_h1"]
    p["ganho_h2_h1_pct"] = 100 * (1 - p["H2"] / p["H1"])
    p["ganho_h3_h2_pct"] = 100 * (1 - p["H3"] / p["H2"])
    tab = p.groupby("n").agg(
        n_instancias=("ordem_ok", "size"), frac_h3_le_h2=("h3_le_h2", "mean"),
        frac_h2_le_h1=("h2_le_h1", "mean"), frac_ordem_ok=("ordem_ok", "mean"),
        ganho_h2_sobre_h1_pct=("ganho_h2_h1_pct", "mean"),
        ganho_h3_sobre_h2_pct=("ganho_h3_h2_pct", "mean")).round(3)
    return {"veredito": bool((tab["frac_ordem_ok"] >= fracao_minima).all()), "tabela": tab,
            "n_max_testado": int(tab.index.max()), "motivo": ""}


def imprimir_veredito(nome: str, res: Dict[str, object]):
    v = res["veredito"]
    txt = "PENDENTE" if v is None else ("CONFIRMADA" if v else "REFUTADA")
    print(f"{nome}: {txt}" + (f"  ({res['motivo']})" if res.get("motivo") else ""))


# --------------------------------------------------------------------------- #
# 10. Visualização
# --------------------------------------------------------------------------- #

def plotar_instancia(inst: InstanciaCTS, ax=None, titulo: Optional[str] = None):
    """Sinks (cor ∝ carga) e fonte de clock."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(inst.xy[:, 0], inst.xy[:, 1], c=inst.cargas, s=8, cmap="viridis")
    ax.scatter([inst.raiz[0]], [inst.raiz[1]], marker="s", s=70, color="red", edgecolor="black", label="fonte r")
    ax.set_aspect("equal")
    ax.set_title(titulo or f"{inst.nome} (n={inst.n}, {inst.distribuicao})")
    ax.legend(loc="upper right", fontsize=7)
    return ax


def plotar_arvore(arv: ArvoreCTS, ax=None, titulo: str = "", mostrar_steiner: bool = True):
    """Desenha a árvore embutida: fios em L (horizontal→vertical); fios serpenteados
    (snaking) não têm o desvio desenhado, mas a diferença aparece no título (WL)."""
    import matplotlib.pyplot as plt
    emb = dme_embed(arv, recalcular=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    pos, n = emb.pos, arv.n
    rx, ry = arv.inst.raiz
    for v in range(2 * n - 1):
        p = arv.pai[v]
        qx, qy = (pos[p] if p != -1 else (rx, ry))
        x, y = pos[v]
        cor = "tab:blue" if v < n else "tab:gray"
        ax.plot([x, qx, qx], [y, y, qy], color=cor, linewidth=0.6 if v < n else 0.9, zorder=1)
    ax.scatter(pos[:n, 0], pos[:n, 1], s=6, color="tab:blue", zorder=2)
    if mostrar_steiner:
        ax.scatter(pos[n:, 0], pos[n:, 1], s=8, color="tab:orange", zorder=2)
    ax.scatter([rx], [ry], marker="s", s=70, color="red", edgecolor="black", zorder=3)
    ax.set_aspect("equal")
    ax.set_title(f"{titulo}  WL={emb.wl:,.0f}  skew={emb.skew:,.1f}")
    return ax


def plotar_boxplot_gap(df: pd.DataFrame, n_max: Optional[int] = None, ax=None):
    """Boxplot do Gap (%) por heurística (gap relativo à melhor solução da instância)."""
    import matplotlib.pyplot as plt
    d = adicionar_gap(df if n_max is None else df[df["n"] <= n_max])
    ordem = [h for h in HEURISTICAS_PROPOSTAS + CONTROLES if h in set(d["heuristica"])]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([d.loc[d["heuristica"] == h, "gap_pct"].values for h in ordem])
    ax.set_xticklabels(ordem)
    ax.set_ylabel("Gap (%) vs. melhor solução da instância")
    ax.set_title("Gap por heurística" + ("" if n_max is None else f" (n ≤ {n_max})"))
    ax.grid(alpha=0.3)
    return ax


def plotar_wl_vs_n(df: pd.DataFrame, heuristicas: Optional[Sequence[str]] = None,
                   n_max: Optional[int] = None, ax=None):
    """WL médio (± desvio entre instâncias) versus n, uma linha por heurística."""
    import matplotlib.pyplot as plt
    d = df if n_max is None else df[df["n"] <= n_max]
    if heuristicas is None:
        heuristicas = [h for h in HEURISTICAS_PROPOSTAS + CONTROLES if h in set(d["heuristica"])]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    piv = d.groupby(["instancia", "heuristica"])["wl"].mean().reset_index().merge(
        d.drop_duplicates("instancia")[["instancia", "n"]], on="instancia")
    for h in heuristicas:
        g = piv[piv["heuristica"] == h].groupby("n")["wl"].agg(["mean", "std"]).fillna(0.0)
        ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", capsize=3, label=h)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (sinks)")
    ax.set_ylabel("WL médio (µm)")
    ax.set_title("Comprimento de fio versus n")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    return ax


def plotar_barras_hipotese2(res2: Dict[str, object], ax=None):
    """Barras agrupadas: gap % em relação a C1 por distribuição, uma barra por proposta."""
    import matplotlib.pyplot as plt
    t = res2["tabela"]
    if t.empty:
        raise ValueError(res2.get("motivo", "sem dados"))
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.5))
    larg = 0.25
    x = np.arange(2)
    for i, h in enumerate(t["heuristica"]):
        vals = [t.loc[t["heuristica"] == h, "gap_uniforme_pct"].iloc[0],
                t.loc[t["heuristica"] == h, "gap_clusterizada_pct"].iloc[0]]
        ax.bar(x + (i - 1) * larg, vals, larg, label=h)
    ax.set_xticks(x)
    ax.set_xticklabels(["uniforme", "clusterizada"])
    ax.set_ylabel("Redução de WL em relação a C1 (%)")
    ax.set_title("Gap das propostas vs. C1 por distribuição")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    return ax


def plotar_convergencia_cts(historicos: Dict[tuple, HistoricoCTS], nomes_instancias: Sequence[str],
                            semente: int, ax=None, heuristicas: Sequence[str] = ("H2", "H3")):
    """Melhor WL até o instante t (normalizado pelo melhor WL final de H2/H3 na
    instância), uma curva por instância e heurística."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.8))
    cores = {"H2": "tab:green", "H3": "tab:purple"}
    for inst in nomes_instancias:
        hs = {h: historicos.get((inst, h, semente)) for h in heuristicas}
        if any(v is None or not v.eventos for v in hs.values()):
            continue
        ref = min(v.eventos[-1][1] for v in hs.values())
        for h, v in hs.items():
            ts = [e[0] for e in v.eventos] + [v.tempo_total]
            ws = [e[1] / ref for e in v.eventos] + [v.eventos[-1][1] / ref]
            ax.step(ts, ws, where="post", color=cores.get(h), alpha=0.7,
                    label=h if inst == nomes_instancias[0] else None)
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("melhor WL / melhor WL final")
    ax.set_title("Convergência: H2 vs. H3")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax
