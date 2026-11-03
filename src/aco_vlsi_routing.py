# -*- coding: utf-8 -*-
"""
aco_vlsi_routing.py
===================

Implementação de **Otimização por Colônia de Formigas (Ant Colony
Optimization - ACO)** aplicada ao problema de **roteamento global (global
routing)** de circuitos VLSI, etapa que sucede o posicionamento (placement)
no fluxo de *Place and Route* (P&R).

Este módulo é a continuação natural de `sa_vlsi_placement.py`: ele recebe o
posicionamento das células (por exemplo, o resultado do Recozimento Simulado)
e decide por onde cada rede (net) da netlist passa na grade de roteamento.

Modelo do problema
------------------
- O chip é particionado em *G-cells* (aqui, um G-cell por sítio da grade de
  placement). Elas formam um **grafo de grade** G = (V, E): cada nó é um
  G-cell e cada aresta liga G-cells vizinhos (4-vizinhança).
- Cada aresta e possui uma **capacidade** c_e (número de trilhas de fio que
  podem cruzar a fronteira entre os dois G-cells).
- Cada rede n é um conjunto de pinos P_n (nós do grafo) e deve ser conectada
  por uma **árvore de Steiner** T_n contida em G.
- Uso da aresta: u_e = #{n : e ∈ T_n}.   Overflow: o_e = max(0, u_e - c_e).
- Objetivo: minimizar primeiro o overflow total e, em seguida, o
  comprimento total de fio (wirelength).

Algoritmos implementados (mesma interface, para comparação justa)
------------------------------------------------------------------
- `RoteadorACO`    : ACO (estilo MAX-MIN Ant System) dentro de um laço de
                     rip-up and reroute. **Método proposto.**
- `RoteadorMaze`   : roteamento por caminhos mínimos (Dijkstra) com custo de
                     congestionamento e custo histórico (negociação à la
                     PathFinder / NTHU-Route). **Baseline forte.**
- `RoteadorPadrao` : roteamento por padrões em L sobre uma MST dos pinos,
                     base do primeiro estágio de roteadores modernos
                     (FastRoute, CUGR, FastGR). **Baseline rápido.**

Os três compartilham o mesmo arcabouço de rip-up and reroute e a mesma função
de custo; a única diferença é *como uma árvore candidata é construída para uma
rede*. Assim, diferenças de resultado refletem o construtor de árvores.

Este módulo é pensado para ser importado em um notebook Jupyter.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Type

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1. Grade de roteamento (grafo de G-cells) e instância a partir do placement
# --------------------------------------------------------------------------- #

class GradeRoteamento:
    """
    Grafo de grade de G-cells.

    Os nós são numerados por linha: no = linha * n_colunas + coluna.
    As arestas são numeradas de 0 a E-1 (horizontais e verticais).

    Atributos úteis
    ---------------
    arestas : List[Tuple[int, int]]   extremos de cada aresta
    adj     : List[List[Tuple[int,int]]]  adj[no] = [(vizinho, id_aresta), ...]
    eid     : Dict[Tuple[int,int], int]   (u, v) -> id da aresta
    capacidade : np.ndarray (E,)      capacidade de cada aresta (editável,
                                      útil para simular bloqueios)
    linhas, colunas : np.ndarray (N,) coordenadas de cada nó
    """

    def __init__(self, n_linhas: int, n_colunas: int, capacidade: float = 2):
        self.n_linhas = n_linhas
        self.n_colunas = n_colunas
        self.n_nos = n_linhas * n_colunas

        self.arestas: List[Tuple[int, int]] = []
        self.adj: List[List[Tuple[int, int]]] = [[] for _ in range(self.n_nos)]
        self.eid: Dict[Tuple[int, int], int] = {}

        for l in range(n_linhas):
            for c in range(n_colunas):
                u = l * n_colunas + c
                if c + 1 < n_colunas:
                    self._add_aresta(u, u + 1)
                if l + 1 < n_linhas:
                    self._add_aresta(u, u + n_colunas)

        self.n_arestas = len(self.arestas)
        self.capacidade = np.full(self.n_arestas, float(capacidade))
        idx = np.arange(self.n_nos)
        self.linhas = idx // n_colunas
        self.colunas = idx % n_colunas

    def _add_aresta(self, u: int, v: int):
        e = len(self.arestas)
        self.arestas.append((u, v))
        self.adj[u].append((v, e))
        self.adj[v].append((u, e))
        self.eid[(u, v)] = e
        self.eid[(v, u)] = e

    def no(self, linha: int, coluna: int) -> int:
        return linha * self.n_colunas + coluna

    def coord(self, no: int) -> Tuple[int, int]:
        return int(self.linhas[no]), int(self.colunas[no])

    def centro(self, no: int) -> Tuple[float, float]:
        """Centro do G-cell em coordenadas (x, y) de desenho (y cresce para baixo)."""
        return self.colunas[no] + 0.5, self.linhas[no] + 0.5


def construir_pinos_das_redes(
    redes: Dict[str, List[int]],
    posicoes: Dict[int, Tuple[int, int]],
    grade: GradeRoteamento,
) -> Dict[str, List[int]]:
    """
    Converte as redes do placement ({net: [índices de componentes]}) em
    {net: [nós da grade]}, usando as posições (linha, coluna) de cada
    componente. Pinos que caem no mesmo G-cell são fundidos; redes que ficam
    com menos de 2 nós distintos não precisam de roteamento global e são
    descartadas.

    Parameters
    ----------
    redes : saída de `sa_vlsi_placement.construir_redes`
    posicoes : posição (linha, coluna) de cada componente, por exemplo
        `historico.melhor_posicoes_global` do SA.
    """
    pinos: Dict[str, List[int]] = {}
    for net, comps in redes.items():
        nos: List[int] = []
        for c in comps:
            linha, coluna = posicoes[c]
            no = grade.no(linha, coluna)
            if no not in nos:
                nos.append(no)
        if len(nos) >= 2:
            pinos[net] = nos
    return pinos


def hpwl_total(pinos_por_rede: Dict[str, List[int]], grade: GradeRoteamento) -> int:
    """Soma do HPWL (meio perímetro do retângulo envolvente) de todas as redes.
    É um **limite inferior** do comprimento de qualquer solução de roteamento,
    pois uma árvore de Steiner retilínea nunca é menor que o HPWL."""
    total = 0
    for nos in pinos_por_rede.values():
        ls = grade.linhas[nos]
        cs = grade.colunas[nos]
        total += int(ls.max() - ls.min() + cs.max() - cs.min())
    return total


def limite_inferior_overflow(pinos_por_rede: Dict[str, List[int]], grade: GradeRoteamento) -> float:
    """
    Limite inferior do overflow total, por **argumento de corte**.

    Considere o corte vertical entre as colunas j e j+1: ele é formado pelas
    arestas horizontais que o atravessam, e sua capacidade total é a soma das
    capacidades dessas arestas. Toda rede com pinos nos dois lados precisa usar
    ao menos uma delas, logo, se k redes cruzam o corte,

        soma_{e no corte} overflow_e  >=  max(0, k - capacidade_do_corte).

    Os cortes (verticais e horizontais) usam conjuntos de arestas disjuntos,
    então os limites se somam.

    Um segundo limite vem do **corte em torno de cada G-cell**: as k_v redes
    com pino no nó v precisam sair dele pelas arestas incidentes (um canto só
    tem 2). Como cada aresta é incidente a dois nós, a soma desses excessos
    conta cada overflow até duas vezes, e portanto

        overflow_total  >=  (1/2) · soma_v max(0, k_v - capacidade_incidente(v)).

    A função devolve o **maior** dos dois limites. O valor é válido para
    *qualquer* roteador, mas não é necessariamente atingível (costuma ser
    folgado). Se for > 0, parte do overflow é culpa do **placement**, e não
    do roteamento.
    """
    ls = {n: grade.linhas[nos] for n, nos in pinos_por_rede.items()}
    cs = {n: grade.colunas[nos] for n, nos in pinos_por_rede.items()}
    eid = grade.eid

    lb_corte = 0.0
    for j in range(grade.n_colunas - 1):  # cortes verticais
        k = sum(1 for n in cs if cs[n].min() <= j < cs[n].max())
        cap = sum(grade.capacidade[eid[(grade.no(l, j), grade.no(l, j + 1))]] for l in range(grade.n_linhas))
        lb_corte += max(0.0, k - cap)
    for i in range(grade.n_linhas - 1):  # cortes horizontais
        k = sum(1 for n in ls if ls[n].min() <= i < ls[n].max())
        cap = sum(grade.capacidade[eid[(grade.no(i, c), grade.no(i + 1, c))]] for c in range(grade.n_colunas))
        lb_corte += max(0.0, k - cap)

    redes_no = np.zeros(grade.n_nos)
    for nos in pinos_por_rede.values():
        for v in nos:
            redes_no[v] += 1
    lb_no = 0.0
    for v in range(grade.n_nos):
        cap_v = sum(grade.capacidade[e] for _, e in grade.adj[v])
        lb_no += max(0.0, redes_no[v] - cap_v)

    return max(lb_corte, lb_no / 2.0)


def gerar_instancia_exemplo(
    caminho_componentes: str = "componentes.csv",
    caminho_netlist: str = "netlist.csv",
    n_componentes: int = 24,
    semente: int = 7,
    sep: str = ";",
) -> None:
    """
    Gera uma instância sintética (componentes + netlist) de porte parecido com
    o do notebook de SA: 24 componentes, redes CLK (10 pinos) e RST (6 pinos),
    uma rede de 4 pinos, sete de 3 pinos e as demais de 2 pinos.
    Só use como substituto quando você não tiver os seus .csv.
    """
    rng = np.random.default_rng(semente)
    comps = [f"C{i + 1}" for i in range(n_componentes)]
    pd.DataFrame(
        {"Componente": comps, "Largura": 1.0, "Altura": 1.0}
    ).to_csv(caminho_componentes, sep=sep, index=False)

    tamanhos = {"CLK": 10, "RST": 6, "N2": 4}
    for n in ("N5", "N11", "N13", "N15", "N18", "N19", "N20"):
        tamanhos[n] = 3
    for k in range(1, 23):
        tamanhos.setdefault(f"N{k}", 2)

    linhas = []
    for net, tam in tamanhos.items():
        escolhidos = rng.choice(n_componentes, size=tam, replace=False)
        linhas += [(net, comps[i]) for i in escolhidos]

    # garante que todo componente participe de ao menos uma rede
    presentes = {c for _, c in linhas}
    for c in comps:
        if c not in presentes:
            linhas.append((f"N{int(rng.integers(1, 23))}", c))
    pd.DataFrame(linhas, columns=["Net", "Componente"]).to_csv(
        caminho_netlist, sep=sep, index=False
    )


def gerar_instancia_grande(
    n_componentes: int = 100,
    n_redes: int = 140,
    semente: int = 11,
    localidade: float = 0.35,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gera (df_componentes, df_netlist) de uma instância sintética maior, para
    testes de escala e de congestionamento. Os tamanhos das redes seguem uma
    distribuição realista (maioria com 2-3 pinos, cauda com redes grandes) e os
    pinos de cada rede são sorteados com certa **localidade** (componentes de
    índices próximos tendem a se conectar), imitando a estrutura de circuitos
    reais, em que o placement consegue agrupar células conectadas.

    Retorna DataFrames compatíveis com `sa_vlsi_placement.construir_redes`.
    """
    rng = np.random.default_rng(semente)
    comps = [f"C{i + 1}" for i in range(n_componentes)]
    df_comp = pd.DataFrame({"Componente": comps, "Largura": 1.0, "Altura": 1.0})

    tamanhos = rng.choice([2, 3, 4, 5, 8, 12], size=n_redes, p=[0.50, 0.25, 0.10, 0.07, 0.05, 0.03])
    linhas = []
    for k, tam in enumerate(tamanhos):
        centro = int(rng.integers(n_componentes))
        raio = max(int(tam * n_componentes * localidade / 4), tam + 1)
        candidatos = [(centro + d) % n_componentes for d in range(-raio, raio + 1)]
        escolhidos = rng.choice(candidatos, size=min(int(tam), len(set(candidatos))), replace=False)
        linhas += [(f"N{k + 1}", comps[int(i)]) for i in escolhidos]
    return df_comp, pd.DataFrame(linhas, columns=["Net", "Componente"])


# --------------------------------------------------------------------------- #
# 2. Utilitários sobre árvores (conjuntos de arestas)
# --------------------------------------------------------------------------- #

def normalizar_arvore(
    grade: GradeRoteamento, arestas: Set[int], pinos: Sequence[int]
) -> Set[int]:
    """
    Transforma um conjunto conexo de arestas em uma árvore: remove ciclos
    (árvore geradora por BFS a partir do primeiro pino) e poda folhas que não
    são pinos. Usado pelo roteador por padrões, cuja união de caminhos em L
    pode conter ciclos ou ramos inúteis.
    """
    vizinhos: Dict[int, List[Tuple[int, int]]] = {}
    for e in arestas:
        u, v = grade.arestas[e]
        vizinhos.setdefault(u, []).append((v, e))
        vizinhos.setdefault(v, []).append((u, e))

    raiz = pinos[0]
    visitados = {raiz}
    fila = [raiz]
    arvore: Set[int] = set()
    while fila:
        u = fila.pop(0)
        for v, e in vizinhos.get(u, []):
            if v not in visitados:
                visitados.add(v)
                arvore.add(e)
                fila.append(v)

    conj_pinos = set(pinos)
    grau: Dict[int, int] = {}
    for e in arvore:
        for x in grade.arestas[e]:
            grau[x] = grau.get(x, 0) + 1
    folhas = [x for x, g in grau.items() if g == 1 and x not in conj_pinos]
    while folhas:
        x = folhas.pop()
        for v, e in vizinhos.get(x, []):
            if e in arvore:
                arvore.discard(e)
                grau[x] -= 1
                grau[v] -= 1
                if grau[v] == 1 and v not in conj_pinos:
                    folhas.append(v)
    return arvore


def arvore_conecta_pinos(
    grade: GradeRoteamento, arvore: Set[int], pinos: Sequence[int]
) -> bool:
    """Verifica se `arvore` conecta todos os `pinos` (útil para validar solução)."""
    vizinhos: Dict[int, List[int]] = {}
    for e in arvore:
        u, v = grade.arestas[e]
        vizinhos.setdefault(u, []).append(v)
        vizinhos.setdefault(v, []).append(u)
    visitados = {pinos[0]}
    pilha = [pinos[0]]
    while pilha:
        u = pilha.pop()
        for v in vizinhos.get(u, []):
            if v not in visitados:
                visitados.add(v)
                pilha.append(v)
    return all(p in visitados for p in pinos)


# --------------------------------------------------------------------------- #
# 3. Histórico de execução
# --------------------------------------------------------------------------- #

@dataclass
class HistoricoRoteamento:
    nome: str = ""
    wirelength: List[float] = field(default_factory=list)
    overflow: List[float] = field(default_factory=list)
    n_arestas_overflow: List[int] = field(default_factory=list)
    congestionamento_max: List[float] = field(default_factory=list)
    tempo_acumulado: List[float] = field(default_factory=list)
    melhor_overflow: float = float("inf")
    melhor_wirelength: float = float("inf")
    melhor_iteracao: int = -1
    primeira_iteracao_viavel: Optional[int] = None
    tempo_total: float = 0.0


# --------------------------------------------------------------------------- #
# 4. Arcabouço comum: rip-up and reroute com custo de congestionamento
# --------------------------------------------------------------------------- #

class RoteadorGlobal:
    """
    Arcabouço de roteamento global por *rip-up and reroute* (RRR).

    A cada iteração (varredura), cada rede é desfeita (rip-up) e roteada de
    novo considerando o uso das demais. As subclasses decidem como construir
    árvores candidatas (`construir_candidatas`); a melhor candidata, avaliada
    pelo custo abaixo, substitui a rota atual (elitismo: a rota anterior
    também concorre).

    Custo por aresta, dado o uso das *outras* redes:

        custo_e = 1                                  (comprimento)
                + peso_suave * u_e / c_e             (pressão de congestionamento)
                + peso_overflow * max(0, u_e+1-c_e)  (overflow se esta rede usar e)
                + h_e                                (custo histórico, cresce nas
                                                      arestas que sofrem overflow)

    e o custo de uma árvore é a soma dos custos de suas arestas.

    Parameters
    ----------
    grade : GradeRoteamento
    redes_pinos : {net: [nós]} (ver `construir_pinos_das_redes`)
    peso_suave, peso_overflow, incremento_historico : coeficientes do custo
    margem_bbox : se não None, restringe a busca ao retângulo envolvente da
        rede expandido por essa margem (comum em roteadores reais).
    semente : semente do gerador de números aleatórios.
    """

    nome = "base"

    def __init__(
        self,
        grade: GradeRoteamento,
        redes_pinos: Dict[str, List[int]],
        peso_suave: float = 0.5,
        peso_overflow: float = 8.0,
        incremento_historico: float = 0.5,
        margem_bbox: Optional[int] = None,
        semente: Optional[int] = None,
    ):
        self.grade = grade
        self.nomes = list(redes_pinos.keys())
        self.pinos = [list(redes_pinos[n]) for n in self.nomes]
        self.n_redes = len(self.nomes)
        self.cap = np.asarray(grade.capacidade, dtype=float).copy()
        self.peso_suave = peso_suave
        self.peso_overflow = peso_overflow
        self.incremento_historico = incremento_historico
        self.margem_bbox = margem_bbox
        self.rng = np.random.default_rng(semente)

        self.uso = np.zeros(grade.n_arestas)
        self.hist_cong = np.zeros(grade.n_arestas)
        self.rotas: List[Optional[Set[int]]] = [None] * self.n_redes
        self.melhor_rotas: Optional[List[Set[int]]] = None

        self.hpwl_rede = []
        self.bbox: List[Optional[np.ndarray]] = []
        for nos in self.pinos:
            ls, cs = grade.linhas[nos], grade.colunas[nos]
            self.hpwl_rede.append(int(ls.max() - ls.min() + cs.max() - cs.min()))
            if margem_bbox is None:
                self.bbox.append(None)
            else:
                m = margem_bbox
                mascara = (
                    (grade.linhas >= ls.min() - m) & (grade.linhas <= ls.max() + m)
                    & (grade.colunas >= cs.min() - m) & (grade.colunas <= cs.max() + m)
                )
                self.bbox.append(mascara)

    # ----------------------------------------------------------------- #
    # Custos e métricas
    # ----------------------------------------------------------------- #
    def _custo_arestas(self) -> np.ndarray:
        excesso = np.maximum(0.0, self.uso + 1.0 - self.cap)
        pressao = self.uso / np.maximum(self.cap, 1.0)
        return 1.0 + self.peso_suave * pressao + self.peso_overflow * excesso + self.hist_cong

    @staticmethod
    def _custo_arvore(arvore: Set[int], custo_e: np.ndarray) -> float:
        idx = np.fromiter(arvore, dtype=int, count=len(arvore))
        return float(custo_e[idx].sum())

    def metricas(self) -> Dict[str, float]:
        excesso = np.maximum(0.0, self.uso - self.cap)
        return {
            "wirelength": float(self.uso.sum()),
            "overflow": float(excesso.sum()),
            "n_arestas_overflow": int((excesso > 0).sum()),
            "congestionamento_max": float((self.uso / np.maximum(self.cap, 1.0)).max()),
        }

    def _conflita(self, i: int) -> bool:
        r = self.rotas[i]
        if not r:
            return False
        idx = np.fromiter(r, dtype=int, count=len(r))
        return bool((self.uso[idx] > self.cap[idx]).any())

    # ----------------------------------------------------------------- #
    # Interface para subclasses
    # ----------------------------------------------------------------- #
    def construir_candidatas(self, i: int, custo_e: np.ndarray) -> List[Set[int]]:
        raise NotImplementedError

    def apos_escolha(self, i: int, arvore: Set[int], custo: float):
        """Gancho chamado após a escolha da rota de uma rede (ACO atualiza
        o feromônio aqui)."""

    # ----------------------------------------------------------------- #
    # Blocos reutilizáveis de busca em grafo
    # ----------------------------------------------------------------- #
    def _dijkstra(
        self,
        fontes: Sequence[int],
        eh_alvo: np.ndarray,
        custo_e: Sequence[float],
        bbox: Optional[np.ndarray] = None,
    ) -> Optional[List[int]]:
        """Dijkstra multi-fonte. Devolve o caminho (lista de nós) da fonte
        mais barata até o alvo mais barato, ou None se não houver caminho."""
        N = self.grade.n_nos
        adj = self.grade.adj
        dist = [math.inf] * N
        pai = [-1] * N
        heap = []
        for s in fontes:
            dist[s] = 0.0
            heap.append((0.0, s))
        heapq.heapify(heap)
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if eh_alvo[u]:
                caminho = [u]
                while pai[caminho[-1]] != -1:
                    caminho.append(pai[caminho[-1]])
                return caminho[::-1]
            for v, e in adj[u]:
                if bbox is not None and not bbox[v]:
                    continue
                nd = d + custo_e[e]
                if nd < dist[v]:
                    dist[v] = nd
                    pai[v] = u
                    heapq.heappush(heap, (nd, v))
        return None

    def _adicionar_caminho(
        self, caminho: List[int], arvore: Set[int], na_arvore: np.ndarray, nos: List[int]
    ):
        eid = self.grade.eid
        for u, v in zip(caminho[:-1], caminho[1:]):
            arvore.add(eid[(u, v)])
        for n in caminho:
            if not na_arvore[n]:
                na_arvore[n] = True
                nos.append(n)

    # ----------------------------------------------------------------- #
    # Laço de rip-up and reroute
    # ----------------------------------------------------------------- #
    def _ordem(self, it: int, apenas_conflitantes: bool) -> List[int]:
        if it == 0 or any(r is None for r in self.rotas):
            # primeira varredura: redes curtas primeiro (menos flexíveis)
            return sorted(range(self.n_redes), key=lambda i: (self.hpwl_rede[i], self.rng.random()))
        conf = [i for i in range(self.n_redes) if self._conflita(i)]
        outras = [i for i in range(self.n_redes) if i not in set(conf)]
        ordem = [int(i) for i in self.rng.permutation(conf)] if conf else []
        if not apenas_conflitantes:
            ordem += [int(i) for i in self.rng.permutation(outras)] if outras else []
        return ordem

    def _reroteia(self, i: int):
        atual = self.rotas[i]
        if atual is not None:
            self.uso[np.fromiter(atual, dtype=int, count=len(atual))] -= 1

        custo_e = self._custo_arestas()
        candidatas = self.construir_candidatas(i, custo_e)
        if atual is not None:
            candidatas = candidatas + [atual]  # elitismo: a rota anterior concorre

        custos = [self._custo_arvore(t, custo_e) for t in candidatas]
        k = int(np.argmin(custos))  # empate favorece a candidata nova
        escolhida = candidatas[k]

        self.rotas[i] = escolhida
        self.uso[np.fromiter(escolhida, dtype=int, count=len(escolhida))] += 1
        self.apos_escolha(i, escolhida, custos[k])

    def executar(
        self,
        n_iteracoes: int = 100,
        paciencia: Optional[int] = None,
        apenas_conflitantes: bool = False,
        callback: Optional[Callable[[int, "RoteadorGlobal", HistoricoRoteamento], None]] = None,
    ) -> HistoricoRoteamento:
        """
        Executa até `n_iteracoes` varreduras de rip-up and reroute.

        Parameters
        ----------
        paciencia : para após esse número de iterações sem melhora do melhor
            par (overflow, wirelength). None desativa.
        apenas_conflitantes : se True, a partir da 2ª iteração só refaz redes
            que usam arestas em overflow (estilo NTHU-Route / PathFinder).
        callback : função (iteração, roteador, histórico) chamada a cada iteração.

        Ao final, o roteador mantém a **melhor** solução encontrada
        (menor overflow; em empate, menor wirelength).
        """
        h = HistoricoRoteamento(nome=self.nome)
        t0 = time.perf_counter()
        melhor_chave = (math.inf, math.inf)
        sem_melhora = 0

        for it in range(n_iteracoes):
            ordem = self._ordem(it, apenas_conflitantes)
            for i in ordem:
                self._reroteia(i)

            m = self.metricas()
            h.wirelength.append(m["wirelength"])
            h.overflow.append(m["overflow"])
            h.n_arestas_overflow.append(m["n_arestas_overflow"])
            h.congestionamento_max.append(m["congestionamento_max"])
            h.tempo_acumulado.append(time.perf_counter() - t0)

            chave = (m["overflow"], m["wirelength"])
            if chave < melhor_chave:
                melhor_chave = chave
                self.melhor_rotas = [set(r) for r in self.rotas]
                h.melhor_overflow, h.melhor_wirelength = chave
                h.melhor_iteracao = it
                sem_melhora = 0
            else:
                sem_melhora += 1
            if m["overflow"] == 0 and h.primeira_iteracao_viavel is None:
                h.primeira_iteracao_viavel = it

            # custo histórico: arestas congestionadas ficam mais caras (negociação)
            self.hist_cong[self.uso > self.cap] += self.incremento_historico

            if callback is not None:
                callback(it, self, h)
            if not ordem:
                break
            if paciencia is not None and sem_melhora >= paciencia:
                break

        h.tempo_total = time.perf_counter() - t0
        self._restaurar_melhor()
        return h

    def _restaurar_melhor(self):
        if self.melhor_rotas is None:
            return
        self.rotas = [set(r) for r in self.melhor_rotas]
        self.uso[:] = 0
        for r in self.rotas:
            self.uso[np.fromiter(r, dtype=int, count=len(r))] += 1


# --------------------------------------------------------------------------- #
# 5. Baseline 1: roteamento por padrões em L
# --------------------------------------------------------------------------- #

class RoteadorPadrao(RoteadorGlobal):
    """
    Constrói, para cada rede, uma árvore geradora mínima (MST de Prim, distância
    de Manhattan) sobre os pinos e roteia cada aresta da MST com um dos dois
    padrões em L (horizontal-primeiro ou vertical-primeiro), escolhendo o de
    menor custo. É a base do primeiro estágio de roteadores modernos por ser
    muito rápido, porém só explora 2 formas por conexão.
    """

    nome = "Padrão em L"

    def _mst(self, nos: List[int]) -> List[Tuple[int, int]]:
        g = self.grade
        n = len(nos)
        em_arvore = [False] * n
        melhor = [math.inf] * n
        pai = [-1] * n
        melhor[0] = 0
        arestas = []
        for _ in range(n):
            u = min((k for k in range(n) if not em_arvore[k]), key=lambda k: melhor[k])
            em_arvore[u] = True
            if pai[u] >= 0:
                arestas.append((nos[pai[u]], nos[u]))
            for v in range(n):
                if not em_arvore[v]:
                    d = abs(g.linhas[nos[u]] - g.linhas[nos[v]]) + abs(g.colunas[nos[u]] - g.colunas[nos[v]])
                    if d < melhor[v]:
                        melhor[v] = d
                        pai[v] = u
        return arestas

    def _caminho_L(self, u: int, v: int, horizontal_primeiro: bool) -> List[int]:
        g = self.grade
        (l1, c1), (l2, c2) = g.coord(u), g.coord(v)
        caminho = [u]
        l, c = l1, c1

        def anda_col():
            nonlocal c
            passo = 1 if c2 > c else -1
            while c != c2:
                c += passo
                caminho.append(g.no(l, c))

        def anda_lin():
            nonlocal l
            passo = 1 if l2 > l else -1
            while l != l2:
                l += passo
                caminho.append(g.no(l, c))

        if horizontal_primeiro:
            anda_col()
            anda_lin()
        else:
            anda_lin()
            anda_col()
        return caminho

    def construir_candidatas(self, i: int, custo_e: np.ndarray) -> List[Set[int]]:
        eid = self.grade.eid
        arvore: Set[int] = set()
        for u, v in self._mst(self.pinos[i]):
            melhor_c, melhor_arestas = math.inf, None
            for hf in (True, False):
                cam = self._caminho_L(u, v, hf)
                arestas = {eid[(a, b)] for a, b in zip(cam[:-1], cam[1:])}
                c = sum(custo_e[e] for e in arestas if e not in arvore)
                if c < melhor_c:
                    melhor_c, melhor_arestas = c, arestas
            arvore |= melhor_arestas
        return [normalizar_arvore(self.grade, arvore, self.pinos[i])]


# --------------------------------------------------------------------------- #
# 6. Baseline 2: Maze routing (Dijkstra) com negociação de congestionamento
# --------------------------------------------------------------------------- #

class RoteadorMaze(RoteadorGlobal):
    """
    Cada rede é roteada incrementalmente: começa com um pino e, a cada passo,
    um Dijkstra multi-fonte a partir de **todos** os nós da árvore até o pino
    ainda não conectado mais barato (heurística de Prim / "tree growing"; é uma
    2-aproximação da árvore de Steiner de menor custo). Combinado ao custo
    histórico e ao rip-up and reroute, forma o esquema de *negociação de
    congestionamento* do PathFinder, núcleo de vários roteadores globais.
    """

    nome = "Maze (Dijkstra + histórico)"

    def construir_candidatas(self, i: int, custo_e: np.ndarray) -> List[Set[int]]:
        N = self.grade.n_nos
        custo_l = custo_e.tolist()
        pinos = self.pinos[i]
        na_arvore = np.zeros(N, dtype=bool)
        na_arvore[pinos[0]] = True
        nos = [pinos[0]]
        arvore: Set[int] = set()
        restantes = set(pinos[1:])
        while restantes:
            alvo = np.zeros(N, dtype=bool)
            alvo[list(restantes)] = True
            caminho = self._dijkstra(nos, alvo, custo_l, self.bbox[i])
            if caminho is None:  # bbox restritivo demais
                caminho = self._dijkstra(nos, alvo, custo_l, None)
            self._adicionar_caminho(caminho, arvore, na_arvore, nos)
            restantes = {p for p in restantes if not na_arvore[p]}
        return [arvore]


# --------------------------------------------------------------------------- #
# 7. Método proposto: ACO (MAX-MIN Ant System) para roteamento global
# --------------------------------------------------------------------------- #

class RoteadorACO(RoteadorGlobal):
    """
    Roteamento global por Otimização por Colônia de Formigas.

    Cada rede possui sua própria **colônia** e sua própria matriz de feromônio
    τ_n(e) sobre as arestas da grade. As colônias interagem apenas através do
    congestionamento (uso das arestas), que entra na informação heurística.

    Construção da árvore por uma formiga
    ------------------------------------
    1. A árvore começa em um pino sorteado.
    2. Escolhe-se o próximo pino a conectar, com probabilidade ∝ 1/(1+d)^2,
       em que d é a distância de Manhattan do pino à árvore.
    3. Uma formiga parte **do pino** e faz um passeio aleatório até **tocar a
       árvore**; em cada nó escolhe o vizinho j com probabilidade

            p(j) ∝ τ(e_ij)^α · (1/custo(e_ij))^β · (1/(1+d_j))^γ

       em que d_j é a distância de Manhattan de j à árvore (informação de
       direção). Ciclos do passeio são apagados (*loop erasure*), o que
       garante um caminho simples. O caminho é incorporado à árvore, e os
       pontos de contato tornam-se pontos de Steiner naturalmente.
    4. Repete-se até conectar todos os pinos.

    Atualização do feromônio (estilo MAX-MIN Ant System)
    ----------------------------------------------------
    Após a escolha da rota da rede (melhor entre as `n_formigas` candidatas e
    a rota anterior):

        τ ← (1-ρ) τ            (evaporação, em todas as arestas da rede)
        τ(e) += ρ · HPWL/custo (depósito, só nas arestas da árvore escolhida)
        τ ← clip(τ, τ_min, τ_max),   τ_max = 1,  τ_min = τ_max/razao_tau

    Parameters
    ----------
    n_formigas : formigas (árvores candidatas) por rede e por iteração.
    alpha, beta, gamma : pesos de feromônio, congestionamento e direção.
    rho : taxa de evaporação (0 < rho <= 1).
    razao_tau : razão τ_max/τ_min do MMAS; limita a perda de exploração.
    max_passos_fator : limite de passos do passeio = fator * (d+1); se
        excedido, usa-se Dijkstra como contingência.
    """

    nome = "ACO (MMAS)"

    def __init__(
        self,
        grade: GradeRoteamento,
        redes_pinos: Dict[str, List[int]],
        n_formigas: int = 8,
        alpha: float = 1.0,
        beta: float = 2.0,
        gamma: float = 1.0,
        rho: float = 0.1,
        razao_tau: float = 50.0,
        margem_bbox: Optional[int] = 2,
        max_passos_fator: int = 8,
        **kwargs,
    ):
        super().__init__(grade, redes_pinos, margem_bbox=margem_bbox, **kwargs)
        self.n_formigas = n_formigas
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.tau_max = 1.0
        self.tau_min = self.tau_max / razao_tau
        self.max_passos_fator = max_passos_fator
        # feromônio por rede e por aresta (denso; adequado a grades pequenas/médias)
        self.tau = np.full((self.n_redes, grade.n_arestas), self.tau_max)

    # ----------------------------------------------------------------- #
    def _dist_ate_arvore(self, nos: List[int]):
        g = self.grade
        rt, ct = g.linhas[nos], g.colunas[nos]
        d = np.abs(g.linhas[:, None] - rt[None, :]) + np.abs(g.colunas[:, None] - ct[None, :])
        d = d.min(axis=1)
        fator = (1.0 / (1.0 + d)) ** self.gamma
        return d.tolist(), fator.tolist()

    def _caminhada(
        self,
        origem: int,
        na_arvore: np.ndarray,
        dist: List[float],
        fator_dir: List[float],
        peso_aresta: List[float],
        custo_l: List[float],
        bbox: Optional[np.ndarray],
    ) -> List[int]:
        """Passeio aleatório com apagamento de ciclos, da `origem` até tocar a árvore."""
        adj = self.grade.adj
        caminho = [origem]
        pos = {origem: 0}
        atual = origem
        max_passos = int(self.max_passos_fator * (dist[origem] + 1)) + 10

        for _ in range(max_passos):
            vizinhos, pesos = [], []
            for v, e in adj[atual]:
                if bbox is not None and not bbox[v]:
                    continue
                vizinhos.append(v)
                pesos.append(peso_aresta[e] * fator_dir[v])
            r = self.rng.random() * sum(pesos)
            acc = 0.0
            escolhido = vizinhos[-1]
            for v, w in zip(vizinhos, pesos):
                acc += w
                if r <= acc:
                    escolhido = v
                    break

            if escolhido in pos:  # apaga o ciclo formado
                k = pos[escolhido]
                for n in caminho[k + 1:]:
                    del pos[n]
                caminho = caminho[:k + 1]
            else:
                pos[escolhido] = len(caminho)
                caminho.append(escolhido)
            atual = escolhido
            if na_arvore[atual]:
                return caminho

        # contingência: a formiga "se perdeu"; completa com o caminho mínimo
        resto = self._dijkstra([atual], na_arvore, custo_l, bbox)
        if resto is None:
            resto = self._dijkstra([atual], na_arvore, custo_l, None)
        return self._sem_ciclos(caminho[:-1] + resto)

    @staticmethod
    def _sem_ciclos(caminho: List[int]) -> List[int]:
        pos, saida = {}, []
        for n in caminho:
            if n in pos:
                k = pos[n]
                for x in saida[k + 1:]:
                    del pos[x]
                saida = saida[:k + 1]
            else:
                pos[n] = len(saida)
                saida.append(n)
        return saida

    def _formiga_constroi_arvore(
        self, i: int, peso_aresta: List[float], custo_l: List[float]
    ) -> Set[int]:
        N = self.grade.n_nos
        pinos = self.pinos[i]
        bbox = self.bbox[i]

        inicio = pinos[int(self.rng.integers(len(pinos)))]
        na_arvore = np.zeros(N, dtype=bool)
        na_arvore[inicio] = True
        nos = [inicio]
        arvore: Set[int] = set()
        restantes = [p for p in pinos if p != inicio]
        dist, fator_dir = self._dist_ate_arvore(nos)

        while restantes:
            w = np.array([1.0 / (1.0 + dist[p]) ** 2 for p in restantes])
            k = int(self.rng.choice(len(restantes), p=w / w.sum()))
            pino = restantes.pop(k)

            caminho = self._caminhada(pino, na_arvore, dist, fator_dir, peso_aresta, custo_l, bbox)
            self._adicionar_caminho(caminho, arvore, na_arvore, nos)
            restantes = [p for p in restantes if not na_arvore[p]]
            if restantes:
                dist, fator_dir = self._dist_ate_arvore(nos)
        return arvore

    def construir_candidatas(self, i: int, custo_e: np.ndarray) -> List[Set[int]]:
        eta = (1.0 / custo_e) ** self.beta
        peso_aresta = ((self.tau[i] ** self.alpha) * eta).tolist()
        custo_l = custo_e.tolist()
        return [self._formiga_constroi_arvore(i, peso_aresta, custo_l) for _ in range(self.n_formigas)]

    def apos_escolha(self, i: int, arvore: Set[int], custo: float):
        tau = self.tau[i]
        tau *= (1.0 - self.rho)
        idx = np.fromiter(arvore, dtype=int, count=len(arvore))
        tau[idx] += self.rho * min(1.0, self.hpwl_rede[i] / max(custo, 1e-9))
        np.clip(tau, self.tau_min, self.tau_max, out=tau)


# --------------------------------------------------------------------------- #
# 8. Experimentos: várias sementes / vários métodos
# --------------------------------------------------------------------------- #

def executar_metodo(
    classe: Type[RoteadorGlobal],
    grade: GradeRoteamento,
    redes_pinos: Dict[str, List[int]],
    n_iteracoes: int = 100,
    semente: Optional[int] = None,
    paciencia: Optional[int] = None,
    apenas_conflitantes: bool = False,
    **kwargs,
) -> Tuple[RoteadorGlobal, HistoricoRoteamento]:
    """Cria o roteador, executa e devolve (roteador, histórico)."""
    r = classe(grade, redes_pinos, semente=semente, **kwargs)
    h = r.executar(n_iteracoes=n_iteracoes, paciencia=paciencia, apenas_conflitantes=apenas_conflitantes)
    return r, h


def comparar_metodos(
    grade: GradeRoteamento,
    redes_pinos: Dict[str, List[int]],
    metodos: Dict[str, Tuple[Type[RoteadorGlobal], dict]],
    n_iteracoes: int = 100,
    sementes: Sequence[int] = tuple(range(10)),
    **kwargs_executar,
) -> pd.DataFrame:
    """
    Executa cada método com cada semente e devolve um DataFrame com o resultado
    final (melhor solução) de cada execução. Métodos determinísticos (Maze,
    Padrão em L) também são repetidos para manter o formato uniforme; a
    única fonte de variação neles é a ordem aleatória das redes.

    metodos : {rótulo: (classe, kwargs_do_construtor)}
    """
    linhas = []
    limite_inferior = hpwl_total(redes_pinos, grade)
    for rotulo, (classe, kw) in metodos.items():
        for s in sementes:
            r, h = executar_metodo(classe, grade, redes_pinos, n_iteracoes, semente=s, **kw, **kwargs_executar)
            m = r.metricas()
            linhas.append({
                "metodo": rotulo,
                "semente": s,
                "overflow": m["overflow"],
                "wirelength": m["wirelength"],
                "wl_sobre_hpwl": m["wirelength"] / limite_inferior,
                "congestionamento_max": m["congestionamento_max"],
                "iter_primeira_viavel": h.primeira_iteracao_viavel if h.primeira_iteracao_viavel is not None else np.nan,
                "tempo_s": h.tempo_total,
            })
    return pd.DataFrame(linhas)


def resumir_comparacao(df: pd.DataFrame) -> pd.DataFrame:
    """Média ± desvio por método, mais a taxa de execuções sem overflow."""
    g = df.groupby("metodo", sort=False)
    resumo = pd.DataFrame({
        "overflow_medio": g["overflow"].mean(),
        "overflow_dp": g["overflow"].std(ddof=0),
        "taxa_viavel": g["overflow"].apply(lambda s: float((s == 0).mean())),
        "wl_medio": g["wirelength"].mean(),
        "wl_dp": g["wirelength"].std(ddof=0),
        "wl_sobre_hpwl": g["wl_sobre_hpwl"].mean(),
        "iter_1a_viavel": g["iter_primeira_viavel"].mean(),
        "tempo_medio_s": g["tempo_s"].mean(),
    })
    return resumo.round(3)


# --------------------------------------------------------------------------- #
# 9. Visualização (matplotlib é importado dentro das funções)
# --------------------------------------------------------------------------- #

def plotar_roteamento(
    df_componentes: pd.DataFrame,
    grade: GradeRoteamento,
    posicoes: Dict[int, Tuple[int, int]],
    uso: np.ndarray,
    rotas: Optional[List[Set[int]]] = None,
    destacar: Optional[int] = None,
    ax=None,
    titulo: str = "Roteamento global",
    mostrar_colorbar: bool = True,
):
    """
    Desenha a grade de G-cells com as células posicionadas e as arestas
    coloridas pela **utilização** u_e / c_e (verde = folgada, vermelho =
    saturada/overflow). A espessura é proporcional a u_e. Se `destacar` (índice
    de uma rede) e `rotas` forem informados, a árvore dessa rede é realçada.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    norma = colors.Normalize(vmin=0.0, vmax=1.5)
    cmap = plt.cm.RdYlGn_r

    for l in range(grade.n_linhas + 1):
        ax.axhline(l, color="lightgray", linewidth=0.4, zorder=0)
    for c in range(grade.n_colunas + 1):
        ax.axvline(c, color="lightgray", linewidth=0.4, zorder=0)

    for e, (u, v) in enumerate(grade.arestas):
        if uso[e] <= 0:
            continue
        (x1, y1), (x2, y2) = grade.centro(u), grade.centro(v)
        razao = uso[e] / max(grade.capacidade[e], 1.0)
        ax.plot([x1, x2], [y1, y2], color=cmap(norma(razao)),
                linewidth=1.0 + 2.0 * uso[e], solid_capstyle="round", zorder=1,
                alpha=0.35 if destacar is not None else 0.95)

    if destacar is not None and rotas is not None:
        for e in rotas[destacar]:
            u, v = grade.arestas[e]
            (x1, y1), (x2, y2) = grade.centro(u), grade.centro(v)
            ax.plot([x1, x2], [y1, y2], color="tab:blue", linewidth=4,
                    solid_capstyle="round", zorder=2)

    for idx, (linha, coluna) in posicoes.items():
        ax.add_patch(plt.Rectangle((coluna + 0.2, linha + 0.2), 0.6, 0.6,
                                   facecolor="dimgray", edgecolor="black", zorder=3))
        ax.annotate(str(df_componentes.loc[idx, "Componente"]), (coluna + 0.5, linha + 0.5),
                    ha="center", va="center", fontsize=6, color="white", zorder=4)

    ax.set_xlim(0, grade.n_colunas)
    ax.set_ylim(0, grade.n_linhas)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(titulo)
    ax.set_xlabel("Coluna")
    ax.set_ylabel("Linha")
    if mostrar_colorbar:
        sm = cm.ScalarMappable(norm=norma, cmap=cmap)
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="utilização u/c")
    return ax


def plotar_convergencia_roteamento(historicos: Dict[str, HistoricoRoteamento], eixos=None):
    """Curvas de overflow e de wirelength por iteração para vários métodos."""
    import matplotlib.pyplot as plt

    if eixos is None:
        _, eixos = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1, ax2 = eixos
    for nome, h in historicos.items():
        its = np.arange(1, len(h.overflow) + 1)
        ax1.plot(its, h.overflow, label=nome, linewidth=1.3)
        ax2.plot(its, h.wirelength, label=nome, linewidth=1.3)
    ax1.set_title("Overflow total por iteração")
    ax1.set_xlabel("Iteração (varredura de rip-up and reroute)")
    ax1.set_ylabel("Overflow total")
    ax2.set_title("Wirelength total por iteração")
    ax2.set_xlabel("Iteração (varredura de rip-up and reroute)")
    ax2.set_ylabel("Wirelength (nº de arestas)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    return eixos


def plotar_feromonio(roteador: RoteadorACO, i: int, df_componentes: pd.DataFrame,
                     posicoes: Dict[int, Tuple[int, int]], ax=None):
    """Mostra o feromônio τ_n(e) aprendido pela colônia da rede `i`
    (espessura e cor ∝ τ) e a árvore final da rede em tracejado."""
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    g = roteador.grade
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.5))
    tau = roteador.tau[i]
    # a escala usa a faixa *observada* de τ: o depósito é no máximo HPWL/custo < 1,
    # então τ_max nominal raramente é atingido e uma escala fixa esconderia o contraste
    t_lo, t_hi = float(tau.min()), float(tau.max())
    norma = colors.Normalize(vmin=t_lo, vmax=max(t_hi, t_lo + 1e-12))
    cmap = plt.cm.viridis
    for l in range(g.n_linhas + 1):
        ax.axhline(l, color="lightgray", linewidth=0.4, zorder=0)
    for c in range(g.n_colunas + 1):
        ax.axvline(c, color="lightgray", linewidth=0.4, zorder=0)
    for e, (u, v) in enumerate(g.arestas):
        (x1, y1), (x2, y2) = g.centro(u), g.centro(v)
        ax.plot([x1, x2], [y1, y2], color=cmap(norma(tau[e])),
                linewidth=0.8 + 5.0 * (tau[e] - t_lo) / max(t_hi - t_lo, 1e-12),
                solid_capstyle="round", zorder=1)
    if roteador.rotas[i] is not None:
        for e in roteador.rotas[i]:
            u, v = g.arestas[e]
            (x1, y1), (x2, y2) = g.centro(u), g.centro(v)
            ax.plot([x1, x2], [y1, y2], color="red", linewidth=1.2, linestyle="--", zorder=2)
    pinos = set(roteador.pinos[i])
    for no in range(g.n_nos):
        x, y = g.centro(no)
        if no in pinos:
            ax.scatter([x], [y], s=120, color="red", edgecolor="black", zorder=3)
    ax.set_xlim(0, g.n_colunas)
    ax.set_ylim(0, g.n_linhas)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"Feromônio da rede {roteador.nomes[i]} (pinos em vermelho)\n"
                 f"escala = faixa observada de τ: [{t_lo:.3f}, {t_hi:.3f}]")
    plt.colorbar(cm.ScalarMappable(norm=norma, cmap=cmap), ax=ax, fraction=0.046, pad=0.04, label="τ")
    return ax
