# -*- coding: utf-8 -*-
"""
sa_vlsi_placement.py
=====================

Implementação do algoritmo de **Recozimento Simulado (Simulated Annealing -
SA)** aplicado ao problema de **posicionamento (placement) de células em
projetos de circuitos integrados (VLSI)**, uma das etapas centrais da fase
de *Place and Route* (P&R) do fluxo de projeto físico de chips.

Contexto do problema
---------------------
Durante o *placement*, cada célula (porta lógica, flip-flop, etc.) do
circuito precisa ser posicionada em um sítio de uma grade (representando o
substrato do chip), de forma a minimizar o comprimento total de fiação
(*wirelength*) necessário para conectar as células de acordo com a
*netlist* do circuito. Um posicionamento ruim aumenta a área de roteamento,
o atraso de propagação (delay) e o consumo de energia do chip, tornando
essa uma etapa crítica de otimização.

Este módulo modela o problema da seguinte forma:

- Cada célula do circuito ocupa um sítio (linha, coluna) em uma grade.
- A qualidade de um posicionamento é medida pelo **HPWL (Half-Perimeter
  Wire Length)** de cada rede (net): a soma, para cada net, de
  (largura do retângulo delimitador dos pinos) + (altura do retângulo
  delimitador dos pinos). Essa é a métrica de estimativa de fiação mais
  usada na literatura de posicionamento de circuitos VLSI.
- O algoritmo de Recozimento Simulado explora o espaço de soluções trocando
  a posição de pares de células (ou movendo uma célula para um sítio vazio),
  aceitando pioras de custo com uma probabilidade que decresce conforme a
  "temperatura" do sistema diminui — permitindo escapar de mínimos locais no
  início da busca e refinando a solução ao final.

Dados de entrada (arquivos .csv)
---------------------------------
1. `componentes.csv` — lista das células do circuito:

    Componente;Largura;Altura
    C1;1;1
    C2;1;1
    ...

2. `netlist.csv` — conectividade do circuito (uma linha por par
   net/componente; uma net pode ter 2 ou mais componentes):

    Net;Componente
    N1;C1
    N1;C4
    N1;C7
    N2;C2
    N2;C3
    ...

Este módulo é pensado para ser importado dentro de um notebook Jupyter, que
usa suas funções/classes para carregar os dados, rodar o algoritmo e plotar
os resultados (posicionamento final e curva de convergência).

Autor: gerado com Claude
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1. Leitura e preparação dos dados
# --------------------------------------------------------------------------- #

def carregar_componentes(caminho_csv: str, sep: str = ";") -> pd.DataFrame:
    """
    Carrega o arquivo .csv com as células (componentes) do circuito.

    Parameters
    ----------
    caminho_csv : str
        Caminho para o arquivo .csv com colunas "Componente;Largura;Altura".
    sep : str
        Separador usado no .csv (padrão ";").

    Returns
    -------
    pd.DataFrame
        DataFrame com as colunas ["Componente", "Largura", "Altura"].
    """
    df = pd.read_csv(caminho_csv, sep=sep, encoding="utf-8-sig")
    colunas_esperadas = {"Componente", "Largura", "Altura"}
    if not colunas_esperadas.issubset(set(df.columns)):
        raise ValueError(
            f"O arquivo deve conter as colunas {colunas_esperadas}, "
            f"mas contém {list(df.columns)}"
        )
    df["Largura"] = df["Largura"].astype(float)
    df["Altura"] = df["Altura"].astype(float)
    return df.reset_index(drop=True)


def carregar_netlist(caminho_csv: str, sep: str = ";") -> pd.DataFrame:
    """
    Carrega o arquivo .csv com a netlist (conectividade) do circuito.

    Parameters
    ----------
    caminho_csv : str
        Caminho para o arquivo .csv com colunas "Net;Componente".
    sep : str
        Separador usado no .csv (padrão ";").

    Returns
    -------
    pd.DataFrame
        DataFrame com as colunas ["Net", "Componente"].
    """
    df = pd.read_csv(caminho_csv, sep=sep, encoding="utf-8-sig")
    colunas_esperadas = {"Net", "Componente"}
    if not colunas_esperadas.issubset(set(df.columns)):
        raise ValueError(
            f"O arquivo deve conter as colunas {colunas_esperadas}, "
            f"mas contém {list(df.columns)}"
        )
    return df.reset_index(drop=True)


def construir_redes(
    df_componentes: pd.DataFrame, df_netlist: pd.DataFrame
) -> Dict[str, List[int]]:
    """
    Converte a netlist (nomes de componentes) em um dicionário
    {nome_da_net: [índices dos componentes]}, usando os índices do DataFrame
    de componentes (mais eficiente para o algoritmo).

    Raises
    ------
    ValueError
        Se a netlist referenciar um componente que não existe em
        `df_componentes`.
    """
    nome_para_indice = {
        nome: idx for idx, nome in enumerate(df_componentes["Componente"])
    }

    redes: Dict[str, List[int]] = {}
    for _, linha in df_netlist.iterrows():
        net = linha["Net"]
        comp = linha["Componente"]
        if comp not in nome_para_indice:
            raise ValueError(
                f"Componente '{comp}' referenciado na netlist não existe "
                f"na lista de componentes."
            )
        redes.setdefault(net, []).append(nome_para_indice[comp])

    # Remove nets degeneradas (com um único pino, sem custo de fiação)
    redes = {net: comps for net, comps in redes.items() if len(comps) >= 2}
    return redes


def dimensoes_grade_sugerida(n_componentes: int, folga: float = 1.2) -> Tuple[int, int]:
    """
    Sugere um número de linhas/colunas para a grade de posicionamento,
    deixando uma folga (sítios vazios) em relação ao número de componentes,
    o que é usual em posicionamento VLSI para viabilizar o roteamento
    (canais de roteamento) e permitir movimentos de troca no SA.

    Parameters
    ----------
    n_componentes : int
        Número de componentes a posicionar.
    folga : float
        Fator multiplicativo de folga sobre o número mínimo de sítios
        (ex.: 1.2 = 20% de sítios vazios).

    Returns
    -------
    Tuple[int, int]
        (n_linhas, n_colunas)
    """
    n_sitios = max(1, math.ceil(n_componentes * folga))
    lado = math.ceil(math.sqrt(n_sitios))
    n_linhas = lado
    n_colunas = math.ceil(n_sitios / lado)
    return n_linhas, n_colunas


# --------------------------------------------------------------------------- #
# 2. Estrutura para guardar o histórico de execução (útil para gráficos)
# --------------------------------------------------------------------------- #

@dataclass
class HistoricoSA:
    custo_por_iteracao: List[float] = field(default_factory=list)
    temperatura_por_iteracao: List[float] = field(default_factory=list)
    taxa_aceitacao_por_temperatura: List[float] = field(default_factory=list)
    melhor_custo_global: float = float("inf")
    melhor_posicoes_global: Optional[Dict[int, Tuple[int, int]]] = None


# --------------------------------------------------------------------------- #
# 3. Algoritmo de Recozimento Simulado (Simulated Annealing) para placement
# --------------------------------------------------------------------------- #

class SAPlacement:
    """
    Recozimento Simulado aplicado ao posicionamento (placement) de células
    de um circuito VLSI em uma grade, minimizando o HPWL total da netlist.

    Parameters
    ----------
    n_componentes : int
        Número de componentes (células) a posicionar.
    redes : Dict[str, List[int]]
        Dicionário {nome_da_net: [índices dos componentes conectados]}.
    n_linhas, n_colunas : int
        Dimensões da grade de posicionamento. O número de sítios
        (n_linhas * n_colunas) deve ser >= n_componentes; sítios excedentes
        ficam vazios (representando espaço livre para roteamento).
    temperatura_inicial : float
        Temperatura inicial do recozimento (deve ser alta o suficiente para
        aceitar praticamente qualquer movimento no início).
    temperatura_final : float
        Temperatura mínima em que o algoritmo para (critério de parada).
    taxa_resfriamento : float
        Fator geométrico de resfriamento aplicado a cada etapa
        (0 < taxa_resfriamento < 1); T_novo = T_atual * taxa_resfriamento.
    iteracoes_por_temperatura : int
        Número de movimentos tentados em cada patamar de temperatura antes
        de resfriar (regra de Metropolis).
    semente : Optional[int]
        Semente para reprodutibilidade dos resultados aleatórios.
    """

    def __init__(
        self,
        n_componentes: int,
        redes: Dict[str, List[int]],
        n_linhas: int,
        n_colunas: int,
        temperatura_inicial: float = 500.0,
        temperatura_final: float = 0.5,
        taxa_resfriamento: float = 0.95,
        iteracoes_por_temperatura: int = 100,
        semente: Optional[int] = None,
    ):
        self.n_componentes = n_componentes
        self.redes = redes
        self.n_linhas = n_linhas
        self.n_colunas = n_colunas
        self.n_sitios = n_linhas * n_colunas

        if self.n_sitios < n_componentes:
            raise ValueError(
                "A grade precisa ter ao menos tantos sítios quanto "
                "componentes (n_linhas * n_colunas >= n_componentes)."
            )

        self.temperatura_inicial = temperatura_inicial
        self.temperatura_final = temperatura_final
        self.taxa_resfriamento = taxa_resfriamento
        self.iteracoes_por_temperatura = iteracoes_por_temperatura

        if semente is not None:
            random.seed(semente)
            np.random.seed(semente)

        # grade[linha, coluna] = índice do componente, ou -1 se vazio
        self.grade = np.full(self.n_sitios, -1, dtype=int)
        self.grade[:n_componentes] = np.arange(n_componentes)
        np.random.shuffle(self.grade)
        self.grade = self.grade.reshape(n_linhas, n_colunas)

        # posicao_componente[i] = (linha, coluna) do componente i
        self.posicao_componente: Dict[int, Tuple[int, int]] = {}
        for linha in range(n_linhas):
            for coluna in range(n_colunas):
                comp = self.grade[linha, coluna]
                if comp >= 0:
                    self.posicao_componente[comp] = (linha, coluna)

        # pré-computa, para cada componente, a lista de nets das quais participa
        # (permite recalcular custo de forma incremental e eficiente)
        self.nets_por_componente: Dict[int, List[str]] = {i: [] for i in range(n_componentes)}
        for net, comps in self.redes.items():
            for c in comps:
                self.nets_por_componente[c].append(net)

        self.historico = HistoricoSA()
        self.custo_atual = self._custo_total()
        self.historico.melhor_custo_global = self.custo_atual
        self.historico.melhor_posicoes_global = dict(self.posicao_componente)

    # ----------------------------------------------------------------- #
    # Cálculo de custo (HPWL)
    # ----------------------------------------------------------------- #
    def _hpwl_rede(self, net: str) -> float:
        comps = self.redes[net]
        linhas = [self.posicao_componente[c][0] for c in comps]
        colunas = [self.posicao_componente[c][1] for c in comps]
        return (max(colunas) - min(colunas)) + (max(linhas) - min(linhas))

    def _custo_total(self) -> float:
        return sum(self._hpwl_rede(net) for net in self.redes)

    def _custo_nets_de(self, componentes: List[int]) -> float:
        """Soma o HPWL apenas das nets que tocam os componentes informados
        (usado para calcular o custo de forma incremental antes/depois de
        um movimento, evitando recalcular a netlist inteira)."""
        nets_afetadas = set()
        for c in componentes:
            nets_afetadas.update(self.nets_por_componente[c])
        return sum(self._hpwl_rede(net) for net in nets_afetadas)

    # ----------------------------------------------------------------- #
    # Movimento: troca de posição entre dois sítios da grade
    # ----------------------------------------------------------------- #
    def _sortear_sitios(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        sitio_a = (random.randrange(self.n_linhas), random.randrange(self.n_colunas))
        sitio_b = (random.randrange(self.n_linhas), random.randrange(self.n_colunas))
        while sitio_b == sitio_a:
            sitio_b = (random.randrange(self.n_linhas), random.randrange(self.n_colunas))
        return sitio_a, sitio_b

    def _trocar(self, sitio_a: Tuple[int, int], sitio_b: Tuple[int, int]):
        comp_a = self.grade[sitio_a]
        comp_b = self.grade[sitio_b]

        self.grade[sitio_a], self.grade[sitio_b] = comp_b, comp_a

        if comp_a >= 0:
            self.posicao_componente[comp_a] = sitio_b
        if comp_b >= 0:
            self.posicao_componente[comp_b] = sitio_a

    def _tentar_movimento(self, temperatura: float) -> bool:
        """Sorteia dois sítios, calcula a variação de custo (HPWL) trocando-os
        e decide (critério de Metropolis) se aceita o movimento. Retorna True
        se o movimento foi aceito."""
        sitio_a, sitio_b = self._sortear_sitios()
        comp_a = self.grade[sitio_a]
        comp_b = self.grade[sitio_b]

        componentes_afetados = [c for c in (comp_a, comp_b) if c >= 0]
        if not componentes_afetados:
            return False  # troca de dois sítios vazios não muda nada

        custo_antes = self._custo_nets_de(componentes_afetados)
        self._trocar(sitio_a, sitio_b)
        custo_depois = self._custo_nets_de(componentes_afetados)

        delta = custo_depois - custo_antes

        aceitar = delta <= 0 or random.random() < math.exp(-delta / max(temperatura, 1e-9))

        if aceitar:
            self.custo_atual += delta
            return True
        else:
            # desfaz o movimento
            self._trocar(sitio_a, sitio_b)
            return False

    # ----------------------------------------------------------------- #
    # Execução principal do algoritmo
    # ----------------------------------------------------------------- #
    def executar(
        self, callback: Optional[Callable[[float, HistoricoSA], None]] = None
    ) -> HistoricoSA:
        """
        Executa o Recozimento Simulado até que a temperatura atinja
        `temperatura_final`.

        Parameters
        ----------
        callback : Optional[Callable]
            Função opcional chamada ao final de cada patamar de temperatura,
            recebendo (temperatura_atual, historico). Útil para progresso.

        Returns
        -------
        HistoricoSA
            Histórico de custo, temperatura e taxa de aceitação por patamar,
            além do melhor posicionamento global encontrado.
        """
        temperatura = self.temperatura_inicial

        while temperatura > self.temperatura_final:
            aceitos = 0
            for _ in range(self.iteracoes_por_temperatura):
                if self._tentar_movimento(temperatura):
                    aceitos += 1

                self.historico.custo_por_iteracao.append(self.custo_atual)
                self.historico.temperatura_por_iteracao.append(temperatura)

                if self.custo_atual < self.historico.melhor_custo_global:
                    self.historico.melhor_custo_global = self.custo_atual
                    self.historico.melhor_posicoes_global = dict(self.posicao_componente)

            taxa_aceitacao = aceitos / self.iteracoes_por_temperatura
            self.historico.taxa_aceitacao_por_temperatura.append(taxa_aceitacao)

            if callback is not None:
                callback(temperatura, self.historico)

            temperatura *= self.taxa_resfriamento

        return self.historico


# --------------------------------------------------------------------------- #
# 4. Funções utilitárias de visualização (usam matplotlib, chamado no notebook)
# --------------------------------------------------------------------------- #

def plotar_posicionamento(
    df_componentes: pd.DataFrame,
    redes: Dict[str, List[int]],
    posicoes: Dict[int, Tuple[int, int]],
    n_linhas: int,
    n_colunas: int,
    ax=None,
    titulo: str = "Posicionamento das células",
):
    """
    Desenha a grade de posicionamento, as células (com seus nomes) e as
    conexões da netlist (cada net desenhada como linhas ligando seus pinos
    ao centróide da net, formando uma "estrela" — representação comum em
    ferramentas de EDA para visualizar conectividade sem poluir o desenho
    com todas as combinações par-a-par).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # desenha a grade
    for l in range(n_linhas + 1):
        ax.axhline(l, color="lightgray", linewidth=0.5, zorder=0)
    for c in range(n_colunas + 1):
        ax.axvline(c, color="lightgray", linewidth=0.5, zorder=0)

    # desenha as nets (conexões) primeiro, por baixo das células
    cores_nets = plt.cm.tab20(np.linspace(0, 1, max(len(redes), 1)))
    for (net, comps), cor in zip(redes.items(), cores_nets):
        xs = [posicoes[c][1] + 0.5 for c in comps]
        ys = [posicoes[c][0] + 0.5 for c in comps]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        for x, y in zip(xs, ys):
            ax.plot([x, cx], [y, cy], color=cor, linewidth=1.0, alpha=0.6, zorder=1)

    # desenha as células
    for idx, (linha, coluna) in posicoes.items():
        nome = df_componentes.loc[idx, "Componente"]
        ax.add_patch(
            plt.Rectangle((coluna + 0.1, linha + 0.1), 0.8, 0.8,
                          facecolor="tab:blue", edgecolor="black", zorder=2)
        )
        ax.annotate(
            str(nome),
            (coluna + 0.5, linha + 0.5),
            ha="center", va="center",
            fontsize=7, color="white", zorder=3,
        )

    ax.set_xlim(0, n_colunas)
    ax.set_ylim(0, n_linhas)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(titulo)
    ax.set_xlabel("Coluna")
    ax.set_ylabel("Linha")
    return ax


def plotar_convergencia(historico: HistoricoSA, ax=None):
    """
    Plota a evolução do custo (HPWL total) ao longo das iterações, com a
    temperatura em um eixo secundário — permitindo visualizar como o
    resfriamento se relaciona com a estabilização da solução.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    iteracoes = range(1, len(historico.custo_por_iteracao) + 1)

    ax.plot(iteracoes, historico.custo_por_iteracao, color="tab:green", linewidth=0.8, label="Custo (HPWL total)")
    ax.axhline(historico.melhor_custo_global, color="tab:red", linestyle="--", label="Melhor custo global")
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Custo (HPWL total)")
    ax.set_title("Convergência do Recozimento Simulado")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(iteracoes, historico.temperatura_por_iteracao, color="tab:orange", linewidth=0.8, alpha=0.6, label="Temperatura")
    ax2.set_ylabel("Temperatura")
    ax2.set_yscale("log")

    linhas1, rotulos1 = ax.get_legend_handles_labels()
    linhas2, rotulos2 = ax2.get_legend_handles_labels()
    ax.legend(linhas1 + linhas2, rotulos1 + rotulos2, loc="upper right")

    return ax


def plotar_taxa_aceitacao(historico: HistoricoSA, ax=None):
    """Plota a taxa de aceitação de movimentos por patamar de temperatura,
    útil para diagnosticar o cronograma de resfriamento (cooling schedule)."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(historico.taxa_aceitacao_por_temperatura, color="tab:purple")
    ax.set_xlabel("Patamar de temperatura (ordem decrescente)")
    ax.set_ylabel("Taxa de aceitação")
    ax.set_title("Taxa de aceitação de movimentos ao longo do resfriamento")
    ax.grid(alpha=0.3)
    return ax
