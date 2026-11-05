# Proposta de Projeto AOC: Global Routing em VLSI com Ant Colony Optimization

### PPGCC Computação Natural -- Marco Túlio Siqueira da Mata

## 1. Descrição do Problema

O *routing* é uma etapa fundamental do projeto físico de circuitos integrados VLSI. Após o posicionamento dos componentes (*placement*), é necessário estabelecer as conexões elétricas entre os pinos, respeitando restrições físicas e tecnológicas, como capacidade das camadas metálicas, obstáculos, espaçamento entre fios e a disponibilidade de vias.

Este projeto terá como foco o **Global Routing**. Nessa etapa, o circuito é abstraído como uma grade de regiões (*G-cell grid*), e cada conexão (*net*) recebe um caminho ou uma árvore que liga seus terminais. O resultado consiste em guias de roteamento que serão posteriormente refinadas pelo *Detailed Routing*.

O problema é combinatório e envolve competição por recursos: diferentes nets podem utilizar as mesmas regiões e arestas da grade, mas a capacidade de roteamento de cada recurso é limitada. Portanto, minimizar o comprimento dos fios isoladamente não é suficiente; é necessário também controlar o congestionamento e o *overflow*, evitando soluções que excedam a capacidade disponível.

## 2. Objetivo

Desenvolver e avaliar um algoritmo baseado em **Ant Colony Optimization (ACO)** para resolver o problema de Global Routing em circuitos VLSI.

A solução deverá buscar um compromisso entre:

- **Wirelength:** minimizar o comprimento total dos caminhos.
- **Congestionamento:** distribuir a demanda de roteamento de acordo com a capacidade disponível.
- **Overflow:** minimizar ou eliminar a demanda que excede a capacidade dos recursos.
- **Viabilidade e escalabilidade:** produzir soluções válidas em instâncias de diferentes dimensões, com tempo e memória compatíveis com os benchmarks.

O desempenho será comparado com ferramentas consolidadas de roteamento global e suas pontuações nos benchmarks associados aos ISPD Global Routing Contests.

## 3. Estado da Arte

O Global Routing é tradicionalmente tratado por métodos que combinam construção de árvores de Steiner, roteamento por padrões, negociação de congestionamento e etapas iterativas de *rip-up and reroute*.

Entre as referências práticas mais relevantes estão:

- **FastRoute:** algoritmo de roteamento global baseado em *pattern routing* e negociação iterativa de congestionamento. É uma referência consolidada e integra o fluxo aberto OpenROAD.
- **CUGR:** roteador global que combina diferentes etapas de construção e refinamento de rotas, incluindo roteamento por padrões, consideração de resistência, *maze routing* e *rip-up and reroute*. Está disponível como alternativa no fluxo OpenROAD.
- **Hippo e RL-Route:** soluções de destaque nos ISPD Global Routing Contests de 2024 e 2025, respectivamente entre os primeiros colocados. Os contests recentes enfatizam escalabilidade, congestionamento e, no caso de 2025, desempenho de roteamento.
- **Métodos analíticos e orientados a desempenho:** trabalhos recentes exploram otimização de topologia, estimativas dinâmicas de congestionamento e co-otimização de temporização e capacidade [7]. Essas abordagens mostram que a qualidade do roteamento não deve ser avaliada apenas pelo comprimento dos fios.

Embora ACO seja uma metaheurística conhecida para problemas de caminho e roteamento, sua aplicação específica a VLSI não condiz com o método dominante nos fluxos de Global Routing de referência atuais. Há trabalhos que investigam variantes híbridas de ACO para roteamento VLSI, inclusive combinando ACO com *River Formation Dynamics*. Essa diferença entre a literatura de ACO e os roteadores de referência motiva investigar se uma formulação cuidadosa de ACO pode oferecer resultados competitivos, especialmente na negociação de recursos congestionados.

Como baselines experimentais, propõe-se utilizar **FastRoute/OpenROAD** e **CUGR**, além de considerar os resultados e as instâncias dos contests ISPD 2024 ou 2025. Os vencedores dos contests, como Hippo e RL-Route, são referências adicionais de estado da arte; a utilização direta dependerá da disponibilidade dos códigos e aplicabilidade a medida que este projeto avança, bem como formatos de entrada dos benchmarks.

## 4. Solução Inicial Proposta: Ant Colony Optimization

A proposta é modelar o Global Routing como um problema de construção cooperativa de caminhos em uma grade de roteamento, utilizando ACO como algoritmo principal. Cada formiga constrói uma rota entre terminais de uma net, escolhendo sucessivamente arestas da grade. As escolhas são guiadas por informação heurística local e pelos níveis de feromônio acumulados em soluções anteriores.

### 4.1 Representação do problema

A região do circuito será discretizada em um grafo $G=(V,E)$, no qual os vértices representam pontos ou regiões da grade e as arestas representam movimentos de roteamento possíveis. Cada aresta $e$ possui capacidade $c_e$, demanda atual $d_e$ e custo associado ao seu uso.

Para uma net de dois terminais, uma formiga constrói um caminho entre origem e destino. Para nets com múltiplos terminais, a extensão inicial poderá construir uma árvore conectando os terminais progressivamente, por exemplo, roteando cada novo terminal até a árvore parcial já construída. A formulação multi-terminal será avaliada após a implementação do caso de dois terminais.

### 4.2 Construção probabilística das rotas

Em cada passo, uma formiga localizada no vértice $i$ escolhe o próximo vértice $j$ entre os vizinhos viáveis. A probabilidade de transição será:

$$
P_{ij}^{(k)} =
\frac{\tau_{ij}^{\alpha}\,\eta_{ij}^{\beta}}
{\displaystyle\sum_{l\in N_i^{(k)}}\tau_{il}^{\alpha}\,\eta_{il}^{\beta}},
\qquad j\in N_i^{(k)}.
$$

Nessa expressão, $\tau_{ij}$ representa o feromônio associado à aresta, $\eta_{ij}$ representa a informação heurística, e $\alpha$ e $\beta$ controlam a influência relativa dessas informações. O conjunto $N_i^{(k)}$ contém os movimentos admissíveis para a formiga $k$.

A informação heurística deverá considerar a distância até o destino e o congestionamento do recurso. Uma possibilidade inicial é:

$$
\eta_{ij} =
\frac{1}
{1+\lambda\,\Delta L_{ij}+\mu\,\phi(d_{ij},c_{ij})},
$$

em que $\Delta L_{ij}$ representa o incremento de comprimento, $\phi$ penaliza a utilização de recursos congestionados e $\lambda,\mu$ ponderam os critérios. Uma penalidade possível é baseada no excesso de demanda:

$$
\phi(d_e,c_e)=
\max\left(0,\frac{d_e-c_e}{c_e}\right).
$$

A função poderá considerar a demanda prevista após a inclusão da rota, evitando que formigas favoreçam as regiões mais próximas de sua capacidade.

### 4.3 Atualização de feromônio e negociação de congestionamento

Após a construção das rotas, o feromônio será atualizado por evaporação e reforço das soluções de melhor qualidade:

$$
\tau_e \leftarrow (1-\rho)\tau_e+\Delta\tau_e,
$$

onde $\rho$ é a taxa de evaporação. O reforço $\Delta\tau_e$ será maior para arestas pertencentes a soluções de menor custo global, considerando comprimento e congestionamento.

A função de avaliação poderá combinar os critérios:

$$
F(R)=w_L L(R)+w_O O(R)+w_C C(R),
$$

em que $L(R)$ é o comprimento total dos fios, $O(R)$ é o overflow total e $C(R)$ é uma penalidade de congestionamento. Os pesos $w_L,w_O,w_C$ controlarão o compromisso entre os objetivos. O overflow deverá receber penalidade elevada, pois representa violação da capacidade de roteamento.

Para lidar com a competição entre nets, será utilizada uma estratégia iterativa inspirada em *Negotiated Congestion*: após uma rodada de roteamento, a demanda dos recursos será recalculada; regiões congestionadas terão seus custos aumentados, influenciando as escolhas das formigas nas iterações seguintes. Rotas problemáticas poderão ser removidas e reconstruídas (*rip-up and reroute*), permitindo que a colônia explore alternativas.

### 4.4 Etapas de implementação e avaliação

A implementação será desenvolvida incrementalmente:

1. **Protótipo em grade 2D:** implementar ACO para roteamento entre dois terminais, com obstáculos, custos de aresta e comparação com caminhos mínimos.
2. **Congestionamento multi-net:** introduzir capacidade por recurso, demanda compartilhada e atualização de custos entre rodadas.
3. **Nets multi-terminais:** estender a construção de caminhos para árvores de conexão, utilizando uma árvore parcial como alvo para novos terminais.
4. **Refinamento:** investigar estratégias de *rip-up and reroute*, inicialização de feromônio com informação geométrica e ajustes adaptativos dos parâmetros.
5. **Avaliação em benchmarks:** integrar a solução aos formatos de entrada e saída disponíveis e comparar seus resultados com FastRoute/OpenROAD e CUGR.

#### Métricas

As métricas principais serão wirelength total, overflow, número de recursos congestionados, tempo de execução e memória utilizada. Quando disponíveis, também serão observadas métricas relacionadas à qualidade temporal e à viabilidade do roteamento detalhado.

### 4.5 Motivação e continuidade do projeto

A proposta dá continuidade a um projeto de VLSI no qual já foram implementadas soluções para outras etapas do projeto físico: **placement**, utilizando Simulated Annealing, e **Clock Tree Synthesis**, formulado como uma heurísticas que envolve o problema de Árvore Retilínea de Steiner (RSMT) e o algoritimo Deferred-Merge Embedding (DME). O presente trabalho pretende avançar para a etapa de **routing**, investigando ACO como uma abordagem de computação natural para explorar caminhos e negociar recursos compartilhados. A intenção é construir, em suma, uma suite de implementações para o problema de **Place&Route**, da industria de EDA, e aplicáveis aos projetos do laboratório Nanocomp; avaliaremos as soluções frente aos benchmarks públicos dos ISPD Contests.

## 5. Referências Principais

1. Xu, Y.; Dong, Y.; Zhang, Y.; et al. **FastRoute: A Step to Integrate Global Routing into Placement.** *Proceedings of the International Conference on Computer-Aided Design (ICCAD)*, 2009.
2. **The OpenROAD Project. Global Routing Documentation.**  https://openroad.readthedocs.io/en/latest/main/src/grt/README.html
3. **ISPD 2024 Contest: GPU/ML-Enhanced Large Scale Global Routing.**  https://www.ispd.cc/contests/24/
4. **ISPD 2025 Contest: Performance-Driven Large Scale Global Routing.** https://www.ispd.cc/contests/25/
5. Liang, R.; Agnesina, A.; Liu, W.-H.; Ren, M. H. **GPU/ML-Enhanced Large Scale Global Routing Contest.** *Proceedings of ISPD*, 2024. https://research.nvidia.com/publication/2024-03_gpuml-enhanced-large-scale-global-routing-contest
6. **A Hybrid RFD-ACO Approach for Routing Optimization in VLSI Physical Design.** *Journal of Information and Optimization Sciences*, 2018. DOI: https://doi.org/10.1080/02522667.2017.1372910
7. **A Dynamic Congestion-Aware Analytic Initial Routing Flow for VLSI Designs.** *Conference of Science and Technology of Integrated Circuits (CSTIC)*, 2025. DOI: https://doi.org/10.1109/CSTIC64481.2025.11017940
8. **TiCo-Route: A Timing-Congestion Co-Optimization Global Routing Framework.** *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, 2026. DOI: https://doi.org/10.1109/TCAD.2026.3715075

## Todas Referências  
1. M. Pan, Y. Xu, Y. Zhang e C. Chu, “FastRoute: An Efficient and High-Quality Global Router,” *VLSI Design*, 2012. [DOI: 10.1155/2012/608362](https://doi.org/10.1155/2012/608362).
2. The OpenROAD Project, [“Global Routing — OpenROAD Documentation”](https://openroad.readthedocs.io/en/latest/main/src/grt/README.html).
3. J. Liu, C.-W. Pui, F. Wang e E. F. Y. Young, “CUGR: Detailed-Routability-Driven 3D Global Routing with Probabilistic Resource Model,” *DAC*, 2020. [DOI: 10.1109/DAC18072.2020.9218646](https://doi.org/10.1109/DAC18072.2020.9218646). Código: [CUHK-EDA/CUGR](https://github.com/cuhk-eda/cu-gr).
4. S. Lin, L. Xiao, J. Liu e E. F. Y. Young, “InstantGR: Scalable GPU Parallelization for Global Routing,” *ICCAD*, 2024. [DOI: 10.1145/3676536.3676787](https://doi.org/10.1145/3676536.3676787). Código: [CUHK-EDA/InstantGR](https://github.com/cuhk-eda/InstantGR).
5. L. Xiao et al., “InstantGR: Scalable GPU Parallelization for 3-D Global Routing,” *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, 2026. [DOI: 10.1109/TCAD.2025.3573685](https://doi.org/10.1109/TCAD.2025.3573685).
6. ISPD, [“ISPD 2025 Performance-Driven Large Scale Global Routing Contest”](https://www.ispd.cc/contests/25/index.html) e [repositório do contest](https://github.com/liangrj2014/ISPD25_contest).
7. J. Zhou, F. Huang, L. Chen, X. Zheng e Z. Zhu, “GPU-Accelerated Global Routing with Balanced Timing and Congestion Optimization,” *ASP-DAC*, 2026. [DOI: 10.1109/ASP-DAC66049.2026.11420639](https://doi.org/10.1109/ASP-DAC66049.2026.11420639).
8. H. Wu, X. Li, B. Yu e W. Zhu, “TiCo-Route: A Timing-Congestion Co-Optimization Global Routing Framework,” *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, 2026. [DOI: 10.1109/TCAD.2026.3715075](https://doi.org/10.1109/TCAD.2026.3715075).
9. ISPD, [“ISPD 2007 Global Routing Contest”](https://ispd.cc/contests/07/contest.html) e [“ISPD 2008 Global Routing Contest”](https://www.ispd.cc/contests/08/ispd08rc.html).
10. ISPD, [“ISPD 2018 Initial Detailed Routing Contest”](https://www.ispd.cc/contests/18/index.html) e [“ISPD 2019 Initial Detailed Routing Contest”](https://ispd.cc/contests/19/).
11. C. C. N. Chu e Y.-C. Wong, “FLUTE: Fast Lookup Table Based Rectilinear Steiner Minimal Tree Algorithm for VLSI Design,” *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, 2008. [DOI: 10.1109/TCAD.2007.907068](https://doi.org/10.1109/TCAD.2007.907068).


<!-- # Proposta de Projeto — Global Routing em VLSI

## 1. Descrição do Problema

O *routing* em circuitos VLSI consiste em estabelecer conexões físicas entre os terminais (*pins*) de cada rede (*net*) após o posicionamento dos componentes. O projeto terá como escopo o **global routing**: determinar caminhos aproximados para as nets em uma representação discretizada do circuito, antes do *detailed routing*, que define trilhas físicas e verifica regras geométricas de fabricação.

A região de roteamento pode ser modelada como um grafo 3D de *global routing cells* (GCells). Vértices representam regiões do circuito; arestas representam recursos de roteamento entre regiões ou camadas metálicas e possuem capacidades limitadas. Cada net, possivelmente com múltiplos terminais, deve ser conectada por uma árvore. O desafio combinatório decorre da competição entre nets pelos mesmos recursos.

Os critérios centrais são *overflow* (uso acima da capacidade), comprimento total dos fios (*wirelength*) e número de vias. Para uma aresta `e`, sejam `u_e` seu uso e `c_e` sua capacidade. O overflow total pode ser expresso por

$$
O = \sum_{e \in E} \max(0, u_e-c_e).
$$

Em geral, busca-se primeiro reduzir o overflow — idealmente a zero — e, em seguida, wirelength e vias. Em um fluxo orientado a desempenho, também são relevantes métricas temporais, como WNS e TNS.

## 2. Objetivo

Desenvolver e avaliar uma heurística de global routing que reduza o congestionamento e o comprimento das interconexões com custo computacional competitivo. A avaliação considerará overflow total/máximo, wirelength, número de vias e tempo de execução; como extensão, poderá incluir a criticidade temporal das nets e métricas WNS/TNS no fluxo OpenROAD.

Como motivação adicional, já foram desenvolvidos um algoritmo de *placement* baseado em *simulated annealing* e uma solução heurística para *clock tree synthesis*. A proposta de routing amplia essa linha de investigação, permitindo avaliar os três problemas com uma metodologia experimental comum e benchmarks dos ISPD Contests.

## 3. Estado da Arte

As soluções modernas combinam construção de topologias de árvores de Steiner, *pattern routing* (por exemplo, caminhos em L e Z), atribuição de camadas e estratégias iterativas de *rip-up and reroute* para resolver congestionamento. Entre as referências mais relevantes estão:

- **FastRoute / OpenROAD:** FastRoute é um global router clássico, de alta qualidade. O módulo de global routing do OpenROAD é baseado no FastRoute 4.1 e oferece iterações de redução de congestionamento, preferência para nets críticas e uma opção experimental para utilizar CUGR. É um baseline aberto e integrado a um fluxo físico completo [1, 2].
- **CUGR:** roteia diretamente no espaço 3D, combinando *pattern routing* e atribuição de camadas, *multi-level maze routing*, custos probabilísticos de recursos e *patching*. É um baseline acadêmico relevante por considerar a facilidade do *detailed routing*, e não apenas overflow e wirelength [3].
- **InstantGR:** router aberto acelerado por GPU, que explora paralelismo entre nets e dentro do roteamento, geração de lotes e transições flexíveis entre camadas. Apresentado no ICCAD 2024 e estendido no IEEE TCAD em 2026, reporta resultados de referência nos benchmarks ISPD 2024 [4, 5]. É especialmente pertinente caso haja acesso a GPU compatível.
- **Contests ISPD 2024–2025:** os contests recentes enfatizam roteamento em larga escala e aceleração por GPU/ML. No ISPD 2025, os três primeiros colocados foram **Hippo**, **RL-Route** e **It's MyRoute**. A avaliação de 2025 utiliza arquivos LEF/DEF/LIB/SDC e um fluxo baseado em OpenROAD, considerando congestionamento, WNS, TNS, potência e runtime [6]. Esses participantes são referências competitivas recentes; a comparação direta dependerá da disponibilidade e reprodutibilidade de seus artefatos.
- **Trabalhos de 2026:** *GPU-Accelerated Global Routing with Balanced Timing and Congestion Optimization* combina padrões L/Z para nets críticas com roteamento orientado a congestionamento para as demais; os autores reportam melhora de 19,4% no score ponderado e redução de 16,2% no runtime frente ao campeão do ISPD 2025 [7]. **TiCo-Route** combina fluxo de custo mínimo e programação linear inteira, com relaxação Lagrangiana para nets críticas, e otimização analítica de topologia para as demais [8]. São referências recentes, embora não necessariamente baselines executáveis de fácil integração.

**Baselines propostos:** FastRoute/OpenROAD como baseline principal; CUGR como baseline acadêmico adicional; e InstantGR caso haja infraestrutura GPU. Os resultados dos vencedores do ISPD 2025 poderão servir como referência competitiva, conforme a disponibilidade dos artefatos. Para os testes, recomenda-se combinar os benchmarks clássicos ISPD 2007/2008 com os conjuntos mais recentes ISPD 2024/2025. Os contests ISPD 2018/2019 tratam de *initial detailed routing*, não de global routing, e portanto não são benchmarks primários para o escopo escolhido [6, 9, 10].

## 4. Solução Inicial Proposta

Propõe-se uma **heurística híbrida de construção incremental com congestionamento negociado (*negotiated congestion*) e *rip-up-and-reroute***. A abordagem produz rapidamente uma solução inicial e revisa seletivamente as rotas que disputam recursos sobrecarregados.

1. **Construção das topologias:** gerar uma árvore de conexão por net, preferencialmente por uma aproximação de árvore de Steiner retilínea, como FLUTE, decompondo-a em segmentos de dois terminais [11].
2. **Roteamento inicial:** conectar os segmentos com padrões L/Z e atribuição de camadas, considerando comprimento e custo de vias.
3. **Detecção de congestionamento:** calcular o uso de cada aresta e identificar recursos cuja utilização exceda a capacidade.
4. **Rip-up and reroute:** remover e reconstruir seletivamente as rotas que atravessam regiões congestionadas. O custo de cada aresta será atualizado dinamicamente conforme comprimento, uso atual e histórico de congestionamento. Quando os padrões simples não forem suficientes, utilizar *maze routing* 3D, por exemplo, com busca A*.
5. **Parada e refinamento:** encerrar quando o overflow for eliminado, o limite de iterações for atingido ou não houver melhora significativa. Uma etapa final poderá reduzir vias e wirelength sem reintroduzir congestionamento.

A implementação poderá começar com uma representação simplificada 2D/3D e evoluir para os formatos e métricas dos contests recentes. A comparação deverá manter constantes benchmarks, recursos computacionais e condições de execução. Como extensão, a prioridade de roteamento e o custo das arestas poderão incorporar a criticidade temporal das nets. -->


