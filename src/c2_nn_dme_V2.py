import os
import re
import random
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

class Node:
    def __init__(self, id_str, x=0.0, y=0.0, is_sink=False, is_root=False):
        self.id = id_str
        self.x = float(x)
        self.y = float(y)
        self.is_sink = is_sink
        self.is_root = is_root
        self.left = None
        self.right = None
        self.delay = 0.0
        self.sinks = [(float(x), float(y))] if is_sink else []

    def centroid(self):
        cx = sum(p[0] for p in self.sinks) / len(self.sinks)
        cy = sum(p[1] for p in self.sinks) / len(self.sinks)
        return (cx, cy)

def dist_manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def ler_instancias(caminho_arquivo):
    """
    Lê instâncias suportando dois formatos:
    1) Múltiplas listas: N no início, seguido por N linhas (x y), terminando em 0.
    2) Lista única direta: apenas coordenadas (x y) em cada linha.
    """
    with open(caminho_arquivo, 'r', encoding='utf-8-sig') as f:
        linhas_brutas = [linha.strip() for linha in f if linha.strip()]

    if not linhas_brutas:
        return [], []

    # Extrai todos os números (inteiros ou decimais) de cada linha, ignorando textos/símbolos
    linhas_numeros = []
    for linha in linhas_brutas:
        if linha.startswith('#') or linha.startswith('//'):
            continue
        nums = [float(x.replace(',', '.')) for x in re.findall(r'[-+]?\d*\.?\d+', linha)]
        if nums:
            linhas_numeros.append(nums)

    if not linhas_numeros:
        return [], linhas_brutas

    instancias = []
    idx = 0

    # Caso 1: Formato com cabeçalho N (ex: primeira linha tem apenas 1 número indicando a quantidade)
    if len(linhas_numeros[0]) == 1:
        while idx < len(linhas_numeros):
            if len(linhas_numeros[idx]) == 1:
                n = int(linhas_numeros[idx][0])
                if n == 0:
                    break
                idx += 1
                pontos = []
                while len(pontos) < n and idx < len(linhas_numeros):
                    if len(linhas_numeros[idx]) >= 2:
                        pontos.append((linhas_numeros[idx][0], linhas_numeros[idx][1]))
                    idx += 1
                if len(pontos) >= 2:
                    instancias.append(pontos)
            else:
                idx += 1

    # Caso 2: Arquivo simples contendo diretamente os pares (x, y) em cada linha
    if not instancias:
        pontos = [(nums[0], nums[1]) for nums in linhas_numeros if len(nums) >= 2]
        if len(pontos) >= 2:
            instancias.append(pontos)

    return instancias, linhas_brutas

def dme_embed_zero_skew(raiz, ultimo_cluster):
    """Posiciona os pontos de Steiner buscando Zero-Skew na métrica Manhattan."""
    def bottom_up(no):
        if no.is_sink:
            no.delay = 0.0
            return
        
        bottom_up(no.left)
        bottom_up(no.right)
        
        p1 = (no.left.x, no.left.y)
        p2 = (no.right.x, no.right.y)
        d = dist_manhattan(p1, p2)
        
        t1 = no.left.delay
        t2 = no.right.delay
        
        if abs(t1 - t2) <= d:
            e1 = (d + t2 - t1) / 2.0
            ratio = e1 / d if d > 0 else 0.5
        else:
            ratio = 0.0 if t1 > t2 else 1.0

        no.x = p1[0] + ratio * (p2[0] - p1[0])
        no.y = p1[1] + ratio * (p2[1] - p1[1])
        no.delay = max(t1 + dist_manhattan((no.x, no.y), p1),
                       t2 + dist_manhattan((no.x, no.y), p2))

    bottom_up(ultimo_cluster)
    raiz.left = ultimo_cluster
    raiz.delay = ultimo_cluster.delay + dist_manhattan((raiz.x, raiz.y), (ultimo_cluster.x, ultimo_cluster.y))

def executar_c1(pontos):
    """Algoritmo 4: C1 - NN+DME (Pareamento Guloso por Vizinho Mais Próximo)."""
    raiz = Node("r", pontos[0][0], pontos[0][1], is_root=True)
    clusters = [Node(f"s{i}", pt[0], pt[1], is_sink=True) for i, pt in enumerate(pontos[1:], start=1)]
    
    todos_nos = [raiz] + clusters.copy()
    contador_steiner = 1

    while len(clusters) > 1:
        melhor_dist = float('inf')
        melhor_par = (0, 1)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = dist_manhattan(clusters[i].centroid(), clusters[j].centroid())
                if d < melhor_dist:
                    melhor_dist = d
                    melhor_par = (i, j)

        i, j = melhor_par
        A = clusters[i]
        B = clusters[j]

        novo_cluster = Node(f"v{contador_steiner}")
        contador_steiner += 1
        novo_cluster.left = A
        novo_cluster.right = B
        novo_cluster.sinks = A.sinks + B.sinks

        clusters.pop(j)
        clusters.pop(i)
        clusters.append(novo_cluster)
        todos_nos.append(novo_cluster)

    ultimo_cluster = clusters[0]
    dme_embed_zero_skew(raiz, ultimo_cluster)

    return raiz, todos_nos

def salvar_saida_e_plotar(raiz, todos_nos, indice_instancia, txt_saida, img_saida):
    arestas = []
    wl_total = 0.0

    def coletar_arestas(no):
        nonlocal wl_total
        for filho in [no.left, no.right]:
            if filho is not None:
                custo = dist_manhattan((no.x, no.y), (filho.x, filho.y))
                wl_total += custo
                arestas.append((no, filho, custo))
                coletar_arestas(filho)

    coletar_arestas(raiz)

    with open(txt_saida, 'w', encoding='utf-8') as f:
        f.write(f"=== RESULTADO HEURÍSTICA DE CONTROLE C1 (NN+DME) ===\n")
        f.write(f"Instância sorteada: #{indice_instancia + 1}\n")
        f.write(f"Comprimento Total de Fio (WL): {wl_total:.2f}\n")
        f.write(f"Atraso Máximo (Raiz -> Sinks): {raiz.delay:.2f}\n\n")
        
        f.write("--- POSIÇÕES DOS NÓS ---\n")
        for no in todos_nos:
            tipo = "Raiz (Fonte)" if no.is_root else ("Sink" if no.is_sink else "Ponto de Steiner")
            f.write(f"{no.id:4s} | {tipo:16s} | Pos: ({no.x:6.2f}, {no.y:6.2f})\n")
            
        f.write("\n--- CONEXÕES DA TOPOLOGIA (PAI -> FILHO) ---\n")
        for pai, filho, custo in arestas:
            f.write(f"{pai.id} ({pai.x:.1f}, {pai.y:.1f}) --> {filho.id} ({filho.x:.1f}, {filho.y:.1f}) | Fio Manhattan: {custo:.2f}\n")

    plt.figure(figsize=(8, 8))
    
    for pai, filho, _ in arestas:
        plt.plot([pai.x, filho.x], [pai.y, pai.y], color='black', linewidth=1.8, zorder=1)
        plt.plot([filho.x, filho.x], [pai.y, filho.y], color='black', linewidth=1.8, zorder=1)

    for no in todos_nos:
        if no.is_root:
            plt.scatter(no.x, no.y, color='#e66101', marker='s', s=180, zorder=3, label='Raiz (Fonte)' if no.id=='r' else "")
        elif no.is_sink:
            plt.scatter(no.x, no.y, color='#1b9e77', marker='s', s=130, zorder=3, label='Sink' if no.id=='s1' else "")
        else:
            plt.scatter(no.x, no.y, color='#2b83ba', marker='o', s=130, zorder=3, label='Ponto de Steiner' if no.id=='v1' else "")
        
        plt.annotate(no.id, (no.x, no.y), textcoords="offset points", xytext=(6, 6), fontsize=10, fontweight='bold')

    plt.title(f"Topologia e Roteamento C1 (NN+DME) - Instância #{indice_instancia + 1}\nWirelength Total (WL) = {wl_total:.2f}")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(img_saida, dpi=300)
    plt.close()

def selecionar_arquivo_txt():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    pasta_inicial = os.path.dirname(os.path.abspath(__file__))
    caminho = filedialog.askopenfilename(
        initialdir=pasta_inicial,
        title="Selecione o arquivo de instâncias (.txt)",
        filetypes=[("Arquivos de Texto", "*.txt"), ("Todos os Arquivos", "*.*")]
    )
    root.destroy()
    return caminho

if __name__ == "__main__":
    print("Aguardando seleção do arquivo na janela aberta...")
    arquivo_entrada = selecionar_arquivo_txt()

    if not arquivo_entrada:
        print("Nenhum arquivo foi selecionado. Operação cancelada.")
    else:
        instancias, linhas_brutas = ler_instancias(arquivo_entrada)
        
        if not instancias:
            print(f"\n[ERRO] O arquivo '{os.path.basename(arquivo_entrada)}' não contém instâncias válidas.")
            print(f"Total de linhas não-vazias encontradas no arquivo: {len(linhas_brutas)}")
            if linhas_brutas:
                print("Primeiras linhas encontradas no arquivo:")
                for l in linhas_brutas[:5]:
                    print(f"  -> {l}")
            else:
                print("Motivo: O arquivo está completamente vazio (0 bytes). Salve o arquivo no editor (Ctrl+S) antes de selecioná-lo!")
        else:
            pasta_saida = os.path.dirname(arquivo_entrada)
            nome_base = os.path.splitext(os.path.basename(arquivo_entrada))[0]

            arquivo_saida_txt = os.path.join(pasta_saida, f"resultado_{nome_base}.txt")
            arquivo_saida_img = os.path.join(pasta_saida, f"topologia_c1_{nome_base}.png")

            idx_sorteado = random.randint(0, len(instancias) - 1)
            instancia_escolhida = instancias[idx_sorteado]

            raiz, todos_nos = executar_c1(instancia_escolhida)
            salvar_saida_e_plotar(raiz, todos_nos, idx_sorteado, arquivo_saida_txt, arquivo_saida_img)

            print(f"\nArquivo lido: {os.path.basename(arquivo_entrada)} ({len(instancias)} instância(s) encontrada(s))")
            print(f"Instância #{idx_sorteado + 1} (com {len(instancia_escolhida)} pontos) processada com sucesso!")
            print(f"Conexões salvas em: {arquivo_saida_txt}")
            print(f"Imagem gerada em: {arquivo_saida_img}")