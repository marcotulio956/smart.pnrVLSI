import os
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
    """Lê o arquivo txt no formato especificado (N seguido por N coordenadas x y, terminando em 0)."""
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        linhas = [linha.strip() for linha in f if linha.strip()]

    instancias = []
    idx = 0
    while idx < len(linhas):
        n = int(linhas[idx])
        if n == 0:
            break
        idx += 1
        pontos = []
        for _ in range(n):
            partes = linhas[idx].split()
            pontos.append((float(partes[0]), float(partes[1])))
            idx += 1
        instancias.append(pontos)
    return instancias

def dme_embed_zero_skew(raiz, ultimo_cluster):
    """
    Executa o DME simplificado para posicionar os pontos internos (Steiner)
    buscando igualar os atrasos (Zero-Skew) na métrica Manhattan.
    """
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
        
        # Distância que o ponto de fusão deve ficar de cada filho para zerar o skew
        if abs(t1 - t2) <= d:
            e1 = (d + t2 - t1) / 2.0
            ratio = e1 / d if d > 0 else 0.5
        else:
            # Caso um ramo já tenha atraso maior que a distância física (exige detour)
            ratio = 0.0 if t1 > t2 else 1.0
            e1 = max(t1, t2) - t1

        # Posiciona o ponto de Steiner no caminho retilíneo entre os filhos
        no.x = p1[0] + ratio * (p2[0] - p1[0])
        no.y = p1[1] + ratio * (p2[1] - p1[1])
        no.delay = max(t1 + dist_manhattan((no.x, no.y), p1),
                       t2 + dist_manhattan((no.x, no.y), p2))

    bottom_up(ultimo_cluster)
    raiz.left = ultimo_cluster
    raiz.delay = ultimo_cluster.delay + dist_manhattan((raiz.x, raiz.y), (ultimo_cluster.x, ultimo_cluster.y))

def executar_c2(pontos):
    """Algoritmo 5: C2 - MMM+DME (Método das Médias e Medianas)."""
    raiz = Node("r", pontos[0][0], pontos[0][1], is_root=True)
    sinks_iniciais = [Node(f"s{i}", pt[0], pt[1], is_sink=True) for i, pt in enumerate(pontos[1:], start=1)]
    
    todos_nos = [raiz] + sinks_iniciais.copy()
    contador_steiner = 1

    def particionar(P, eixo):
        nonlocal contador_steiner
        if len(P) == 1:
            return P[0]
        
        if eixo == 'x':
            P_ordenado = sorted(P, key=lambda no: (no.x, no.y))
            outro = 'y'
        else:
            P_ordenado = sorted(P, key=lambda no: (no.y, no.x))
            outro = 'x'
            
        meio = len(P_ordenado) // 2
        P1 = P_ordenado[:meio]
        P2 = P_ordenado[meio:]
        
        novo_no = Node(f"v{contador_steiner}")
        contador_steiner += 1
        
        novo_no.left = particionar(P1, outro)
        novo_no.right = particionar(P2, outro)
        novo_no.sinks = novo_no.left.sinks + novo_no.right.sinks
        
        todos_nos.append(novo_no)
        return novo_no

    ultimo_cluster = particionar(sinks_iniciais, 'x')
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

    # 1. Gera o arquivo TXT com as conexões e métricas
    with open(txt_saida, 'w', encoding='utf-8') as f:
        f.write(f"=== RESULTADO HEURÍSTICA DE CONTROLE C2 (MMM+DME) ===\n")
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

    # 2. Gera a imagem da topologia com roteamento retilíneo (em L)
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

    plt.title(f"Topologia e Roteamento C2 (MMM+DME) - Instância #{indice_instancia + 1}\nWirelength Total (WL) = {wl_total:.2f}")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(img_saida, dpi=300)
    plt.close()

def selecionar_arquivo_txt():
    """Abre uma caixa de diálogo para o usuário escolher qualquer arquivo .txt."""
    root = tk.Tk()
    root.withdraw()  # Oculta a janela principal vazia do Tkinter
    root.attributes('-topmost', True)  # Força a caixa de diálogo a aparecer na frente do VS Code
    
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
        # Salva as saídas na mesma pasta onde está o arquivo .txt escolhido
        pasta_saida = os.path.dirname(arquivo_entrada)
        nome_base = os.path.splitext(os.path.basename(arquivo_entrada))[0]

        arquivo_saida_txt = os.path.join(pasta_saida, f"resultado_c2_{nome_base}.txt")
        arquivo_saida_img = os.path.join(pasta_saida, f"topologia_c2_{nome_base}.png")

        instancias = ler_instancias(arquivo_entrada)
        
        if not instancias:
            print("O arquivo selecionado não contém instâncias válidas.")
        else:
            idx_sorteado = random.randint(0, len(instancias) - 1)
            instancia_escolhida = instancias[idx_sorteado]

            raiz, todos_nos = executar_c2(instancia_escolhida)
            salvar_saida_e_plotar(raiz, todos_nos, idx_sorteado, arquivo_saida_txt, arquivo_saida_img)

            print(f"Arquivo lido: {os.path.basename(arquivo_entrada)}")
            print(f"Instância #{idx_sorteado + 1} processada com sucesso!")
            print(f"Conexões salvas em: {arquivo_saida_txt}")
            print(f"Imagem gerada em: {arquivo_saida_img}")