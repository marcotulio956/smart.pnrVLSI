"""Run CTS heuristics on ISPD 2009 clock-synthesis benchmarks.

The generated result uses the contest's native format.  A buffer is inserted
at every branch edge (at the parent coordinate), which keeps the topology
compatible with the contest electrical evaluator without changing the
geometric tree produced by :mod:`cts_heuristicas`.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from cts_heuristicas import (
    ArvoreCTS,
    InstanciaCTS,
    dme_embed,
    heuristica_c2,
    heuristica_c3,
    heuristica_h3,
)


@dataclass
class ISPDDesign:
    path: Path
    instancia: InstanciaCTS
    wire_type: int = 0
    buffer_type: int = 0


def ler_design(caminho: str | Path) -> ISPDDesign:
    """Parse the subset of the ISPD input format needed by CTS."""
    path = Path(caminho)
    lines = [line.split("//", 1)[0].strip() for line in path.read_text().splitlines()]
    lines = [line for line in lines if line]
    source = None
    sinks: List[Tuple[float, float]] = []
    cargas: List[float] = []
    wire_type = 0
    buffer_type = 0
    nome = path.name
    i = 0
    while i < len(lines):
        fields = lines[i].split()
        key = fields[0].lower()
        if len(fields) == 4 and all(re.fullmatch(r"[-+]?\d+", x) for x in fields):
            i += 1
            continue
        if key == "source":
            source = (float(fields[2]), float(fields[3]))
        elif key == "num" and len(fields) >= 3 and fields[1].lower() == "sink":
            count = int(fields[2])
            for row in lines[i + 1:i + 1 + count]:
                f = row.split()
                sinks.append((float(f[1]), float(f[2])))
                cargas.append(float(f[3]))
            i += count
        elif key == "num" and len(fields) >= 3 and fields[1].lower() == "wirelib":
            count = int(fields[2])
            i += count
        elif key == "num" and len(fields) >= 3 and fields[1].lower() == "buflib":
            count = int(fields[2])
            if count:
                buffer_type = int(lines[i + 1].split()[0])
            i += count
        i += 1
    if source is None or not sinks:
        raise ValueError(f"invalid ISPD design: {path}")
    inst = InstanciaCTS(nome, np.asarray(sinks), np.asarray(cargas), source)
    return ISPDDesign(path, inst, wire_type, buffer_type)


def _node_position(arv: ArvoreCTS, emb, node: int) -> Tuple[float, float]:
    if node < arv.n:
        x, y = arv.inst.xy[node]
        return float(x), float(y)
    x, y = emb.pos[node]
    return float(x), float(y)


def escrever_resultado(
    design: ISPDDesign,
    arv: ArvoreCTS,
    destino: str | Path,
    inserir_buffers: bool = True,
) -> Path:
    """Write an :class:`ArvoreCTS` as an ISPD result file."""
    emb = dme_embed(arv)
    path = Path(destino)
    internal = list(range(arv.n, 2 * arv.n - 1))
    sink_ids = {i: 2 * arv.n + i for i in range(arv.n)}
    node_positions: Dict[int, Tuple[float, float]] = {
        v: _node_position(arv, emb, v) for v in internal
    }
    wires: List[Tuple[int, int, int]] = []
    buffers: List[Tuple[int, int, int]] = []
    next_id = max(
        max(node_positions, default=0), max(sink_ids.values(), default=0)
    ) + 1

    def endpoint(node: int) -> int:
        return sink_ids[node] if node < arv.n else node

    def add_edge(
        parent_id: int, child: int, parent_xy: Tuple[float, float], depth: int
    ) -> None:
        nonlocal next_id
        child_id = endpoint(child)
        if not inserir_buffers:
            wires.append((parent_id, child_id, design.wire_type))
            return
        buffer_id = next_id
        next_id += 1
        node_positions[buffer_id] = parent_xy
        buffers.append((parent_id, buffer_id, design.buffer_type))
        if child < arv.n and depth % 2 == 0:
            extra_id = next_id
            next_id += 1
            node_positions[extra_id] = parent_xy
            buffers.append((buffer_id, extra_id, design.buffer_type))
            buffer_id = extra_id
        wires.append((buffer_id, child_id, design.wire_type))

    root_xy = (float(design.instancia.raiz[0]), float(design.instancia.raiz[1]))
    add_edge(0, arv.top, root_xy, 1)
    stack = [(arv.top, 1)]
    while stack:
        parent, depth = stack.pop()
        parent_xy = node_positions[parent]
        for child in (arv.esq[parent], arv.dir[parent]):
            add_edge(parent, child, parent_xy, depth + 1)
            if child >= arv.n:
                stack.append((child, depth + 1))

    with path.open("w") as f:
        f.write("sourcenode 0 0\n")
        f.write(f"num node {len(node_positions)}\n")
        for node, (x, y) in sorted(node_positions.items()):
            f.write(f"{node} {x:.6f} {y:.6f}\n")
        f.write(f"num sinknode {arv.n}\n")
        for sink, node in sink_ids.items():
            f.write(f"{node} {sink + 1}\n")
        f.write(f"num wire {len(wires)}\n")
        for src, dst, kind in wires:
            f.write(f"{src} {dst} {kind}\n")
        f.write(f"num buffer {len(buffers)}\n")
        for src, dst, kind in buffers:
            f.write(f"{src} {dst} {kind}\n")
    return path


def executar_heuristica(
    design: ISPDDesign,
    heuristic: str = "H3",
    t_max: float = 2.0,
) -> ArvoreCTS:
    """Build a tree with H3, or deterministic C2/C3 for comparison."""
    if heuristic == "H3":
        arv, _ = heuristica_h3(design.instancia, math.inf, t_max=t_max, semente=0)
    elif heuristic == "C2":
        arv, _ = heuristica_c2(design.instancia, math.inf)
    elif heuristic == "C3":
        arv, _ = heuristica_c3(design.instancia, math.inf)
    else:
        raise ValueError("heuristic must be H3, C2, or C3")
    assert arv is not None
    return arv


def avaliar_resultado(
    design: ISPDDesign,
    result: str | Path,
    evaluator: str | Path,
    model: str | Path,
    ignorar_bloqueios: bool = True,
) -> str:
    """Run the official evaluator and return its stdout/stderr text."""
    evaluator = Path(evaluator).resolve()
    model = Path(model).resolve()
    command = ["perl", str(evaluator)]
    if ignorar_bloqueios:
        command.append("-b")
    command.append("-s")
    result = Path(result).resolve()
    workdir = result.parent
    links = []
    for subckt in Path(model).resolve().parent.glob("*.subckt"):
        target = workdir / subckt.name
        if not target.exists():
            target.symlink_to(subckt)
            links.append(target)
    command.extend([str(design.path.resolve()), result.name, str(model)])
    try:
        proc = subprocess.run(
            command, text=True, capture_output=True, check=False, cwd=workdir,
        )
    finally:
        for link in links:
            link.unlink(missing_ok=True)
    return (proc.stdout + proc.stderr).strip()


def avaliar_diretorio(
    benchmark_dir: str | Path,
    output_dir: str | Path,
    evaluator: str | Path,
    model: str | Path,
    heuristic: str = "H3",
    t_max: float = 2.0,
) -> List[dict]:
    """Evaluate every standard ``ispd09f*`` design in a directory."""
    benchmark_dir, output_dir = Path(benchmark_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(benchmark_dir.glob("ispd09f*")):
        if not path.is_file():
            continue
        design = ler_design(path)
        start = time.perf_counter()
        arv = executar_heuristica(design, heuristic, t_max)
        result = escrever_resultado(design, arv, output_dir / f"{path.name}.{heuristic}.res")
        report = avaliar_resultado(design, result, evaluator, model)
        match = re.search(r"\bCLR\s+([0-9.]+).*?\bC\s+([0-9.]+)", report)
        rows.append({
            "design": path.name,
            "sinks": design.instancia.n,
            "heuristic": heuristic,
            "result": str(result),
            "CLR_ps": float(match.group(1)) if match else math.nan,
            "cap_fF": float(match.group(2)) if match else math.nan,
            "seconds": time.perf_counter() - start,
            "report": report,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--heuristic", choices=("H3", "C2", "C3"), default="H3")
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()
    for row in avaliar_diretorio(
        args.benchmark_dir, args.output_dir, args.evaluator, args.model,
        args.heuristic, args.seconds):
        print(row["report"])


if __name__ == "__main__":
    main()
