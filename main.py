import heapq, os, time, json
import warnings
from collections import deque
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
warnings.filterwarnings("ignore")

# LOAD DATA DARI CSV
NODES_CSV = "data/nodes.csv"
EDGES_CSV = "data/edges.csv"
GEOM_CSV  = "data/edges_geometry.csv"

def load_data():
    nodes_df = pd.read_csv(NODES_CSV)
    edges_df = pd.read_csv(EDGES_CSV)

    # NODES: dict nama → {lat, lon, type}
    NODES = {
        row["name"]: {"lat": row["lat"], "lon": row["lon"], "type": row["type"]}
        for _, row in nodes_df.iterrows()
    }

    # GEOMETRY: dict (from,to) → [[lon,lat], ...]  (dua arah)
    GEOM = {}
    if os.path.exists(GEOM_CSV):
        geom_df = pd.read_csv(GEOM_CSV)
        for _, row in geom_df.iterrows():
            coords = json.loads(row["coords_json"])
            GEOM[(row["from"], row["to"])] = coords
            GEOM[(row["to"],  row["from"])] = list(reversed(coords))

    # EDGES: list of (from, to, dist_m, speed_kmh, road_name)
    EDGES_RAW = [
        (
            row["from"],
            row["to"],
            row["dist_m"],
            row["speed_kmh"],
            row["road_name"],
            row.get("travel_time_sec", 0),
            row.get("traffic_light_delay_sec", 0),
            row.get("maneuver_delay_sec", 0),
            row["weight_sec"]
        )

        for _, row in edges_df.iterrows()
    ]

    return NODES, EDGES_RAW, GEOM

# BANGUN GRAF
def build_graph(NODES, EDGES_RAW) -> Dict:
    """
    Adjacency list. Bobot = waktu tempuh (detik):
        td = (3.6 × dist_m) / speed_kmh
    """
    graph = {name: [] for name in NODES}
    for (u,v,dist_m,speed_kmh,road_name,travel_time,traffic_delay,maneuver_delay,weight_sec) in EDGES_RAW:
        if u not in graph or v not in graph:
            continue
        w = weight_sec
        edge_u = {"to": v, "weight": w, "dist_m": dist_m, "speed": speed_kmh, "road": road_name, "travel_time_sec": travel_time, "traffic_light_delay_sec": traffic_delay, "maneuver_delay_sec": maneuver_delay}
        edge_v = {"to": u, "weight": w, "dist_m": dist_m, "speed": speed_kmh, "road": road_name, "travel_time_sec": travel_time, "traffic_light_delay_sec": traffic_delay, "maneuver_delay_sec": maneuver_delay}
        graph[u].append(edge_u)
        graph[v].append(edge_v)
    return graph

# 1. DIJKSTRA
def dijkstra(graph, start, end) -> Dict:
    """Shortest path berdasarkan bobot waktu. O((V+E) log V)."""
    t0 = time.perf_counter()
    dist  = {n: float("inf") for n in graph}
    dist[start] = 0.0
    pred  = {n: None for n in graph}
    heap  = [(0.0, start)]
    visited = set()
    nodes_visited, n_exam = [], 0

    while heap:
        cur_d, cur = heapq.heappop(heap)
        if cur in visited: continue
        visited.add(cur); nodes_visited.append(cur); n_exam += 1
        if cur == end: break
        for e in graph.get(cur, []):
            nb = e["to"]
            if nb in visited: continue
            nd = cur_d + e["weight"]
            if nd < dist[nb]:
                dist[nb] = nd; pred[nb] = (cur, e)
                heapq.heappush(heap, (nd, nb))

    t_ms = (time.perf_counter() - t0) * 1000
    if dist[end] == float("inf"):
        return {"found": False, "algo": "Dijkstra", "nodes_examined": n_exam, "time_ms": t_ms}

    path, edges_used, cur = [], [], end
    while cur:
        path.append(cur)
        if pred[cur]: edges_used.append(pred[cur][1])
        cur = pred[cur][0] if pred[cur] else None
    path.reverse(); edges_used.reverse()

    return {"found": True, "algo": "Dijkstra", "path": path,
            "total_sec": dist[end], "total_min": dist[end]/60,
            "total_dist_m": sum(e["dist_m"] for e in edges_used),
            "edges": edges_used, "nodes_visited": nodes_visited,
            "nodes_examined": n_exam, "time_ms": t_ms}

# 2. BFS
def bfs(graph, start, end) -> Dict:
    """Shortest path berdasarkan jumlah hop (tanpa bobot). O(V+E)."""
    t0 = time.perf_counter()
    visited = {start}; pred = {start: None}
    queue = deque([start])
    nodes_visited, n_exam = [], 0

    while queue:
        cur = queue.popleft()
        nodes_visited.append(cur); n_exam += 1
        if cur == end: break
        for e in graph.get(cur, []):
            nb = e["to"]
            if nb not in visited:
                visited.add(nb); pred[nb] = (cur, e); queue.append(nb)

    t_ms = (time.perf_counter() - t0) * 1000
    if end not in pred:
        return {"found": False, "algo": "BFS", "nodes_examined": n_exam, "time_ms": t_ms}

    path, edges_used, cur = [], [], end
    while cur:
        path.append(cur)
        if pred[cur]: edges_used.append(pred[cur][1])
        cur = pred[cur][0] if pred[cur] else None
    path.reverse(); edges_used.reverse()
    total_sec = sum(e["weight"] for e in edges_used)

    return {"found": True, "algo": "BFS", "path": path,
            "total_sec": total_sec, "total_min": total_sec/60,
            "total_dist_m": sum(e["dist_m"] for e in edges_used),
            "edges": edges_used, "nodes_visited": nodes_visited,
            "nodes_examined": n_exam, "time_ms": t_ms}

# 3. DFS
def dfs(graph, start, end) -> Dict:
    """Path via depth-first (tidak menjamin optimal). O(V+E)."""
    t0 = time.perf_counter()
    visited = set(); pred = {start: None}
    stack = [start]
    nodes_visited, n_exam = [], 0

    while stack:
        cur = stack.pop()
        if cur in visited: continue
        visited.add(cur); nodes_visited.append(cur); n_exam += 1
        if cur == end: break
        for e in reversed(graph.get(cur, [])):
            nb = e["to"]
            if nb not in visited and nb not in pred:
                pred[nb] = (cur, e); stack.append(nb)

    t_ms = (time.perf_counter() - t0) * 1000
    if end not in pred and start != end:
        return {"found": False, "algo": "DFS", "nodes_examined": n_exam, "time_ms": t_ms}

    path, edges_used, cur = [], [], end
    while cur:
        path.append(cur)
        if pred[cur]: edges_used.append(pred[cur][1])
        cur = pred[cur][0] if pred[cur] else None
    path.reverse(); edges_used.reverse()
    total_sec = sum(e["weight"] for e in edges_used)

    return {"found": True, "algo": "DFS", "path": path,
            "total_sec": total_sec, "total_min": total_sec/60,
            "total_dist_m": sum(e["dist_m"] for e in edges_used),
            "edges": edges_used, "nodes_visited": nodes_visited,
            "nodes_examined": n_exam, "time_ms": t_ms}

# PERBANDINGAN
def compare_algorithms(graph, start, end) -> Dict:
    print(f"  ⏳ Dijkstra ..."); r_d = dijkstra(graph, start, end)
    print(f"  ⏳ BFS      ..."); r_b = bfs(graph, start, end)
    print(f"  ⏳ DFS      ..."); r_f = dfs(graph, start, end)

    comp = {"Dijkstra": r_d, "BFS": r_b, "DFS": r_f, "start": start, "end": end}
    comp["same_path"] = {
        "Dijkstra == BFS": r_d.get("path") == r_b.get("path"),
        "Dijkstra == DFS": r_d.get("path") == r_f.get("path"),
        "BFS == DFS":      r_b.get("path") == r_f.get("path"),
    }
    times = {a: comp[a].get("time_ms", 9999) for a in ["Dijkstra","BFS","DFS"]}
    comp["fastest_algo"]      = min(times, key=times.get)
    comp["optimal_time_algo"] = "Dijkstra"
    return comp

# CETAK HASIL
def print_comparison(comp):
    start, end = comp["start"], comp["end"]
    print(f"\n{'='*65}")
    print(f"  PERBANDINGAN: DIJKSTRA vs BFS vs DFS")
    print(f"  Dari  : {start}")
    print(f"  Ke    : {end}")
    print(f"{'='*65}")

    for algo in ["Dijkstra","BFS","DFS"]:
        r = comp[algo]
        if not r.get("found"):
            print(f"  {algo}: RUTE TIDAK DITEMUKAN"); continue
        print(f"\n  [{algo}]")
        print(f"    Waktu Tempuh   : {r['total_min']:.2f} menit")
        print(f"    Total Jarak    : {r['total_dist_m']/1000:.3f} km")
        print(f"    Node di Jalur  : {len(r['path'])}")
        print(f"    Node Diperiksa : {r['nodes_examined']}")
        print(f"    Waktu Eksekusi : {r['time_ms']:.4f} ms")
        print(f"    Jalur          : {' → '.join(r['path'])}")

    print(f"\n{'─'*65}")
    print(f"  {'Metrik':<32} {'Dijkstra':>10} {'BFS':>10} {'DFS':>10}")
    print(f"  {'─'*62}")
    rows = [
        ("Waktu Tempuh (menit)",
         f"{comp['Dijkstra'].get('total_min',0):.2f}",
         f"{comp['BFS'].get('total_min',0):.2f}",
         f"{comp['DFS'].get('total_min',0):.2f}"),
        ("Jarak (km)",
         f"{comp['Dijkstra'].get('total_dist_m',0)/1000:.2f}",
         f"{comp['BFS'].get('total_dist_m',0)/1000:.2f}",
         f"{comp['DFS'].get('total_dist_m',0)/1000:.2f}"),
        ("Node Diperiksa",
         str(comp['Dijkstra'].get('nodes_examined',0)),
         str(comp['BFS'].get('nodes_examined',0)),
         str(comp['DFS'].get('nodes_examined',0))),
        ("Waktu Eksekusi (ms)",
         f"{comp['Dijkstra'].get('time_ms',0):.4f}",
         f"{comp['BFS'].get('time_ms',0):.4f}",
         f"{comp['DFS'].get('time_ms',0):.4f}"),
    ]
    for label, d, b, f in rows:
        print(f"  {label:<32} {d:>10} {b:>10} {f:>10}")

    print(f"\n  Jalur sama? Dijkstra==BFS: {'✅' if comp['same_path']['Dijkstra == BFS'] else '❌'}  "
          f"Dijkstra==DFS: {'✅' if comp['same_path']['Dijkstra == DFS'] else '❌'}  "
          f"BFS==DFS: {'✅' if comp['same_path']['BFS == DFS'] else '❌'}")
    print(f"  ⚡ Tercepat       : {comp['fastest_algo']}")
    print(f"  🏆 Jalur optimal  : {comp['optimal_time_algo']}")
    print(f"{'='*65}\n")

# VISUALISASI
NODE_COLORS = {
    "kampus": "#2ECC71",
    "mall": "#3498DB",
    "transportasi": "#E74C3C",
    "rumahsakit": "#E67E22",
    "pemerintahan": "#9B59B6",
    "landmark": "#F39C12",
    "wisata": "#1ABC9C",
    "persimpangan": "#95A5A6",
}

ALGO_COLORS = {
    "Dijkstra": "#FF4757",
    "BFS": "#2ED573",
    "DFS": "#FFA502"
}

def plot_comparison(comp, graph, NODES, GEOM, save_path=None):
    fig = plt.figure(figsize=(24, 13))
    fig.patch.set_facecolor("#0d1117")
    # LAYOUT
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=[2.4, 1, 1],
        height_ratios=[1, 1, 1],
        hspace=0.18,
        wspace=0.12
    )
    ax_map = fig.add_subplot(gs[:, 0])
    ax_dijkstra = fig.add_subplot(gs[0, 1:])
    ax_bfs      = fig.add_subplot(gs[1, 1:])
    ax_dfs      = fig.add_subplot(gs[2, 1:])
    all_axes = [
        ax_map,
        ax_dijkstra,
        ax_bfs,
        ax_dfs
    ]
    for ax in all_axes:
        ax.set_facecolor("#0d1117")
    start = comp["start"]
    end   = comp["end"]
    # DRAW ROAD NETWORK
    drawn = set()
    for u, nb_list in graph.items():
        for e in nb_list:
            v = e["to"]
            key = tuple(sorted([u, v]))
            if key in drawn:
                continue
            drawn.add(key)
            coords = GEOM.get((u, v), [])
            if coords:
                xs = [p[0] for p in coords]
                ys = [p[1] for p in coords]
            else:
                xs = [
                    NODES[u]["lon"],
                    NODES[v]["lon"]
                ]
                ys = [
                    NODES[u]["lat"],
                    NODES[v]["lat"]
                ]
            ax_map.plot(
                xs,
                ys,
                color="#1f3a5f",
                linewidth=0.7,
                alpha=0.40,
                zorder=1
            )
    # DRAW PATHS
    offsets = {
        "Dijkstra": 0.0,
        "BFS": 0.00012,
        "DFS": -0.00012
    }
    lws = {
        "Dijkstra": 4.5,
        "BFS": 3,
        "DFS": 3
    }
    zorders = {
        "Dijkstra": 10,
        "BFS": 8,
        "DFS": 7
    }

    for algo in ["DFS", "BFS", "Dijkstra"]:
        r = comp[algo]
        if not r.get("found"):
            continue
        path = r["path"]
        off = offsets[algo]
        for i in range(len(path)-1):
            u = path[i]
            v = path[i+1]
            coords = GEOM.get((u, v), [])
            if coords:
                xs = [p[0] + off for p in coords]
                ys = [p[1] + off for p in coords]
            else:
                xs = [
                    NODES[u]["lon"] + off,
                    NODES[v]["lon"] + off
                ]
                ys = [
                    NODES[u]["lat"] + off,
                    NODES[v]["lat"] + off
                ]
            ax_map.plot(
                xs,
                ys,
                color=ALGO_COLORS[algo],
                linewidth=lws[algo],
                alpha=0.95,
                zorder=zorders[algo]
            )
    # DRAW NODES + LABELS
    for name, nd in NODES.items():
        color = NODE_COLORS.get(nd["type"], "#aaaaaa")
        # START
        if name == start:
            ax_map.scatter(nd["lon"],nd["lat"],c="#FFFFFF",s=360,marker="*",edgecolors="#2ECC71",linewidths=2.5,zorder=20)
        # END
        elif name == end:
            ax_map.scatter(nd["lon"],nd["lat"],c="#FFFFFF",s=260,marker="D",edgecolors="#E74C3C",linewidths=2.5,zorder=20)
        # NORMAL
        else:
            ax_map.scatter(nd["lon"],nd["lat"],c=color,s=58,edgecolors="white",linewidths=0.45,zorder=5)
        # LABEL
        ax_map.annotate(name,(nd["lon"], nd["lat"]),xytext=(4, 3),textcoords="offset points",fontsize=6.7,color="#dddddd",alpha=0.92)

    # MAP SETTINGS
    ax_map.set_title(
        f"{start} → {end}",
        fontsize=18,
        color="white",
        fontweight="bold",
        pad=12
    )
    ax_map.tick_params(colors="#777777")
    ax_map.grid(alpha=0.08)
    for spine in ax_map.spines.values():
        spine.set_color("#334155")

    # LEGEND
    legend_elements = (
        [
            mpatches.Patch(color=c,label=t.capitalize())
            for t, c in NODE_COLORS.items()
        ]
        +
        [
            mlines.Line2D([0],[0],color=c,lw=3,label=a)
            for a, c in ALGO_COLORS.items()
        ]
    )

    ax_map.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=8,
        ncol=2,
        facecolor="#111827",
        edgecolor="#334155",
        labelcolor="white"
    )

    # BEST METRIC DETECTION
    metric_values = {
        "total_min": {
            algo: comp[algo]["total_min"]
            for algo in ["Dijkstra", "BFS", "DFS"]
        },
        "total_dist_m": {
            algo: comp[algo]["total_dist_m"]
            for algo in ["Dijkstra", "BFS", "DFS"]
        },
        "path_len": {
            algo: len(comp[algo]["path"])
            for algo in ["Dijkstra", "BFS", "DFS"]
        },
        "nodes_examined": {
            algo: comp[algo]["nodes_examined"]
            for algo in ["Dijkstra", "BFS", "DFS"]
        },
        "time_ms": {
            algo: comp[algo]["time_ms"]
            for algo in ["Dijkstra", "BFS", "DFS"]
        }
    }
    # winner tiap metric
    best_metrics = {}
    # score tiap algorithm
    algo_scores = {
        "Dijkstra": 0,
        "BFS": 0,
        "DFS": 0
    }
    for metric, values in metric_values.items():
        best_algo = min(values, key=values.get)
        best_metrics[metric] = best_algo
        algo_scores[best_algo] += 1
    # algorithm terbaik overall
    best_algorithm = max(
        algo_scores,
        key=algo_scores.get
    )
    # INFO PANELS
    algo_axes = {
        "Dijkstra": ax_dijkstra,
        "BFS": ax_bfs,
        "DFS": ax_dfs
    }
    for algo, ax in algo_axes.items():
        r = comp[algo]
        color = ALGO_COLORS[algo]
        ax.set_xticks([])
        ax.set_yticks([])
        # PANEL HIGHLIGHT
        if algo == best_algorithm:
            ax.set_facecolor("#111827")
            glow_width = 4
            title_text = f"{algo} Algorithm  [BEST]"
        else:
            glow_width = 2
            title_text = f"{algo} Algorithm"
        for sp in ax.spines.values():
            sp.set_color(color)
            sp.set_linewidth(glow_width)
        ax.set_title(
            title_text,
            fontsize=20,
            color=color,
            fontweight="bold",
            loc="left",
            pad=10
        )
        # METRIC ROWS
        metrics = [
            (
                "Waktu Tempuh",
                f"{r['total_min']:.2f} menit",
                best_metrics["total_min"] == algo
            ),
            (
                "Jarak",
                f"{r['total_dist_m']/1000:.2f} km",
                best_metrics["total_dist_m"] == algo
            ),
            (
                "Node Jalur",
                f"{len(r['path'])}",
                best_metrics["path_len"] == algo
            ),
            (
                "Node Dicek",
                f"{r['nodes_examined']}",
                best_metrics["nodes_examined"] == algo
            ),
            (
                "Komputasi",
                f"{r['time_ms']:.4f} ms",
                best_metrics["time_ms"] == algo
            ),
        ]
        y = 0.80
        for label, value, is_best in metrics:
            # highlight best metric
            if is_best:
                bbox = dict(
                    boxstyle="round,pad=0.25",
                    facecolor=color,
                    alpha=0.25,
                    edgecolor=color
                )
                text_color = "#ffffff"
                fontweight = "bold"
            else:
                bbox = None
                text_color = "#dddddd"
                fontweight = "normal"
            ax.text(
                0.03,
                y,
                f"{label:<14}: {value}",
                fontsize=14,
                color=text_color,
                fontweight=fontweight,
                family="monospace",
                bbox=bbox,
                transform=ax.transAxes
            )
            y -= 0.14
        # SCORE
        ax.text(
            0.78,
            0.82,
            f"Score: {algo_scores[algo]}/5",
            fontsize=16,
            color=color,
            fontweight="bold",
            transform=ax.transAxes
        )
        # PATH
        short_path = " → ".join(r["path"])
        ax.text(
            0.03,
            0.08,
            f"Path: {short_path}",
            fontsize=10,
            color="#bbbbbb",
            transform=ax.transAxes
        )
    # MAIN TITLE
    fig.suptitle(
        "Smart Navigation System — Dijkstra vs BFS vs DFS",
        fontsize=24,
        color="white",
        fontweight="bold",
        y=0.96
    )

    if save_path:
        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
            facecolor="#0d1117"
        )
    plt.show()
    plt.close()

# DIJKSTRA ANIMATION
NODE_STATE_COLORS = {
    "unvisited": "#57606f",
    "frontier": "#f1c40f",
    "visited": "#ff7f50",
    "current": "#ff4757",
    "path": "#2ed573",
    "start": "#00d2d3",
    "end": "#ff6b81"
}

def animate_dijkstra_advanced(
    graph,
    NODES,
    GEOM,
    start,
    end,
    interval=120
    ):

    # FIGURE
    fig, ax = plt.subplots(figsize=(18, 13))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # DIJKSTRA STATE
    dist = {n: float("inf") for n in graph}
    dist[start] = 0
    pred = {}
    visited = set()
    frontier = set([start])
    heap = [(0, start)]
    frames = []

    # STEP-BY-STEP DIJKSTRA
    while heap:
        cur_dist, cur = heapq.heappop(heap)
        if cur in visited:
            continue
        visited.add(cur)
        frontier.discard(cur)
        # save frame state
        frames.append({
            "current": cur,
            "visited": visited.copy(),
            "frontier": frontier.copy(),
            "pred": pred.copy(),
            "dist": dist.copy()
        })
        if cur == end:
            break

        # relax edges
        for e in graph[cur]:
            nb = e["to"]
            nd = cur_dist + e["weight"]
            if nd < dist[nb]:
                dist[nb] = nd
                pred[nb] = cur
                heapq.heappush(
                    heap,
                    (nd, nb)
                )
                frontier.add(nb)

    # FINAL PATH
    final_path = []
    if end in pred:
        node = end
        while node != start:
            final_path.append(node)
            node = pred[node]
        final_path.append(start)
        final_path.reverse()

    # DRAW FUNCTION
    def draw_frame(frame_idx):
        ax.clear()
        ax.set_facecolor("#0d1117")
        state = frames[frame_idx]
        current = state["current"]
        visited_nodes = state["visited"]
        frontier_nodes = state["frontier"]
        pred_now = state["pred"]

        # DRAW BASE EDGES
        drawn = set()
        for u, nb_list in graph.items():
            for e in nb_list:
                v = e["to"]
                key = tuple(sorted([u, v]))
                if key in drawn:
                    continue
                drawn.add(key)
                coords = GEOM.get((u, v), [])
                if coords:
                    xs = [p[0] for p in coords]
                    ys = [p[1] for p in coords]
                    ax.plot(
                        xs,
                        ys,
                        color="#2f3542",
                        linewidth=0.8,
                        alpha=0.35,
                        zorder=1
                    )
                else:
                    ax.plot(
                        [NODES[u]["lon"], NODES[v]["lon"]],
                        [NODES[u]["lat"], NODES[v]["lat"]],
                        color="#2f3542",
                        linewidth=0.5,
                        alpha=0.25,
                        zorder=1
                    )

        # DRAW SEARCH TREE
        for child, parent in pred_now.items():
            if parent not in NODES or child not in NODES:
                continue
            coords = GEOM.get((parent, child), [])
            if coords:
                xs = [p[0] for p in coords]
                ys = [p[1] for p in coords]
                ax.plot(
                    xs,
                    ys,
                    color="#70a1ff",
                    linewidth=2,
                    alpha=0.55,
                    zorder=2
                )

            else:
                ax.plot(
                    [NODES[parent]["lon"], NODES[child]["lon"]],
                    [NODES[parent]["lat"], NODES[child]["lat"]],
                    color="#70a1ff",
                    linewidth=1.5,
                    alpha=0.5,
                    zorder=2
                )

        # DRAW NODES
        for name, nd in NODES.items():
            color = NODE_STATE_COLORS["unvisited"]
            size = 60
            # START
            if name == start:
                color = NODE_STATE_COLORS["start"]
                size = 240
            # END
            elif name == end:
                color = NODE_STATE_COLORS["end"]
                size = 240
            # CURRENT
            elif name == current:
                color = NODE_STATE_COLORS["current"]
                size = 220
            # FRONTIER
            elif name in frontier_nodes:
                color = NODE_STATE_COLORS["frontier"]
                size = 120
            # VISITED
            elif name in visited_nodes:
                color = NODE_STATE_COLORS["visited"]
                size = 90
            ax.scatter(
                nd["lon"],
                nd["lat"],
                s=size,
                c=color,
                edgecolors="white",
                linewidths=0.7,
                zorder=5
            )
            ax.annotate(
                name,
                (nd["lon"], nd["lat"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6.2,
                color="white",
                alpha=0.9
            )

        # FINAL PATH
        if frame_idx == len(frames) - 1:
            for i in range(len(final_path)-1):
                u = final_path[i]
                v = final_path[i+1]
                coords = GEOM.get((u, v), [])
                if coords:
                    xs = [p[0] for p in coords]
                    ys = [p[1] for p in coords]
                    ax.plot(
                        xs,
                        ys,
                        color=NODE_STATE_COLORS["path"],
                        linewidth=5,
                        alpha=0.95,
                        zorder=10
                    )
                else:
                    ax.plot(
                        [NODES[u]["lon"], NODES[v]["lon"]],
                        [NODES[u]["lat"], NODES[v]["lat"]],
                        color=NODE_STATE_COLORS["path"],
                        linewidth=5,
                        alpha=0.95,
                        zorder=10
                    )

        # INFO PANEL
        info_text = (
            f"Dijkstra Algorithm Visualization\n\n"
            f"Current Node : {current}\n"
            f"Visited      : {len(visited_nodes)}\n"
            f"Frontier     : {len(frontier_nodes)}\n"
            f"Current Cost : {state['dist'][current]:.2f} sec\n"
        )

        ax.text(
            0.01,
            0.99,
            info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            color="white",
            bbox=dict(
                boxstyle="round",
                facecolor="#1e272e",
                alpha=0.85,
                edgecolor="#485460"
            )
        )

        # TITLE
        ax.set_title(
            f"SMART NAVIGATION SYSTEM\n"
            f"Dijkstra Traversal Animation\n"
            f"{start} → {end}",
            fontsize=16,
            color="white",
            fontweight="bold",
            pad=15
        )

        ax.tick_params(colors="#aaaaaa")

        for spine in ax.spines.values():
            spine.set_color("#444444")
        ax.grid(alpha=0.08)

    # CREATE ANIMATION
    ani = animation.FuncAnimation(
        fig,
        draw_frame,
        frames=len(frames),
        interval=interval,
        repeat=False
    )

    plt.show()

# MAIN
def main():
    print("\n" + "="*65)
    print("  SMART NAVIGATION SYSTEM — 40 NODE SURABAYA")
    print("  Dijkstra vs BFS vs DFS")
    print("="*65)

    # Cek CSV sudah ada
    for f in [NODES_CSV, EDGES_CSV]:
        if not os.path.exists(f):
            print(f"\n   File tidak ditemukan: {f}")
            return

    os.makedirs("output", exist_ok=True)

    print(f"\n   Memuat data dari CSV ...")
    NODES, EDGES_RAW, GEOM = load_data()
    print(f"   {len(NODES)} node, {len(EDGES_RAW)} edge, {len(GEOM)//2} polyline dimuat")

    graph = build_graph(NODES, EDGES_RAW)

    # Daftar node
    node_list = list(NODES.keys())
    print(f"\n  40 NODE SURABAYA:")
    for i, name in enumerate(node_list, 1):
        print(f"  {i:>2}. {name:<40} [{NODES[name]['type']}]")

    print(f"\n{'='*65}")
    print("INTERAKTIF: Masukkan nomor node ASAL dan TUJUAN untuk membandingkan algoritma")
    while True:
        s = input("\n  Nomor ASAL   (Enter untuk keluar): ").strip()
        if not s: break
        e = input("  Nomor TUJUAN : ").strip()
        try:
            start = node_list[int(s)-1]
            end   = node_list[int(e)-1]
            comp = compare_algorithms(
                graph,
                start,
                end
            )
            print_comparison(comp)
            animate_dijkstra_advanced(
                graph,
                NODES,
                GEOM,
                start,
                end
            )
            if input("  Simpan visualisasi? (y/n): ").strip().lower() == "y":
                plot_comparison(comp, graph, NODES, GEOM, save_path=f"output/interaktif_{s}_{e}.png")
        except (IndexError, ValueError) as ex:
            print(f"   Input tidak valid: {ex}")
        except KeyboardInterrupt:
            break

    print("\n   Selesai. Output tersimpan di: output/")

if __name__ == "__main__":
    main()