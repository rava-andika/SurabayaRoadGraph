import pandas as pd
import numpy as np
import json, os, math, warnings
from collections import deque
from shapely import wkt as shapely_wkt
warnings.filterwarnings("ignore")

IN_NODES      = "data/nodes_raw.csv"
IN_EDGES      = "data/edges_raw.csv"
OUT_NODES     = "data/nodes.csv"
OUT_EDGES     = "data/edges.csv"
OUT_GEOM      = "data/edges_geometry.csv"

# 40 LANDMARK SURABAYA
LANDMARKS = [
    ("UNESA Ketintang",            -7.3136, 112.7241, "kampus"),
    ("ITS Sukolilo",               -7.2754, 112.7964, "kampus"),
    ("Universitas Airlangga",      -7.2828, 112.7810, "kampus"),
    ("Universitas Surabaya",       -7.3019, 112.7780, "kampus"),
    ("Universitas Negeri UM",      -7.2675, 112.7340, "kampus"),
    ("STTS Surabaya",              -7.2611, 112.7512, "kampus"),
    ("Tunjungan Plaza",            -7.2577, 112.7376, "mall"),
    ("Grand City Mall",            -7.2480, 112.7512, "mall"),
    ("Galaxy Mall",                -7.2913, 112.7805, "mall"),
    ("BG Junction",                -7.2518, 112.7305, "mall"),
    ("Ciputra World",              -7.2908, 112.7388, "mall"),
    ("Royal Plaza",                -7.3028, 112.7310, "mall"),
    ("Stasiun Gubeng",             -7.2653, 112.7522, "transportasi"),
    ("Stasiun Pasar Turi",         -7.2448, 112.7266, "transportasi"),
    ("Terminal Bungurasih",        -7.3564, 112.7387, "transportasi"),
    ("Terminal Wonokromo",         -7.3003, 112.7318, "transportasi"),
    ("Pelabuhan Tanjung Perak",    -7.2014, 112.7283, "transportasi"),
    ("RSUD Dr. Soetomo",           -7.2690, 112.7503, "rumahsakit"),
    ("RS Premier Surabaya",        -7.3025, 112.7658, "rumahsakit"),
    ("RS Siloam Surabaya",         -7.2610, 112.7411, "rumahsakit"),
    ("Balai Kota Surabaya",        -7.2492, 112.7376, "pemerintahan"),
    ("Tugu Pahlawan",              -7.2459, 112.7369, "landmark"),
    ("Masjid Al-Akbar",            -7.3322, 112.7291, "landmark"),
    ("Kebun Binatang Surabaya",    -7.2949, 112.7307, "wisata"),
    ("Monkasel Surabaya",          -7.2643, 112.7479, "wisata"),
    ("House of Sampoerna",         -7.2387, 112.7310, "wisata"),
    ("Simpang Jl. Ahmad Yani",     -7.3136, 112.7322, "persimpangan"),
    ("Simpang Wonokromo",          -7.3003, 112.7318, "persimpangan"),
    ("Simpang Darmo - Polda",      -7.2839, 112.7350, "persimpangan"),
    ("Simpang Urip Sumoharjo",     -7.2720, 112.7320, "persimpangan"),
    ("Simpang Basuki Rahmat",      -7.2622, 112.7388, "persimpangan"),
    ("Simpang Embong Malang",      -7.2590, 112.7340, "persimpangan"),
    ("Simpang Pemuda - Gubernur",  -7.2550, 112.7450, "persimpangan"),
    ("Bundaran Waru",              -7.3625, 112.7322, "persimpangan"),
    ("Simpang Kenjeran",           -7.2320, 112.7750, "persimpangan"),
    ("Simpang Jl. Diponegoro",     -7.2780, 112.7460, "persimpangan"),
    ("Simpang Raya Darmo",         -7.2910, 112.7360, "persimpangan"),
    ("Simpang Joyoboyo",           -7.3060, 112.7290, "persimpangan"),
    ("Simpang HR Muhammad",        -7.2915, 112.7550, "persimpangan"),
    ("Simpang Mulyosari",          -7.2550, 112.7850, "persimpangan"),
]

# HELPERS
def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2-lat1); dl = math.radians(lon2-lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def snap_to_nearest(lat, lon, nodes_df):
    df = nodes_df.copy()
    df["dist"] = df.apply(lambda r: haversine(lat, lon, r["lat"], r["lon"]), axis=1)
    return df.nsmallest(1, "dist").iloc[0]

def geometry_wkt_to_coords(wkt_str):
    """WKT LineString → list of [lon, lat]"""
    try:
        geom = shapely_wkt.loads(wkt_str)
        return [[round(x, 6), round(y, 6)] for x, y in geom.coords]
    except:
        return []

def bfs_path_with_geom(adj, src, dst, max_steps=10000):
    """BFS — kembalikan list edge dict (termasuk geometry coords)."""
    if src == dst: return []
    visited = {src}
    q = deque([(src, [])])
    steps = 0
    while q:
        cur, path = q.popleft()
        steps += 1
        if steps > max_steps: return None
        for e in adj.get(cur, []):
            nb = e["v"]
            if nb == dst:
                return path + [e]
            if nb not in visited:
                visited.add(nb)
                q.append((nb, path + [e]))
    return None

# LOAD RAW DATA
print("[1/5] Membaca data mentah OSMnx ...")
nodes_raw = pd.read_csv(IN_NODES)
edges_raw = pd.read_csv(IN_EDGES)
print(f"      {len(nodes_raw)} nodes, {len(edges_raw)} edges")

# SNAP LANDMARKS → OSMID
print("[2/5] Snap 40 landmark ke OSMnx node terdekat ...")
landmark_rows = []
osmid_to_name = {}

for (name, lat, lon, tipe) in LANDMARKS:
    nearest     = snap_to_nearest(lat, lon, nodes_raw)
    osmid       = int(nearest["osmid"])
    slat, slon  = float(nearest["lat"]), float(nearest["lon"])
    snap_d      = haversine(lat, lon, slat, slon)
    landmark_rows.append({"name": name, "lat": slat, "lon": slon,
                           "type": tipe, "osmid": osmid,
                           "snap_dist_m": round(snap_d, 1)})
    osmid_to_name[osmid] = name
    print(f"      {name:<35} → {osmid} ({snap_d:.0f} m)")

nodes_out = pd.DataFrame(landmark_rows)

# BANGUN ADJACENCY — SERTAKAN GEOMETRY COORDS
print("\n[3/5] Membangun adjacency list dengan geometry ...")
adj = {}   # osmid → list of {v, dist_m, speed_kmh, road_name, coords}

for _, row in edges_raw.iterrows():
    u   = int(row["u"]); v = int(row["v"])
    d   = float(row["dist_m"])
    spd = float(row["speed_kmh"]) if not pd.isna(row["speed_kmh"]) else 40.0
    rn  = str(row["road_name"])
    ow  = bool(row["oneway"]) if "oneway" in row else False

    # Parse geometry: WKT → [[lon,lat], ...]
    coords_fwd = geometry_wkt_to_coords(str(row.get("geometry_wkt", "")))
    coords_rev = list(reversed(coords_fwd))   # untuk arah balik

    edge_fwd = {
    "v": v,
    "dist_m": d,
    "speed_kmh": spd,
    "travel_time_sec": float(
        row.get("travel_time_sec", 0)),
    "traffic_light_delay_sec": float(
        row.get("traffic_light_delay_sec", 0)),
    "maneuver_penalty_sec": float(
        row.get("maneuver_penalty_sec", 0)),
    "intersection_delay_sec": float(
        row.get("intersection_delay_sec", 0)),
    "weight": float(
        row.get("total_time_sec", 0)),
    "road_name": rn,
    "coords": coords_fwd,
    "maneuver": str(
        row.get("maneuver", "straight")),
    }
    edge_rev = {
    "v": u,
    "dist_m": d,
    "speed_kmh": spd,
    "travel_time_sec": float(
        row.get("travel_time_sec", 0)),
    "traffic_light_delay_sec": float(
        row.get("traffic_light_delay_sec", 0)),
    "maneuver_penalty_sec": float(
        row.get("maneuver_penalty_sec", 0)),
    "intersection_delay_sec": float(
        row.get("intersection_delay_sec", 0)),
    "weight": float(
        row.get("total_time_sec", 0)),
    "road_name": rn,
    "coords": coords_rev,
    "maneuver": str(
        row.get("maneuver", "straight")),
}

    adj.setdefault(u, []).append(edge_fwd)
    if not ow:
        adj.setdefault(v, []).append(edge_rev)

print(f"      Adjacency siap: {len(adj)} OSM node")

# TEMUKAN EDGE ANTAR LANDMARK + KUMPULKAN GEOMETRY
print("\n[4/5] Mencari koneksi + geometry antar 40 landmark via BFS ...")
osmid_list = [r["osmid"] for r in landmark_rows]
name_list  = [r["name"]  for r in landmark_rows]

edge_rows  = []
geom_rows  = []
found = skipped = 0

for i, src in enumerate(osmid_list):
    for j, dst in enumerate(osmid_list[i+1:], start=i+1):
        path_edges = bfs_path_with_geom(adj, src, dst)
        if path_edges is None:
            skipped += 1; continue

        total_dist = sum(e["dist_m"] for e in path_edges)
        avg_speed  = float(np.mean([e["speed_kmh"] for e in path_edges])) if path_edges else 40.0
        road_name  = path_edges[0]["road_name"] if path_edges else "Unknown"

        # Filter jarak terlalu muter
        direct_m = haversine(
            nodes_out.iloc[i]["lat"], nodes_out.iloc[i]["lon"],
            nodes_out.iloc[j]["lat"], nodes_out.iloc[j]["lon"],
        )
        if total_dist > max(direct_m * 4, 500):
            skipped += 1; continue

        src_name = name_list[i]
        dst_name = name_list[j]
        weight = round(sum(e["weight"] for e in path_edges),2)

        edge_rows.append({
            "from": src_name, "to": dst_name,
            "dist_m": round(total_dist, 1), "speed_kmh": round(avg_speed, 1),
            "weight_sec": weight, "road_name": road_name,
            "travel_time_sec": round(sum(e["travel_time_sec"] for e in path_edges),1),
            "traffic_light_delay_sec": round(sum(e["traffic_light_delay_sec"] for e in path_edges),1),
            "maneuver_delay_sec": round(sum(e["maneuver_penalty_sec"] for e in path_edges),1),
            "intersection_delay_sec": round(sum(e["intersection_delay_sec"] for e in path_edges),1),
            "left_turn_count": sum(1 for e in path_edges if e.get("maneuver") == "left"),
            "right_turn_count": sum(1 for e in path_edges if e.get("maneuver") == "right"),
            "straight_count": sum(1 for e in path_edges if e.get("maneuver") == "straight"),
        })

        # Gabung semua koordinat sepanjang jalur menjadi satu polyline
        full_coords = []
        for e in path_edges:
            c = e.get("coords", [])
            if not c:
                continue
            # Hindari duplikasi titik sambungan antar segmen
            if full_coords and c[0] == full_coords[-1]:
                full_coords.extend(c[1:])
            else:
                full_coords.extend(c)

        geom_rows.append({
            "from":        src_name,
            "to":          dst_name,
            "coords_json": json.dumps(full_coords),
        })
        found += 1

print(f"      {found} edge ditemukan, {skipped} dilewati")

# SIMPAN OUTPUT
print(f"\n[5/5] Menyimpan ...")
nodes_out.to_csv(OUT_NODES, index=False)
pd.DataFrame(edge_rows).to_csv(OUT_EDGES, index=False)
pd.DataFrame(geom_rows).to_csv(OUT_GEOM,  index=False)

print(f"   {OUT_NODES}  ({len(nodes_out)} node)")
print(f"   {OUT_EDGES}  ({len(edge_rows)} edge)")
print(f"   {OUT_GEOM}   ({len(geom_rows)} polyline)")
print("\nTransformasi selesai!")