import osmnx as ox
import pandas as pd
import os
import math

PLACE      = "Surabaya, East Java, Indonesia"
NETWORK    = "drive"
OUT_DIR    = "data"
NODES_FILE = f"{OUT_DIR}/nodes_raw.csv"
EDGES_FILE = f"{OUT_DIR}/edges_raw.csv"

os.makedirs(OUT_DIR, exist_ok=True)

# Fetch
print(f"[1/3] Fetching OSMnx graph: {PLACE} ...")
G = ox.graph_from_place(PLACE, network_type=NETWORK, simplify=True)
print(f"      Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")

# Nodes 
print(f"[2/3] Menyimpan nodes → {NODES_FILE} ...")
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
nodes_df = nodes_gdf[["y", "x"]].reset_index()
nodes_df.columns = ["osmid", "lat", "lon"]
node_lookup = nodes_df.set_index("osmid")
nodes_df.to_csv(NODES_FILE, index=False)
print(f"      {len(nodes_df)} node tersimpan.")

# Edges + geometry
print(f"[3/3] Menyimpan edges + geometry → {EDGES_FILE} ...")
edges_gdf = edges_gdf.reset_index()

def parse_speed(val):
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return 40
        val = val[0]
    try:
        import numpy as np
        if isinstance(val, np.ndarray):
            if len(val) == 0:
                return 40
            val = val[0]
    except:
        pass
    if pd.isna(val):
        return 40
    try:
        text = str(val)
        text = text.split(";")[0]
        text = text.split()[0]
        speed = float(text)
        if speed <= 0:
            return 40
        return speed
    except:
        return 40

def parse_name(val):
    if isinstance(val, list): return val[0]
    return str(val) if not pd.isna(val) else "Unknown"

edges_gdf["speed_kmh"] = edges_gdf["maxspeed"].apply(parse_speed) if "maxspeed" in edges_gdf.columns else 40
edges_gdf["road_name"] = edges_gdf["name"].apply(parse_name)      if "name"     in edges_gdf.columns else "Unknown"
edges_gdf["dist_m"]    = edges_gdf["length"].round(1)

# TRAFFIC SIGNAL DETECTION
traffic_signal_nodes = set()

if "highway" in nodes_gdf.columns:

    signal_nodes = nodes_gdf[
        nodes_gdf["highway"] == "traffic_signals"]

    traffic_signal_nodes = set(signal_nodes.index)

# Delay traffic light
edges_gdf["traffic_light_delay_sec"] = edges_gdf["v"].apply(
    lambda x: 15 if x in traffic_signal_nodes else 0
)

# MANEUVER DETECTION Berdasarkan arah geometry jalan

def calculate_bearing(lat1, lon1, lat2, lon2):

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1)
        * math.cos(lat2)
        * math.cos(dlon)
    )
    bearing = math.degrees(
        math.atan2(x, y)
    )
    return (bearing + 360) % 360

def detect_maneuver(row):
    try:
        u = row["u"]
        v = row["v"]
        lat1 = node_lookup.loc[u, "lat"]
        lon1 = node_lookup.loc[u, "lon"]
        lat2 = node_lookup.loc[v, "lat"]
        lon2 = node_lookup.loc[v, "lon"]
        bearing = calculate_bearing(
            lat1,
            lon1,
            lat2,
            lon2
        )
        # kanan
        if 45 <= bearing < 135:
            return "right", 10
        # kiri
        elif 225 <= bearing < 315:
            return "left", 5
        # lurus
        else:
            return "straight", 3
    except:
        return "straight", 3

maneuver_data = edges_gdf.apply(detect_maneuver,axis=1,result_type="expand")
edges_gdf["maneuver"] = maneuver_data[0]
edges_gdf["maneuver_penalty_sec"] = maneuver_data[1]

# waktu dasar perjalanan
edges_gdf["travel_time_sec"] = (3.6 * edges_gdf["dist_m"]) / edges_gdf["speed_kmh"]

# total final weight
edges_gdf["intersection_delay_sec"] = edges_gdf.apply(
    lambda r:
        r["traffic_light_delay_sec"]
        if r["traffic_light_delay_sec"] > 0
        else r["maneuver_penalty_sec"],
    axis=1
)
edges_gdf["total_time_sec"] = (
    edges_gdf["travel_time_sec"]
    + edges_gdf["intersection_delay_sec"]
)

edges_gdf["oneway"]    = edges_gdf["oneway"].fillna(False)         if "oneway"   in edges_gdf.columns else False

# Simpan geometry
edges_gdf["geometry_wkt"] = edges_gdf["geometry"].apply(
    lambda g: g.wkt if g is not None else ""
)

out_cols = [
    "u",
    "v",
    "dist_m",
    "speed_kmh",
    "travel_time_sec",
    "traffic_light_delay_sec",
    "maneuver",
    "maneuver_penalty_sec",
    "intersection_delay_sec",
    "total_time_sec",
    "road_name",
    "oneway",
    "geometry_wkt"
]
edges_gdf[out_cols].to_csv(EDGES_FILE, index=False)
print(f"      {len(edges_gdf)} edge tersimpan (dengan geometry).")

print("\n Selesai!")
print(f"   {NODES_FILE}")
print(f"   {EDGES_FILE}")