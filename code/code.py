import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from time import time
from collections import defaultdict
import math
import random
import datashader.bundling as bd
import json
from scipy.spatial import ConvexHull
from networkx.algorithms.community import greedy_modularity_communities
import gc


############################################################################################
# PARAMETERS
############################################################################################

d = 1.0  # edge weight parameter, weight = distance^d
random_state = 42  # for reproducibility

manual_thresholds = {
    "CollegeMessage_100_i": 0.1,
    # "InfoVis_100_i": 0.25,
    "school_network_converted_100_i": 50,
    "USAir97_100_i": 0.04,
}

# --- Trimming the Hairball config (edge cutting + fill) ---
TRIMMING_CFG = dict(
    enabled= True,                 # turn on/off
    strategy="information",       # "frequency" | "betweenness" | "information" | "random"
    target_edges_per_node=4.0,    # skeleton density target (≈ 2E/N)
    reconnect=True,               # keep graph connected while cutting
    betweenness_k=128,            # sample size for betweenness (None = exact, slower)
    seed=42
)

# --- Solar/Collapse config  ---
SOLAR_CFG = dict(
    enabled = True,               
    solar_strength = 0.2,
    solar_hops = 1,
    collapse_strength = 0.2,
    collapse_iters = 5,
    seed=42
)

# --- Virtual edges config  ---
VIRTUAL_EDGES_CFG = {
    "enabled": True,
    "K": 6,                 # consider only sun pairs (S,T) with dist(S,T) <= K_dist
    "strategy": "knn",          # "ring" or "knn" or "all" or "none"
    "knn_k": 1,                  # if strategy == "knn", use k in {1,2}
    "w_fake": 0.5,              # fixed weight for all virtual edges
    "tag": "virtual",            # attribute key to mark virtual edges
    "global_degree_cap": None,          # cap total virtual degree per node
    "max_group_size": 400,           # guard for huge groups
    "max_targets_per_S": None,          # optional: limit candidate T per S (None = all)
    # "treat_graph_as_undirected": True,   # compute distances on undirected view for reachability
    "undirected_for_virtuals": True,     # if DiGraph, also add reverse edge so FA2 spring is undirected
    "include_moons": True,
    "include_planets": True,
}

INFLUENCE_PATH_CFG = {
    "enabled": True,
    "k_per_pair": 3,      # Consider up to 3 paths between each Sun pair
    "top_k_global": 5,    # Show only the best 5 overall
    "weight_attr": "weight",
}

# --- Experiment config  ---
EXPERIMENT_CFG = {
    "sun_percentages": [0.02, 0.05, 0.10],
    "sun_methods": ["single_discount", "degree_discount", "random", "top_CI"],
    "min_spacing": 2,
    "seed": 42,
}

FILTERED_DATASETS = {"enron", "PGP", "Rmining"}

############################################################################################
# TRIMMING
############################################################################################

def _rank_edges_for_trimming(G, strategy="information", weight="weight", betweenness_k=None, seed=42):
    """
    Return a removal order (list of edges) from weakest/least-informative to strongest.
    - frequency:      remove by ascending weight
    - betweenness:    remove by DESCENDING edge betweenness (bridges first)
    - information:    remove by ascending PMI-like informativeness
    - random:         random order
    """
    rng = random.Random(seed)
    if G.number_of_edges() == 0:
        return []

    if strategy == "frequency":
        # Keep strong edges, remove weak ones first
        order = sorted(G.edges(data=True), key=lambda e: e[2].get(weight, 1.0))
        return [(u, v) for u, v, _ in order]

    if strategy == "betweenness":
        dist_attr = "distance_weight" if "distance_weight" in next(iter(G.edges(data=True)))[2] else None
        eb = nx.edge_betweenness_centrality(G, weight=dist_attr, k=betweenness_k, seed=seed)
        # Remove highest betweenness first
        order = sorted(eb.items(), key=lambda kv: kv[1], reverse=True)
        return [e for e, _ in order]

    if strategy == "information":
        # robust to arbitrary positive weights (co-occurrence-like)
        eps = 1e-12
        W = sum(d.get(weight, 1.0) for _, _, d in G.edges(data=True)) + eps
        s = {n: 0.0 for n in G.nodes()}
        for u, v, d in G.edges(data=True):
            w = d.get(weight, 1.0)
            s[u] += w
            s[v] += w
        pmi = {}
        for u, v, d in G.edges(data=True):
            w = d.get(weight, 1.0) + eps
            pmi[(u, v)] = np.log(w * W / ((s[u] + eps) * (s[v] + eps)))
        order = sorted(pmi.items(), key=lambda kv: kv[1])  # lowest informativeness first
        return [e for e, _ in order]

    if strategy == "random":
        E = list(G.edges())
        rng.shuffle(E)
        return E

    raise ValueError(f"Unknown trimming strategy: {strategy}")


def trim_and_fill(G, *, strategy="information", target_edges_per_node=4.0,
                  reconnect=True, betweenness_k=None, seed=42, weight="weight"):
    """
    Stage A (trim): remove edges down to target EPN, optionally preserving connectivity.
    Stage B (fill): Louvain-like greedy modularity on skeleton, then re-add *all original*
                    intra-community edges.
    Returns: (skeleton_undirected, filled_undirected, communities_list)
    """
    # Work on an undirected copy for visual clarity and connectivity tests
    G0 = G.to_undirected(as_view=False).copy()
    n = G0.number_of_nodes()
    if n <= 1:
        return G0.copy(), G0.copy(), [set(G0.nodes())]

    # Target number of edges from EPN: E_target ≈ target_epn * N / 2
    E_target = int(max(0, round(target_edges_per_node * n / 2.0)))
    E_now = G0.number_of_edges()
    if E_target >= E_now:
        # Already sparse enough
        skel = G0.copy()
    else:
        skel = G0.copy()
        order = _rank_edges_for_trimming(
            skel, strategy=strategy, weight=weight, betweenness_k=betweenness_k, seed=seed
        )
        for u, v in order:
            if skel.number_of_edges() <= E_target:
                break
            if not skel.has_edge(u, v):
                continue
            data_uv = G0[u][v]
            skel.remove_edge(u, v)
            if reconnect and not nx.is_connected(skel):
                # put it back; removing it would disconnect the skeleton
                skel.add_edge(u, v, **data_uv)

    # Communities on skeleton
    comms = list(greedy_modularity_communities(skel, weight=weight))
    # Stage B fill: re-add *original* intra-community edges
    filled = nx.Graph()
    filled.add_nodes_from(G0.nodes(data=True))
    for comm in comms:
        sub = G0.subgraph(comm)
        filled.add_edges_from((u, v, d) for u, v, d in sub.edges(data=True))

    return skel, filled, comms


############################################################################################
# FAKE EDGES
############################################################################################

def _bfs_distances_undirected(G, source, cutoff):
    """Hop distances on an undirected view (prevents direction gaps)."""
    H = G.to_undirected(as_view=True)
    return nx.single_source_shortest_path_length(H, source, cutoff=cutoff)

def assign_unique_sun_owners(G, suns, cutoff=None):
    """Graph Voronoi: assign each node to exactly one sun."""
    H = G.to_undirected(as_view=True)
    # distances from each sun
    dist = {S: nx.single_source_shortest_path_length(H, S, cutoff=cutoff) for S in suns}
    owners = {}
    for n in G.nodes():
        # collect (d, S) pairs where n is reachable from S
        cand = [(dist[S].get(n, np.inf), S) for S in suns]
        dmin = min(d for d, _ in cand)
        if not np.isfinite(dmin):
            continue  # unreachable from all suns 
        # tie-break: by dmin, then by string of S for stability
        best = sorted([ (d, S) for d, S in cand if d == dmin ], key=lambda t: (t[0], str(t[1])))[0][1]
        owners[n] = best
    return owners

def _neighbors_planets_owned(G, S, pos, node_owner):
    if G.is_directed():
        nbrs = set(G.successors(S)) | set(G.predecessors(S))
    else:
        nbrs = set(G.neighbors(S))
    return [u for u in nbrs if u in pos and node_owner.get(u) == S]

def _moons_owned(G, S, pos, owners, planets=None):
    """
    Moons = depth-2 neighbors of S (neighbors of planets), filtered to:
      - present in pos,
      - owner == S,
      - not S and not a planet.
    """
    if planets is None:
        planets = _neighbors_planets_owned(G, S, pos, owners)

    moons = set()
    for p in planets:
        if G.is_directed():
            nbrs = set(G.successors(p)) | set(G.predecessors(p))
        else:
            nbrs = set(G.neighbors(p))
        for m in nbrs:
            if m == S or m in planets:
                continue
            if m in pos and owners.get(m) == S:
                moons.add(m)
    return list(moons)

def _normalize_uv(u, v):
    return (u, v) if u <= v else (v, u)

def _is_real_edge(G, a, b, virtual_tag):
    """Return True if (a,b) or (b,a) exists as a NON-virtual edge in G."""
    if G.has_edge(a, b) and not G[a][b].get(virtual_tag, False):
        return True
    if G.is_directed() and G.has_edge(b, a) and not G[b][a].get(virtual_tag, False):
        return True
    return False

def gather_virtual_and_paths_for_viz(
    G, pos, suns, owners, cfg, role_sets,
    virtual_tag="virtual",
    include_moons_for_viz=False
):
    """
    Keep for viz:
      - virtual_pairs (tagged)
      - planet/moon → T* shortest-path pairs
      - REAL edges on sun↔sun shortest paths (S to T* within K and top max_targets_per_S),
        BUT **only** those whose **both endpoints** are role nodes (sun/planet/moon).
    """
    K               = cfg.get("K", 6)
    max_T_per_S     = cfg.get("max_targets_per_S", 3)
    include_planets = cfg.get("include_planets", True)

    H = G.to_undirected(as_view=True)
    dist_from_T = {T: nx.single_source_shortest_path_length(H, T, cutoff=K) for T in suns}

    def nearest_Ts_for_S(S):
        cand = [(dist_from_T[T].get(S, np.inf), T) for T in suns if T != S]
        cand = [(d, T) for (d, T) in cand if np.isfinite(d) and d <= K]
        cand.sort(key=lambda dt: (dt[0], str(dt[1])))
        if max_T_per_S is None:
            return [T for _, T in cand]
        return [T for _, T in cand[:max_T_per_S]]

    # --- virtual edges currently in graph
    virtual_pairs = set()
    for u, v, d in G.edges(data=True):
        if d.get(virtual_tag, False) and u in pos and v in pos:
            virtual_pairs.add(_normalize_uv(u, v))

    # --- planet/moon shortest paths to assigned T*
    planet_moon_path_pairs = set()
    if include_planets or include_moons_for_viz:
        for S in suns:
            if S not in pos:
                continue
            planets = _neighbors_planets_owned(G, S, pos, owners) if include_planets else []
            moons   = _moons_owned(G, S, pos, owners, planets=planets) if include_moons_for_viz else []

            Ts = nearest_Ts_for_S(S)
            if not Ts:
                continue
            distS = {T: dist_from_T[T].get(S, np.inf) for T in Ts}

            def _assign_and_add(nodes, max_du):
                for u in nodes:
                    best, best_du = None, np.inf
                    for T in Ts:
                        du = dist_from_T[T].get(u, np.inf)
                        if not np.isfinite(du) or distS[T] > K or du > max_du:
                            continue
                        if (du < best_du) or (du == best_du and (distS[T], str(T)) <
                                              (distS.get(best, np.inf), str(best))):
                            best, best_du = T, du
                    if best is None:
                        continue
                    try:
                        p = nx.shortest_path(H, source=u, target=best)
                        for a, b in zip(p, p[1:]):
                            if a in pos and b in pos:
                                planet_moon_path_pairs.add(_normalize_uv(a, b))
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass

            if include_planets:
                _assign_and_add(planets, K - 1)
            if include_moons_for_viz:
                _assign_and_add(moons, max(0, K - 2))

    # --- real edges on sun-sun shortest paths
    role_nodes = set(role_sets.get("suns", set())) \
               | set(role_sets.get("planets", set())) \
               | set(role_sets.get("moons", set()))
    sunpath_real_pairs = set()

    for S in suns:
        if S not in pos:
            continue
        Ts = nearest_Ts_for_S(S)
        for T in Ts:
            if T not in pos:
                continue
            try:
                p = nx.shortest_path(H, source=S, target=T)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            for a, b in zip(p, p[1:]):
                if a not in pos or b not in pos:
                    continue
                # keep only if both endpoints are role nodes and edge exists as real
                if a in role_nodes and b in role_nodes and _is_real_edge(G, a, b, virtual_tag):
                    sunpath_real_pairs.add(_normalize_uv(a, b))

    keep_pairs = virtual_pairs | planet_moon_path_pairs | sunpath_real_pairs
    keep_nodes = set(x for uv in keep_pairs for x in uv)

    return {
        "virtual_pairs": virtual_pairs,
        "path_pairs": planet_moon_path_pairs,
        "sunpath_real_pairs": sunpath_real_pairs,
        "keep_pairs": keep_pairs,
        "keep_nodes": keep_nodes,
    }


def compute_role_sets(G, pos, suns, owners, include_moons=True):
    """
    Returns dict with 'suns', 'planets', 'moons' as sets.
    Ownership is unique; planets = depth-1 owned neighbors, moons = depth-2 owned neighbors.
    Only nodes present in `pos` are included.
    """
    sun_set = {s for s in suns if s in pos}
    planet_set, moon_set = set(), set()

    for S in suns:
        if S not in pos:
            continue
        pls = _neighbors_planets_owned(G, S, pos, owners)
        planet_set.update(pls)
        if include_moons:
            mns = _moons_owned(G, S, pos, owners, planets=pls)
            moon_set.update(mns)

    # Safety: remove any sun from planet/moon sets if overlap
    planet_set -= sun_set
    moon_set   -= sun_set
    # Also remove planets from moons if anything overlapped 
    moon_set   -= planet_set

    return {"suns": sun_set, "planets": planet_set, "moons": moon_set}


def plot_only_virtuals_and_planet_paths(
    pos, G, dataset_name, label, output_folder,
    keep_info, highlight_suns=None,
    role_sets=None,           
    edge_lw=1.0, edge_alpha=0.6
):
    os.makedirs(output_folder, exist_ok=True)

    virtual_pairs = keep_info["virtual_pairs"]
    path_pairs    = keep_info["path_pairs"]
    keep_pairs    = keep_info["keep_pairs"]
    keep_nodes    = [n for n in keep_info["keep_nodes"] if n in pos]

    if not keep_pairs:
        print(f"[viz] Nothing to draw for {dataset_name} / {label} (no kept edges).")
        return

    # Default role sets if not provided
    if role_sets is None:
        role_sets = {"suns": set(), "planets": set(), "moons": set()}

    suns   = [n for n in keep_nodes if n in role_sets.get("suns", set())]
    planets= [n for n in keep_nodes if n in role_sets.get("planets", set())]
    moons  = [n for n in keep_nodes if n in role_sets.get("moons", set())]
    others = [n for n in keep_nodes if n not in role_sets.get("suns", set())
                                     and n not in role_sets.get("planets", set())
                                     and n not in role_sets.get("moons", set())]

    plt.figure(figsize=(10, 10))

    # Draw edges first
    for u, v in path_pairs:
        plt.plot([pos[u][0], pos[v][0]],[pos[u][1], pos[v][1]],
                 linewidth=edge_lw, alpha=edge_alpha, color="gray", zorder=1)

    for u, v in virtual_pairs:
        plt.plot([pos[u][0], pos[v][0]],[pos[u][1], pos[v][1]],
                 linestyle="-", linewidth=edge_lw, alpha=edge_alpha, color="orange", zorder=2)

    # Draw nodes by role (moons under, then planets, then suns on top)
    def _scatter(nodes, s, c, z):
        if nodes:
            xs = [pos[n][0] for n in nodes]; ys = [pos[n][1] for n in nodes]
            plt.scatter(xs, ys, s=s, c=c, alpha=0.95, zorder=z)

    _scatter(moons,   16, "mediumseagreen", 3)
    _scatter(planets, 18, "steelblue",      4)
    _scatter(others,  12, "lightgray",      4)
    if suns:
        xs = [pos[s][0] for s in suns]; ys = [pos[s][1] for s in suns]
        plt.scatter(xs, ys, s=80, c="crimson", edgecolors="yellow", linewidths=1.6, zorder=5)

    plt.title(f"{label} (Virtual + Planet shortest paths only)")
    plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout()
    out = f"{output_folder}/{dataset_name}_{label}_ONLY_virtuals_and_planet_paths.png"
    plt.savefig(out, dpi=200); plt.close()
    print(f"[viz] Saved {out}")



def add_virtual_edges(G, pos, suns, cfg=VIRTUAL_EDGES_CFG, owners=None):
    """
    For each sun S: assign each planet u of S to its nearest other sun T* (w.r.t. hop distance),
    under constraints dist(S,T*) <= K and dist(u,T*) <= K-1. Within each (S, T*) group,
    add sparse temporary virtual edges (ring or kNN). Mutates G in-place.
    Returns: (added_edges_list, stats_dict)
    """
    if not cfg.get("enabled", True):
        return [], {"added": 0}

    K           = cfg.get("K", 6)         
    strategy    = cfg.get("strategy", "ring")
    knn_k       = cfg.get("knn_k", 1)
    w_fake      = cfg.get("w_fake", 0.05)
    tag            = cfg.get("tag", "virtual")
    cap_global     = cfg.get("global_degree_cap", 3)
    max_group      = cfg.get("max_group_size", 400)
    max_T_per_S    = cfg.get("max_targets_per_S", 3)
    mirror_virtual = cfg.get("undirected_for_virtuals", True)
    include_moons  = cfg.get("include_moons", True)   
    include_planets= cfg.get("include_planets", True)

    # --- ownership map (unique sun per node) ---
    if owners is None:
        owners = assign_unique_sun_owners(G, suns, cutoff=None)

    #  1) Precompute distances from each target sun T to all nodes, cutoff=K
    dist_from_T = {T: _bfs_distances_undirected(G, T, cutoff=K) for T in suns}

    def nearest_Ts_for_S(S):
        cand = [(dist_from_T[T].get(S, np.inf), T) for T in suns if T != S]
        cand = [(d, T) for (d, T) in cand if np.isfinite(d) and d <= K]
        cand.sort(key=lambda dt: (dt[0], str(dt[1])))
        if max_T_per_S is None: 
            return [T for _, T in cand]
        return [T for _, T in cand[:max_T_per_S]]

    added = []
    virt_deg = defaultdict(int) 
    groups_touched = 0
    seen_pairs = set()           

    # 2) Per sun S: assign each planet to its nearest eligible T*
    for S in suns:
        if S not in pos:
            continue

        # --- owned layers ---
        planets = _neighbors_planets_owned(G, S, pos, owners) if include_planets else []
        moons   = _moons_owned(G, S, pos, owners, planets=planets) if include_moons else []
        moon_limit = max(0, K - 2)

        if not planets and not moons:
            continue

        Ts = nearest_Ts_for_S(S)
        if not Ts:
            continue

        # Build a list of (T, dist_S_T) once for tie-breaking
        distS = {T: dist_from_T[T].get(S, np.inf) for T in Ts}

        groups = defaultdict(list)  # T* -> list of (node, kind) where kind in {"planet","moon"}

        # planets (≤ K-1)
        for u in planets:
            best, best_du = None, np.inf
            for T in Ts:
                du = dist_from_T[T].get(u, np.inf)
                if not np.isfinite(du) or distS[T] > K or du > K - 1:
                    continue
                if (du < best_du) or (du == best_du and (distS[T], str(T)) < (distS.get(best, np.inf), str(best))):
                    best, best_du = T, du
            if best is not None:
                groups[best].append((u, "planet"))

        # moons (≤ K-2)
        if include_moons:
            moon_limit = max(0, K - 2)
            for u in moons:
                best, best_du = None, np.inf
                for T in Ts:
                    du = dist_from_T[T].get(u, np.inf)
                    if not np.isfinite(du) or distS[T] > K or du > moon_limit:
                        continue
                    if (du < best_du) or (du == best_du and (distS[T], str(T)) < (distS.get(best, np.inf), str(best))):
                        best, best_du = T, du
                if best is not None:
                    groups[best].append((u, "moon"))

        # helper: generate pairs from a *same-type* node list using your strategy
        def _pairs_from_nodes(nodes):
            if len(nodes) < 2:
                return []
            if strategy == "ring":
                sxy = np.array(pos[S])
                order = sorted(nodes, key=lambda n: math.atan2(pos[n][1]-sxy[1], pos[n][0]-sxy[0]))
                return [(order[i], order[(i+1) % len(order)]) for i in range(len(order))]
            elif strategy == "knn":
                P = np.array([pos[n] for n in nodes], float)
                pairs = []
                for i, u in enumerate(nodes):
                    dists = np.linalg.norm(P - P[i], axis=1)
                    idx = np.argsort(dists)[1:1+max(1, knn_k)]
                    for j in idx:
                        pairs.append((nodes[i], nodes[j]))
                return pairs
            elif strategy == "all":
                nodes2 = list(nodes)
                P = np.array([pos[n] for n in nodes2], float)
                triples = []
                for i, u in enumerate(nodes2):
                    for j in range(i+1, len(nodes2)):
                        v = nodes2[j]
                        triples.append((float(np.linalg.norm(P[i] - P[j])), u, v))
                triples.sort(key=lambda t: t[0])
                return [(u, v) for _, u, v in triples]
            elif strategy == "none":
                return []
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        # --- within each (S, T*) group: build same-type pairs only, then add with caps
        for Tstar, items in groups.items():
            if len(items) < 2:
                continue
            if len(items) > max_group:
                items = items[:max_group]

            planets_in = [n for (n, k) in items if k == "planet"]
            moons_in   = [n for (n, k) in items if k == "moon"]

            candidate_pairs = []
            candidate_pairs += _pairs_from_nodes(planets_in)  # planets - planets only
            candidate_pairs += _pairs_from_nodes(moons_in)    # moons - moons only

            for (u, v) in candidate_pairs:
                if u == v:
                    continue
                key = tuple(sorted((u, v)))
                if key in seen_pairs:
                    continue
                if G.has_edge(u, v):
                    continue
                if cap_global is not None and (virt_deg[u] >= cap_global or virt_deg[v] >= cap_global):
                    continue

                G.add_edge(u, v, weight=w_fake, **{tag: True})
                if G.is_directed() and mirror_virtual and not G.has_edge(v, u):
                    G.add_edge(v, u, weight=w_fake, **{tag: True})

                added.append((u, v))
                virt_deg[u] += 1
                virt_deg[v] += 1
                seen_pairs.add(key)


    stats = {
        "added": len(added),
        "groups_touched": groups_touched,
        "K": K,
        "strategy": strategy,
        "knn_k": knn_k,
        "w_fake": w_fake,
        "cap_global": cap_global,
        "include_planets": include_planets,
        "include_moons": include_moons,
    }
    return added, stats

def remove_virtual_edges(G, tag="virtual"):
    to_remove = [(u, v) for u, v, d in G.edges(data=True) if d.get(tag, False)]
    G.remove_edges_from(to_remove)
    return len(to_remove)


############################################################################################
# FORCE-DIRECTED LAYOUT FUNCTIONS
############################################################################################


def calculate_neighborhood_preservation(graph_distances, layout_distances, k=5):
    """
    Calculate neighborhood preservation of a reduced graph layout.

    Parameters:
    - graph_distances: Pairwise distances in the original graph.
    - layout_distances: Pairwise Euclidean distances in the reduced layout.
    - k: Number of nearest neighbors to compare.

    Returns:
    - Neighborhood preservation score (between 0 and 1).
    """
    # Get k-nearest neighbors for both spaces
    graph_neighbors = np.argsort(graph_distances, axis=1)[:, 1:k+1]
    layout_neighbors = np.argsort(layout_distances, axis=1)[:, 1:k+1]

    # Calculate the neighborhood preservation score
    total_overlap = 0
    for i in range(graph_neighbors.shape[0]):
        total_overlap += len(set(graph_neighbors[i]) & set(layout_neighbors[i]))

    # Normalize by total number of neighbors
    score = total_overlap / (k * graph_distances.shape[0])
    return score


def analyze_tsne_output(tsne_output, dataset_name):
    """
    Analyze t-SNE output for potential issues.
    """
    print(f"Analyzing t-SNE Output for {dataset_name}")
    
    # Convert dictionary to a numpy array
    if isinstance(tsne_output, dict):
        tsne_output = np.array(list(tsne_output.values()))

    # Calculate pairwise distances
    distances = pairwise_distances(tsne_output)
    distances[range(distances.shape[0]), range(distances.shape[0])] = np.inf  # Ignore diagonal
    finite_distances = distances[np.isfinite(distances)]
    
    print(f"Min Distance: {finite_distances.min():.4f}")
    print(f"Mean Distance: {finite_distances.mean():.4f}")
    print(f"Max Distance: {finite_distances.max():.4f}")
    print(f"Number of Near-Zeros (<1e-5): {(finite_distances < 1e-5).sum()}")
    print(f"Number of NaN Distances: {np.isnan(distances).sum()}")
    print("-" * 50)

def add_jitter(tsne_output, epsilon=0.5):
    """
    Add small random noise to t-SNE output to separate overlapping points.
    """
    return tsne_output + np.random.uniform(-epsilon, epsilon, tsne_output.shape)

def stabilize_distances(tsne_output, epsilon=1e-5):
    """
    Stabilize pairwise distances by replacing zeros with a small value.
    """
    distances = pairwise_distances(tsne_output)
    distances = np.maximum(distances, epsilon)  # Replace zeros with epsilon
    return distances

def normalize_tsne_output(tsne_output):
    """
    Normalize the t-SNE output to ensure consistent scaling.
    """
    tsne_output -= np.mean(tsne_output, axis=0)  # Center at origin
    tsne_output /= np.linalg.norm(tsne_output, axis=0).max()  # Scale to unit length
    return tsne_output


def find_optimal_clusters(data, max_k=10):
    """
    Finds the optimal number of clusters by maximizing the silhouette score.

    Parameters:
    - data: Array-like, shape (n_samples, n_features). Data points in the reduced space.
    - max_k: Maximum number of clusters to test.

    Returns:
    - best_k: The optimal number of clusters.
    - best_score: The silhouette score for the best_k.
    """
    best_score = -1
    best_k = 2
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        score = silhouette_score(data, cluster_labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, best_score

def calculate_node_overlap(reduced_positions, min_dist=0.01):
    """
    Count the number of node pairs that are closer than a threshold.
    """
    overlap_count = 0
    positions = list(reduced_positions.values())
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i] - positions[j]) < min_dist:
                overlap_count += 1
    return overlap_count

def calculate_area_utilization(reduced_positions):
    """
    Compute the ratio of convex hull area to bounding box area.
    """
    points = np.array(list(reduced_positions.values()))
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    bbox_area = (max_x - min_x) * (max_y - min_y)

    if points.shape[0] < 3:
        return 0  # Convex hull requires at least 3 points

    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # 2D volume = area
        utilization = hull_area / bbox_area if bbox_area > 0 else 0
        return utilization
    except Exception:
        return 0  # fallback if hull fails (e.g., all points colinear)
    
def edges_intersect(p1, q1, p2, q2):
    """Check if two line segments (p1-q1 and p2-q2) intersect."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return ccw(p1, p2, q2) != ccw(q1, p2, q2) and ccw(p1, q1, p2) != ccw(p1, q1, q2)

def count_edge_crossings(reduced_positions, graph):
    """
    Count how many edges in the layout intersect.
    """
    edge_list = list(graph.edges())
    positions = reduced_positions
    crossings = 0

    for i in range(len(edge_list)):
        u1, v1 = edge_list[i]
        if u1 not in positions or v1 not in positions:
            continue
        p1, q1 = positions[u1], positions[v1]
        for j in range(i + 1, len(edge_list)):
            u2, v2 = edge_list[j]
            # skip if edges share a node
            if len({u1, v1, u2, v2}) < 4:
                continue
            if u2 not in positions or v2 not in positions:
                continue
            p2, q2 = positions[u2], positions[v2]
            if edges_intersect(p1, q1, p2, q2):
                crossings += 1
    return crossings

def dijkstra_all_pairs_matrix_cutoff(graph, weight="weight", cutoff=None):
    """
    Computes all-pairs shortest paths using Dijkstra's algorithm with an optional cutoff.

    Parameters:
    - graph (nx.Graph): The input graph.
    - weight (str): The edge attribute to use as weight.
    - cutoff (float or int): Maximum path length to consider. Paths longer than this are ignored.

    Returns:
    - np.ndarray: Dense (n x n) matrix of shortest path distances.
    """
    nodes = list(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    dist_matrix = np.full((n, n), np.inf)

    for u in nodes:
        dist_dict = nx.single_source_dijkstra_path_length(graph, source=u, weight=weight, cutoff=cutoff)
        i = index[u]
        for v, d in dist_dict.items():
            j = index[v]
            dist_matrix[i, j] = d

    return dist_matrix

def compare_pagerank_and_degree(graph, pagerank_scores):
    data = []
    for node in graph.nodes():
        degree = graph.degree(node)
        pagerank = pagerank_scores.get(node, 0)
        data.append({"Node": node, "Degree": degree, "PageRank": pagerank})
    
    df = pd.DataFrame(data)
    df["Degree_norm"] = (df["Degree"] - df["Degree"].min()) / (df["Degree"].max() - df["Degree"].min())
    df["PageRank_norm"] = (df["PageRank"] - df["PageRank"].min()) / (df["PageRank"].max() - df["PageRank"].min())
    
    return df.sort_values(by="PageRank", ascending=False)



############################################################################################
# EDGE-PATH BUNDLING FUNCTIONS
############################################################################################

def _euclid_len(pos, u, v):
    a = np.asarray(pos[u], float)
    b = np.asarray(pos[v], float)
    return float(np.linalg.norm(a - b))

def _subdivide_midpoints(points, times=2):
    """Paper-style smoothing: recursively add midpoints."""
    pts = [np.asarray(p, float) for p in points]
    for _ in range(max(0, int(times))):
        out = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            mid = 0.5*(a + b)
            out.extend([mid, b])
        pts = out
    return [tuple(p) for p in pts]

def edge_path_bundling_epb(G, pos, k=2.0, d=2, smoothing=2):
    """
    Faithful EPB (Wallinger et al., TVCG'22).
    - G: (Di)Graph with nodes in 'pos'
    - pos: dict layout coordinates
    - k: max distortion factor
    - d: edge weight exponent
    - smoothing: recursive midpoint insertion
    Returns: control_point_lists, stats
    """
    directed = G.is_directed()
    # Precompute Euclidean length and EPB weight = len^d
    epb_len = {}
    epb_w = {}
    for u, v in G.edges():
        L = _euclid_len(pos, u, v)
        epb_len[(u, v)] = L
        epb_w[(u, v)] = (L ** d)
        if not directed:
            epb_len[(v, u)] = L
            epb_w[(v, u)] = (L ** d)

    # Sort edges by length descending (long first)
    sorted_edges = sorted(G.edges(), key=lambda e: epb_len[(e[0], e[1])], reverse=True)

    # skip/lock maps
    skip = {e: False for e in G.edges()}
    lock = {e: False for e in G.edges()}
    if not directed:
        skip.update({(v, u): False for u, v in G.edges()})
        lock.update({(v, u): False for u, v in G.edges()})

    def weight_fn(u, v, edata):
        # exclude only edges in 'skip'; allow locked edges to participate in paths
        return math.inf if skip.get((u, v), False) else epb_w[(u, v)]

    control_point_lists = []
    bundled = skipped = nopath = 0

    for (s, t) in tqdm(sorted_edges, desc="EPB"):
        # respect locks: edges on previous control paths are not re-bundled
        if lock.get((s, t), False):
            continue

        # temporarily exclude the current edge from Dijkstra
        skip[(s, t)] = True
        if not directed:
            skip[(t, s)] = True

        # shortest path by Euclidean^d
        try:
            path_nodes = nx.shortest_path(G, source=s, target=t, weight=weight_fn)
        except nx.NetworkXNoPath:
            path_nodes = None

        if not path_nodes or len(path_nodes) < 2:
            # reinstate the edge; nothing bundled
            skip[(s, t)] = False
            if not directed: skip[(t, s)] = False
            nopath += 1
            continue

        # path length in Euclidean (sum over path edges)
        path_len = 0.0
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            path_len += epb_len[(u, v)]
        straight_len = _euclid_len(pos, s, t)

        if path_len > k * straight_len:
            # too much detour; don't bundle this edge; put it back into shortest paths
            skip[(s, t)] = False
            if not directed: skip[(t, s)] = False
            skipped += 1
            continue

        # lock the edges along the path so they won't be bundled later
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            lock[(u, v)] = True
            if not directed: lock[(v, u)] = True

        # control points are the coordinates of the nodes on the path
        ctrl = [tuple(pos[n]) for n in path_nodes]
        ctrl = _subdivide_midpoints(ctrl, times=smoothing)
        control_point_lists.append(ctrl)

        bundled += 1

    stats = {
        "total_edges": G.number_of_edges(),
        "bundled_edges": bundled,
        "too_long": skipped,
        "no_path": nopath
    }
    print(f"EPB done: {bundled} bundled, {skipped} too-long, {nopath} no-path.")
    return control_point_lists, stats


############################################################################################
# PIPELINE FUNCTIONS
############################################################################################

def compute_node_stress(positions, graph_distances):
    """
    Compute stress per node
    Returns: dict {node: stress}
    """
    nodes = list(positions.keys())
    idx = {n: i for i, n in enumerate(nodes)}

    max_graph = np.max(graph_distances)
    if max_graph == 0:
        max_graph = 1.0

    stress = {}

    for v in nodes:
        iv = idx[v]
        pos_v = positions[v]

        diffs = np.linalg.norm(
            np.array([positions[u] for u in nodes]) - pos_v,
            axis=1
        )

        diffs_norm = diffs / np.max(diffs) if np.max(diffs) > 0 else diffs
        graph_d_norm = graph_distances[iv] / max_graph

        stress[v] = np.sum((diffs_norm - graph_d_norm) ** 2)

    return stress

def gini_coefficient(values):
    """
    Compute Gini coefficient of a list or array of non-negative values.
    """
    x = np.array(values, dtype=np.float64)
    if np.allclose(x, 0):
        return 0.0

    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)

    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def filter_edge_dataframe(df, frac=0.3, min_weight=None, random_state=42):
    """
    Filters the edge DataFrame by sampling a fraction and optionally applying a minimum weight threshold.

    Parameters:
    - df (pd.DataFrame): DataFrame with columns ['Source', 'Target', 'Weight']
    - frac (float): Fraction of edges to randomly sample
    - min_weight (float or None): If set, only keep edges with weight >= min_weight
    - random_state (int): Seed for reproducibility

    Returns:
    - pd.DataFrame: Filtered and sampled edge list
    """
    if min_weight is not None:
        df = df[df["Weight"] >= min_weight]
    df_sampled = df.sample(frac=frac, random_state=random_state)
    return df_sampled

def load_clean_edge_list(filepath, source_col="Source", target_col="Target", weight_col="Weight", min_weight=None):
    """
    Load and clean an edge list CSV file, guessing delimiter, dropping invalid rows,
    and optionally filtering by weight.

    Parameters:
    - filepath (str): Path to the CSV file.
    - source_col (str): Column name for source node.
    - target_col (str): Column name for target node.
    - weight_col (str): Column name for edge weight.
    - min_weight (float or None): If set, filters edges with weight below this.

    Returns:
    - pd.DataFrame: Cleaned edge list
    """
    # Try both comma and semicolon delimiters
    try:
        df = pd.read_csv(filepath, delimiter=',')
        if len(df.columns) <= 1:
            df = pd.read_csv(filepath, delimiter=';')
    except Exception as e:
        raise ValueError(f"Failed to read CSV with both ',' and ';' delimiters: {e}")

    # Drop rows with missing nodes
    df[source_col] = df[source_col].astype(str).str.strip()
    df[target_col] = df[target_col].astype(str).str.strip()
    df = df.dropna(subset=[source_col, target_col])

    # Filter by minimum weight if applicable
    if min_weight is not None and weight_col in df.columns:
        df = df[df[weight_col] >= min_weight]

    return df

def filter_low_degree_nodes(graph, min_degree=1):
    """
    Removes nodes with degree less than or equal to a specified threshold.

    Parameters:
    - graph (nx.Graph): Input graph.
    - min_degree (int): Minimum degree a node must have to be retained.

    Returns:
    - nx.Graph: Filtered graph.
    """
    filtered_graph = graph.copy()
    low_degree_nodes = [node for node, degree in dict(graph.degree()).items() if degree < min_degree]
    filtered_graph.remove_nodes_from(low_degree_nodes)
    return filtered_graph


def find_top_influence_paths_between_suns(
    G,
    suns,
    influence_attr="weight",
    k_per_pair=3,
    top_k_global=5,
):
    """
    For each ordered Sun pair, find the k_per_pair most influential simple paths.
    Path cost = 1 / influence (positive), so Dijkstra can be used safely.
    """
    print("[DEBUG] Entered find_top_influence_paths_between_suns")
    print("[DEBUG] G nodes:", G.number_of_nodes())
    print("[DEBUG] Missing Suns at entry:", set(suns) - set(G.nodes()))


    results = []

    for i, s1 in enumerate(suns):
        for s2 in suns[i+1:]:
            if s1 == s2:
                continue

            # Build a positive-cost graph H
            H = nx.DiGraph() if G.is_directed() else nx.Graph()
            H.add_nodes_from(G.nodes())

            for u, v, d in G.edges(data=True):
                inf = float(d.get(influence_attr, 0.0))
                cost = 1e6 if inf <= 0 else 1.0 / inf
                H.add_edge(u, v, cost=cost, influence=inf)

            # Try generating shortest simple paths
            try:
                gen = nx.shortest_simple_paths(H, s1, s2, weight="cost")
            except (nx.NetworkXNoPath, nx.NodeNotFound, ValueError):
                continue

            # Collect top-k paths for this pair
            try:
                for j, path in enumerate(gen):
                    if j >= k_per_pair:
                        break

                    total_inf = 1.0
                    for a, b in zip(path[:-1], path[1:]):
                        total_inf *= H[a][b]["influence"]

                    results.append({
                        "pair": (s1, s2),
                        "path": path,
                        "total_influence": total_inf
                    })

            except nx.NetworkXNoPath:
                continue

    # Sort all collected paths globally
    results.sort(key=lambda x: x["total_influence"], reverse=True)
    return results[:top_k_global]


def select_spaced_suns(G, ranked_nodes, k, min_distance=2):
    """
    Select spaced Suns from a pre-ranked node list.
    Highest-ranked nodes are considered first, but each new Sun
    must be at least `min_distance` hops away from all already chosen Suns.

    Parameters
    ----------
    G : nx.Graph
        Graph on which spacing is enforced.
    ranked_nodes : list
        Nodes sorted from highest priority to lowest (no scores needed).
    k : int
        Number of Suns to select.
    min_distance : int
        Minimum allowed shortest-path distance between any two Suns.

    Returns
    -------
    suns : list
        Final spaced selection of Suns.
    """

    suns = []

    for node in ranked_nodes:
        # Check spacing constraint against already selected Suns
        too_close = False
        for s in suns:
            if nx.has_path(G, node, s):
                d = nx.shortest_path_length(G, node, s)
                if d < min_distance:
                    too_close = True
                    break

        if not too_close:
            suns.append(node)

        # Stop when we have enough Suns
        if len(suns) >= k:
            break

    print(f"[INFO] Selected {len(suns)} spaced Suns (requested {k}, min_distance={min_distance})")

    return suns

def pull_neighbors_toward_source(pos, graph, suns, strength=0.2, hops=1):
    """
    Slightly contracts the positions of nodes around each sun node.

    Parameters:
    - pos (dict): Original node positions {node: (x, y)}.
    - graph (nx.Graph): The graph structure.
    - suns (list): List of sun (high-CI) nodes.
    - strength (float): Fraction by which to pull neighbors toward the sun.
    - hops (int): Radius around each sun to include as its 'solar system'.

    Returns:
    - new_pos (dict): Updated node positions with local adjustments.
    """
    new_pos = pos.copy()

    for sun in suns:
        if sun not in pos:
            continue
        sun_pos = np.array(pos[sun])

        # Get ego nodes within 'hops' steps, excluding the sun itself
        ego_nodes = set(nx.ego_graph(graph, sun, radius=hops).nodes()) - {sun}

        for node in ego_nodes:
            if node not in pos:
                continue
            node_pos = np.array(pos[node])
            direction = sun_pos - node_pos
            adjusted_pos = node_pos + strength * direction
            new_pos[node] = adjusted_pos.tolist()

    return new_pos

# =========================
# SEED SELECTION HELPERS
# =========================


def single_discount(G, k, weight=None):
    """
    Single Discount heuristic from Chen et al. (KDD'09).

    Parameters:
    - G : nx.Graph
    - k : int (number of seeds to select)
    - weight : str or None, edge weight attribute (ignored for SD)

    Returns:
    - S : list (selected seed nodes)
    """
    S = []
    degree = {v: G.degree(v) for v in G}
    discounted = degree.copy()

    for _ in range(k):
        u = max(discounted, key=discounted.get)
        S.append(u)
        discounted[u] = -1 

        # Discount neighbors (1 per seed neighbor)
        for v in G.neighbors(u):
            if v not in S:
                discounted[v] -= 1

    return S


def degree_discount_ic(G, k, p=0.01):
    """
    DegreeDiscountIC heuristic for Independent Cascade model.
    Chen, Wang & Yang (KDD'09).

    Parameters: 
    - G : nx.Graph
    - k : int (number of seeds)
    - p : float (propagation probability (IC model))

    Returns:
    - S : list (selected seed nodes)
    """
    S = []
    deg = {v: G.degree(v) for v in G}
    t = {v: 0 for v in G}          # seed-neighbor count
    dd = deg.copy()               # discounted degree

    for _ in range(k):
        u = max(dd, key=dd.get)
        S.append(u)
        dd[u] = -1

        for v in G.neighbors(u):
            if v not in S:
                t[v] += 1
                dv = deg[v]
                tv = t[v]
                # DDIC discount formula
                dd[v] = dv - 2*tv - (dv - tv)*tv*p

    return S

def get_k_from_percent(G, k=None, percent=None):
    """
    Convert percent → count. If k is provided, return k.
    If percent is provided, compute number of nodes.
    """
    n = G.number_of_nodes()
    if k is not None:
        return k
    if percent is not None:
        return max(1, int(n * percent))
    raise ValueError("Either k or percent must be provided.")

def random_seeds(G, k, seed=42):
    """
    Select k random nodes from the graph.
    """
    rng = np.random.default_rng(seed)
    return list(rng.choice(list(G.nodes()), size=k, replace=False))

def select_suns(G, k, percent, method, ci_scores=None):
    """
    method: "random", "single_discount", "degree_discount", "top_CI"
    """         
    k_suns = get_k_from_percent(
        G,
        k=k,
        percent=percent
    )
    if method == "random":
        return random_seeds(G, k_suns)
    elif method == "single_discount":
        return single_discount(G, k_suns)
    elif method == "degree_discount":
        return degree_discount_ic(G, k=k_suns, p=0.01)
    elif method == "top_CI":
        return sorted(ci_scores, key=ci_scores.get, reverse=True)[:k]
    else:
        raise ValueError("Error: Unknown sun selection method.")
    

def save_layout_plot_with_edges(layout_positions, graph, dataset_name, method, output_folder, highlight_nodes=None, highlight_edges=None):
    """
    Save a scatter plot of the reduced data with straight-line edges.
    """
    plt.figure(figsize=(10, 10))

    # Scatter plot for nodes
    x, y = zip(*layout_positions.values())
    plt.scatter(x, y, s=10, c="blue", label="Nodes", zorder=2)

    #prepare highlight edges 
    highlight_edge_set = set()

    if highlight_edges:
        # Normalize tuples to ensure (u,v) and (v,u) both work
        for (u, v) in highlight_edges:
            if (u, v) in graph.edges():
                highlight_edge_set.add((u, v))
            elif (v, u) in graph.edges():  # undirected case
                highlight_edge_set.add((v, u))

    default_edges = [e for e in graph.edges() if e not in highlight_edge_set]

    # Draw straight-line edges
    for edge in graph.edges():
        source, target = edge
        if source in layout_positions and target in layout_positions:
            x_values = [layout_positions[source][0], layout_positions[target][0]]
            y_values = [layout_positions[source][1], layout_positions[target][1]]
            plt.plot(x_values, y_values, color="gray", alpha=0.5, zorder=1)

    if highlight_nodes:
        nx.draw_networkx_nodes(
            graph, layout_positions,
            nodelist=highlight_nodes,
            node_size=80,
            node_color="red",
            edgecolors="yellow",
            linewidths=1.5,
            alpha=0.9
        )

    for (u, v) in highlight_edge_set:
        if u in layout_positions and v in layout_positions:
            x_vals = [layout_positions[u][0], layout_positions[v][0]]
            y_vals = [layout_positions[u][1], layout_positions[v][1]]
            plt.plot(
                x_vals,
                y_vals,
                color="red",
                alpha=0.9,
                linewidth=2.5,
                zorder=3,
            )

    plt.title(f"{method} layout for {dataset_name} (Edges)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc="best")

    os.makedirs(output_folder, exist_ok=True)
    filename = f"{output_folder}/{dataset_name}_{method}_layout_with_edges.png"
    plt.savefig(filename)
    plt.close()

def save_bundled_edge_plot(control_point_lists, reduced_positions, dataset_name, method, output_folder, colormap="plasma", sun_nodes=None):
    """
    Save a scatter plot of the reduced data with edge-path bundling applied.
    """

    def path_deviation(path):
        if len(path) < 3:
            return 0.0
        start = np.array(path[0])
        end = np.array(path[-1])
        total_dev = 0
        for pt in path[1:-1]:
            pt = np.array(pt)
            dev = np.linalg.norm(np.cross(end - start, start - pt)) / np.linalg.norm(end - start)
            total_dev += dev
        return total_dev

    # Compute deviations
    deviations = [path_deviation(p) for p in control_point_lists]
    max_dev = max(deviations) if deviations else 1.0
    cmap = plt.get_cmap(colormap)

    # Plot
    plt.figure(figsize=(10, 10))

    # Plot nodes
    x, y = zip(*reduced_positions.values())
    plt.scatter(x, y, s=10, c="royalblue", label="Nodes", alpha=0.8, zorder=2)

    if sun_nodes:
        sun_x = []
        sun_y = []
        for s in sun_nodes:
            if s in reduced_positions:
                sun_x.append(reduced_positions[s][0])
                sun_y.append(reduced_positions[s][1])

        plt.scatter(sun_x, sun_y, color="gold", edgecolors="black", s=25, zorder=3, label="Top CI Nodes")

    # Plot bundled curves
    for control_points, dev in zip(control_point_lists, deviations):
        norm_dev = dev / max_dev
        color = cmap(norm_dev)
        x_values, y_values = zip(*control_points)                                         
        plt.plot(x_values, y_values, color="gray", linewidth=0.6, alpha=0.25, zorder=1)     

    plt.title(f"{method} layout for {dataset_name} (EPB)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc="best")

    os.makedirs(output_folder, exist_ok=True)
    filename = f"{output_folder}/{dataset_name}_{method}_layout_with_epb_edges.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved edge-bundled layout with colored curves to: {filename}")

def save_layout_plot_with_virtual_overlay_2(
    pos,
    G,
    dataset_name,
    label,
    output_folder,
    tag="virtual",
    highlight_nodes=None,          
    added_virtual_edges=None,       
    role_sets=None,                 
    edge_lw=1.0,
    edge_alpha=0.6
):
    """
    Overlay plot that shows:
      - All non-virtual edges in gray
      - All virtual edges in solid orange
      - Nodes colored by role if role_sets provided: suns, planets, moons, others

    Parameters
    ----------
    pos : dict[node] -> (x, y)
    G : nx.Graph or nx.DiGraph
    dataset_name : str
    label : str
    output_folder : str
    tag : str
        Edge attribute key used to mark virtual edges.
    role_sets : dict or None
        {'suns': set, 'planets': set, 'moons': set}. If None, draws all nodes in blue.
    edge_lw : float
        Line width for both real and virtual edges.
    edge_alpha : float
        Alpha for both real and virtual edges.
    """
    os.makedirs(output_folder, exist_ok=True)

    plt.figure(figsize=(10, 10))

    # --- draw non-virtual edges (gray) ---
    for u, v, d in G.edges(data=True):
        if d.get(tag, False):
            continue
        if u in pos and v in pos:
            plt.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                color="gray",
                alpha=0.5,              
                linewidth=edge_lw,
                zorder=1
            )

    # --- draw virtual edges (solid orange, same lw/alpha as normal) ---
    for u, v, d in G.edges(data=True):
        if not d.get(tag, False):
            continue
        if u in pos and v in pos:
            plt.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                linestyle="-",
                color="orange",
                alpha=edge_alpha,
                linewidth=edge_lw,
                zorder=2
            )

    # --- draw nodes ---
    if role_sets is None:
        xs, ys = zip(*[(x, y) for n, (x, y) in pos.items() if n in G])
        plt.scatter(xs, ys, s=10, c="blue", alpha=0.95, zorder=3)
    else:
        suns    = [n for n in role_sets.get("suns", set())    if n in pos]
        planets = [n for n in role_sets.get("planets", set()) if n in pos]
        moons   = [n for n in role_sets.get("moons", set())   if n in pos]
        others  = [n for n in pos if n not in role_sets.get("suns", set())
                                 and n not in role_sets.get("planets", set())
                                 and n not in role_sets.get("moons", set())]

        def _scatter(nodes, s, c, z):
            if nodes:
                plt.scatter(
                    [pos[n][0] for n in nodes],
                    [pos[n][1] for n in nodes],
                    s=s, c=c, alpha=0.95, zorder=z
                )

        # moons under, then planets, then others, suns on top
        _scatter(moons,   16, "mediumseagreen", 3)
        _scatter(planets, 18, "steelblue",      4)
        _scatter(others,  12, "lightgray",      4)

        if suns:
            plt.scatter(
                [pos[s][0] for s in suns],
                [pos[s][1] for s in suns],
                s=80, c="crimson", edgecolors="yellow", linewidths=1.6, zorder=5
            )

    if highlight_nodes:
        keep = [n for n in highlight_nodes if n in pos]
        if keep:
            plt.scatter(
                [pos[n][0] for n in keep],
                [pos[n][1] for n in keep],
                s=90, facecolors="none", edgecolors="yellow", linewidths=1.8, zorder=6
            )

    plt.title(f"{label} layout for {dataset_name} (Edges)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    out = f"{output_folder}/{dataset_name}_{label}_layout_with_edges.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[viz] Saved {out}")



def compute_edge_betweenness(G):
        start_time = time()
        edge_betweenness = nx.edge_betweenness_centrality(G, weight="distance_weight")
        return edge_betweenness, time() - start_time

def compute_edge_flow_centrality(G):
    start_time = time()
    L = nx.laplacian_matrix(G).todense()
    eigvals, eigvecs = np.linalg.eigh(L)

    # Map each node to its corresponding index in the eigenvector matrix
    node_to_index = {node: idx for idx, node in enumerate(G.nodes())}
    
    edge_flow = {}
    for u, v in G.edges():
        idx_u = node_to_index[u]
        idx_v = node_to_index[v]
        edge_flow[(u, v)] = sum((eigvecs[idx_u, i] - eigvecs[idx_v, i])**2 for i in range(len(eigvals)))
    
    duration = time() - start_time
    return edge_flow, duration

def compute_collective_influence(G, l=2):
    start_time = time()
    ci_scores = {node: 0 for node in G.nodes()}
    
    for node in G.nodes():
        neighbors_l = nx.single_source_dijkstra_path_length(G, node, cutoff=l, weight="distance_weight")
        ci_scores[node] = (G.degree(node, weight="weight") - 1) * sum((G.degree(neighbor, weight="weight")
                                                                                - 1) for neighbor in neighbors_l if neighbor != node)

    edge_ci = {(u, v): ci_scores[u] + ci_scores[v] for u, v in G.edges()}
    duration = time() - start_time
    print(f"CI Score Range: {min(edge_ci.values()):.2f} to {max(edge_ci.values()):.2f}")

    return ci_scores, edge_ci, duration

def compute_kcore_edges(G):
    start_time = time()
    G.remove_edges_from(nx.selfloop_edges(G))
    kcore_values = nx.core_number(G)
    edge_kcore = {(u, v): min(kcore_values[u], kcore_values[v]) for u, v in G.edges()}
    duration = time() - start_time
    return edge_kcore, duration

def compute_community_bridging_edges(G):
    start_time = time()
    communities = list(greedy_modularity_communities(G))
    node_community = {}
    
    for i, comm in enumerate(communities):
        for node in comm:
            node_community[node] = i  # Assign community ID

    edge_community = {(u, v): 1 if node_community[u] != node_community[v] else 0 for u, v in G.edges()}
    duration = time() - start_time
    return edge_community, duration



def calculate_fairness_metric(positions, graph_distances, group_A, group_B):
    nodes = list(positions.keys())
    idx = {n: i for i, n in enumerate(nodes)}

    # Filter to layout nodes (safety)
    group_A = [n for n in group_A if n in idx]
    group_B = [n for n in group_B if n in idx]
    if len(group_A) == 0 or len(group_B) == 0:
        return np.nan

    max_graph = np.max(graph_distances)
    if max_graph == 0:
        max_graph = 1.0

    def stress_of_group(group):
        s = 0
        # eps = 1e-6
        for u in group:
            iu = idx[u]
            pos_u = positions[u]
            diffs = np.linalg.norm(
                np.array([positions[v] for v in nodes]) - pos_u,
                axis=1
            )

            diffs_norm = diffs / np.max(diffs) if np.max(diffs) > 0 else diffs
            graph_d = graph_distances[iu] / max_graph

            s += np.sum((diffs_norm - graph_d) ** 2)
            # s += np.mean(((diffs_norm - graph_d) / (graph_d + eps)) ** 2)
        return s / len(group)

    stress_A = stress_of_group(group_A)
    stress_B = stress_of_group(group_B)

    return (stress_A - stress_B) ** 2


def get_all_solar_nodes(graph, suns, hops=2):
    """
    Returns the union of ego-nets around all suns.
    No ownership: a node belongs to the solar group
    if it is within `hops` of ANY sun.
    """
    solar_nodes = set()
    for s in suns:
        ego = nx.ego_graph(graph, s, radius=hops)
        solar_nodes.update(ego.nodes())
    return solar_nodes


def forceatlas2_layout(
    G,
    pos=None,
    *,
    degree_weight_attribute=None,  # New parameter for weighted degree
    max_iter=100,
    jitter_tolerance=1.0,
    scaling_ratio=2.0,
    gravity=1.0,
    distributed_action=False,
    strong_gravity=False,
    node_mass=None,
    node_size=None,
    weight=None,
    dissuade_hubs=False,
    linlog=True,
    seed=None,
    dim=2,
):
    """Position nodes using the ForceAtlas2 force-directed layout algorithm."""

    if len(G) == 0:
        return {}
    # parse optional pos positions
    if pos is None:
        pos = nx.random_layout(G, dim=dim, seed=seed)
        pos_arr = np.array(list(pos.values()))
    else:
        rng = np.random.default_rng(random_state)
        # set default node interval within the initial pos values
        pos_init = np.array(list(pos.values()))
        max_pos = pos_init.max(axis=0)
        min_pos = pos_init.min(axis=0)
        dim = max_pos.size
        pos_arr = min_pos + rng.random((len(G), dim)) * (max_pos - min_pos)
        for idx, node in enumerate(G):
            if node in pos:
                pos_arr[idx] = pos[node].copy()

    mass = np.zeros(len(G))
    size = np.zeros(len(G))

    # Only adjust for size when the users specifies size other than default
    adjust_sizes = False
    if node_size is None:
        node_size = {}
    else:
        adjust_sizes = True

    if node_mass is None:
        node_mass = {}

    for idx, node in enumerate(G):
        if node in node_mass:
            mass[idx] = node_mass.get(node, G.degree(node) + 1)
        else:
            # Use weighted degree if attribute is provided, otherwise default to unweighted
            if degree_weight_attribute is not None:
                mass[idx] = G.degree(node, weight=degree_weight_attribute)
            else:
                mass[idx] = G.degree(node) + 1
        size[idx] = node_size.get(node, 1)

    n = len(G)
    gravities = np.zeros((n, dim))
    attraction = np.zeros((n, dim))
    repulsion = np.zeros((n, dim))
    A = nx.to_numpy_array(G, weight=weight)

    def estimate_factor(n, swing, traction, speed, speed_efficiency, jitter_tolerance):
        import numpy as np

        # estimate jitter
        opt_jitter = 0.05 * np.sqrt(n)
        min_jitter = np.sqrt(opt_jitter)
        max_jitter = 10
        min_speed_efficiency = 0.05

        other = min(max_jitter, opt_jitter * traction / n**2)
        jitter = jitter_tolerance * max(min_jitter, other)

        if swing / traction > 2.0:
            if speed_efficiency > min_speed_efficiency:
                speed_efficiency *= 0.5
            jitter = max(jitter, jitter_tolerance)
        if swing == 0:
            target_speed = np.inf
        else:
            target_speed = jitter * speed_efficiency * traction / swing

        if swing > jitter * traction:
            if speed_efficiency > min_speed_efficiency:
                speed_efficiency *= 0.7
        elif speed < 1000:
            speed_efficiency *= 1.3

        max_rise = 0.5
        speed = speed + min(target_speed - speed, max_rise * speed)
        return speed, speed_efficiency

    speed = 1
    speed_efficiency = 1
    swing = 1
    traction = 1
    for _ in range(max_iter):
        # compute pairwise difference
        diff = pos_arr[:, None] - pos_arr[None]
        # compute pairwise distance
        distance = np.linalg.norm(diff, axis=-1)

        # linear attraction
        if linlog:
            attraction = -np.log(1 + distance) / (distance + 1e-5)
            np.fill_diagonal(attraction, 0)
            attraction = np.einsum("ij, ij -> ij", attraction, A)
            attraction = np.einsum("ijk, ij -> ik", diff, attraction)

        else:
            attraction = -np.einsum("ijk, ij -> ik", diff, A)

        if distributed_action:
            attraction /= mass[:, None]

        # repulsion
        tmp = mass[:, None] @ mass[None]
        if adjust_sizes:
            distance += -size[:, None] - size[None]

        d2 = np.maximum(distance**2, 1e-10)  # Prevent division by zero
        # remove self-interaction
        np.fill_diagonal(tmp, 0)
        np.fill_diagonal(d2, 1)
        factor = (tmp / d2) * scaling_ratio
        repulsion = np.einsum("ijk, ij -> ik", diff, factor)

        # gravity
        gravities = (
            -gravity
            * mass[:, None]
            * pos_arr
            / np.linalg.norm(pos_arr, axis=-1)[:, None]
        )

        if strong_gravity:
            gravities *= np.linalg.norm(pos_arr, axis=-1)[:, None]
        # total forces
        update = attraction + repulsion + gravities

        # compute total swing and traction
        swing += (mass * np.linalg.norm(pos_arr - update, axis=-1)).sum()
        traction += (0.5 * mass * np.linalg.norm(pos_arr + update, axis=-1)).sum()

        speed, speed_efficiency = estimate_factor(
            n,
            swing,
            traction,
            speed,
            speed_efficiency,
            jitter_tolerance,
        )

        # update pos
        if adjust_sizes:
            swinging = mass * np.linalg.norm(update, axis=-1)
            factor = 0.1 * speed / (1 + np.sqrt(speed * swinging))
            df = np.linalg.norm(update, axis=-1)
            factor = np.minimum(factor * df, 10.0 * np.ones(df.shape)) / df
        else:
            swinging = mass * np.linalg.norm(update, axis=-1)
            factor = speed / (1 + np.sqrt(speed * swinging))

        pos_arr += update * factor[:, None]
        if abs((update * factor[:, None]).sum()) < 1e-10:
            break

    return dict(zip(G, pos_arr))



def save_metadata(dataset_name, output_folder, stress_tsne, runtime_stats, params):
    """
    Save metadata about the dataset, results, and pipeline parameters to a text file.
    """
    metadata_filepath = f"{output_folder}/{dataset_name}_metadata.txt"
    with open(metadata_filepath, "w") as f:
        # General Dataset Information
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Number of Nodes: {runtime_stats['nodes']}\n")
        f.write(f"Number of Edges: {runtime_stats['edges']}\n")
        f.write(f"Graph Type: {'Directed' if runtime_stats['directed'] else 'Undirected'}\n\n")

        # times
        f.write(f"Times:\n")
        # f.write(f"  Runtime: {runtime_stats['tsne_time']:.2f} seconds\n")
        f.write(f"  Total pipeline time: {runtime_stats['total_pipeline_time']:.2f} seconds\n")
        f.write(f"  Graph processing time: {runtime_stats['graph_processing_time']:.2f} seconds\n")
        f.write(f"  Dimensionality reduction time: {runtime_stats['dr_processing_time']:.2f} seconds\n")
        f.write(f"  Betweenness time: {runtime_stats['betweenness_time']:.2f} seconds\n")
        f.write(f"  Flow time: {runtime_stats['flow_time']:.2f} seconds\n")
        f.write(f"  CI time: {runtime_stats['ci_time']:.2f} seconds\n")
        f.write(f"  Kcore time: {runtime_stats['kcore_time']:.2f} seconds\n")
        f.write(f"  Community time: {runtime_stats['community_time']:.2f} seconds\n")
        f.write(f"  PageRank time: {runtime_stats['pagerank_time']:.2f} seconds\n")
        
        #metrics
        f.write("Metrics:\n")
        if stress_tsne is not None:
            f.write(f"  Stress: {stress_tsne:.4f}\n")
        f.write(f"  Neighbourhood Preservation score (tsne): {runtime_stats['neighborhood_preservation_tsne']:.4f} \n")
        f.write(f"  Neighbourhood Preservation score (force-directed): {runtime_stats['neighborhood_preservation_fd']:.4f} \n")
        f.write(f"  Silhouette score (tsne): {runtime_stats['silhouette_score_tsne']:.4f} \n")
        f.write(f"  Best 'k' (tsne): {runtime_stats['best_k_tsne']:.4f} \n")
        # f.write(f"  Silhouette score DB (tsne): {runtime_stats['silhouette_score_DB_tsne']:.4f} \n")
        f.write(f"  Silhouette score (force-directed): {runtime_stats['silhouette_score_fd']:.4f} \n")
        f.write(f"  Best 'k' (force-directed): {runtime_stats['best_k_fd']:.4f} \n")
        # f.write(f"  Silhouette score DB (force-directed): {runtime_stats['silhouette_score_DB_fd']:.4f} \n")

        f.write(f"  Node overlap (tsne): {runtime_stats['node_overlap']} \n")
        f.write(f"  Area utilization (tsne): {runtime_stats['area_utilization']} \n")
        f.write(f"  Edge crossings (tsne): {runtime_stats['edge_crossings']} \n")

        f.write(f"Fairness Suns vs Rest (tsne): {runtime_stats['fairness_suns_vs_rest_tsne']:.6f}\n")
        f.write(f"Fairness Top 10% CI vs Rest (tsne): {runtime_stats['fairness_top10_vs_rest_tsne']:.6f}\n")
        f.write(f"Fairness Solar System vs Rest (tsne): {runtime_stats['fairness_solar_vs_rest_tsne']:.6f}\n")
        f.write(f"Fairness Suns vs Rest (force-directed): {runtime_stats['fairness_suns_vs_rest']:.6f}\n")
        f.write(f"Fairness Top 10% CI vs Rest (force-directed): {runtime_stats['fairness_top10_vs_rest']:.6f}\n")
        f.write(f"Fairness Solar System vs Rest (force-directed): {runtime_stats['fairness_solar_vs_rest']:.6f}\n")

        f.write(f"  Gini Stress (tsne): {runtime_stats['gini_stress_tsne']:.4f}\n")
        f.write(f"  Gini Stress (FA2): {runtime_stats['gini_stress_fa2']:.4f}\n")


        # Hyperparameters / Config
        def _dump_section(title, cfg):
            if not cfg:
                return
            f.write(f"{title}:\n")
            for k, v in cfg.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

        if params:
            _dump_section("t-SNE Parameters", params.get("tsne"))
            _dump_section("ForceAtlas2 (Baseline) Parameters", params.get("fa2_baseline"))
            _dump_section("Trimming the Hairball (Edge-Cutting) Parameters", params.get("thb_trimming"))
            _dump_section("DBSCAN Parameters", params.get("dbscan"))
            _dump_section("Edge-Path Bundling (EPB) Parameters", params.get("epb"))
            _dump_section("Solar Merger / Path Collapse Parameters", params.get("solar"))

        # Visualizations
        f.write("Generated Visualizations:\n")
        for vis in runtime_stats['visualizations']:
            f.write(f"  {vis}\n")


############################################################################################
# EXECUTION
############################################################################################

def run_full_pipeline_single_config(dataset_name, dataset_info, sun_method, sun_percent, output_folder, random_state):

    print(f"Running config: {dataset_name} | {sun_method} | {sun_percent}")

    print(f"Processing dataset: {dataset_name}")
    pipeline_start = time()
    filepath = dataset_info["filepath"]
    directed = dataset_info.get("directed", False)

    # --- Load dataset and prepare CSV graph 'g' ---
    df = load_clean_edge_list(filepath, source_col="Source", target_col="Target", weight_col="Weight")
    if dataset_name in FILTERED_DATASETS:
        print(f"{dataset_name} checked")
        if dataset_name == "PGP":
            df = filter_edge_dataframe(df, frac=0.3, min_weight=None, random_state=random_state)
        else:
            df = filter_edge_dataframe(df, frac=0.5, min_weight=None, random_state=random_state)
    g = nx.DiGraph() if directed else nx.Graph()

    # --- Now add edges from CSV (which creates nodes) ---
    for _, row in df.iterrows():
        source, target, weight = row["Source"], row["Target"], row["Weight"]
        g.add_edge(source, target, weight=weight ** d)
    
    g = filter_low_degree_nodes(g, min_degree=0)

    runtime_stats = {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(), "directed": directed, "visualizations": []}


    # Apply Pathfinder Network Scaling
    print("Applying Pathfinder Network Scaling...")
    pns_graph = g
    runtime_stats["pns_edges"] = pns_graph.number_of_edges()
    print(f"PNS applied. Number of edges reduced from {g.number_of_edges()} to {pns_graph.number_of_edges()}")

    # --- Trimming the Hairball: skeleton + fill (for FA2 comparison) ---
    if TRIMMING_CFG.get("enabled", True):
        print("[THB] Building skeleton and filled graphs...")
        skel_graph, filled_graph, comms = trim_and_fill(
            pns_graph,
            strategy=TRIMMING_CFG["strategy"],
            target_edges_per_node=TRIMMING_CFG["target_edges_per_node"],
            reconnect=TRIMMING_CFG["reconnect"],
            betweenness_k=TRIMMING_CFG["betweenness_k"],
            seed=TRIMMING_CFG["seed"],
            weight="weight"
        )
        pns_graph = filled_graph
        print(f"[THB] Skeleton edges: {skel_graph.number_of_edges()} | Filled edges: {filled_graph.number_of_edges()}")
    
    EPS = 1e-12

    edges_to_remove = [
        (u, v) for u, v, d in pns_graph.edges(data=True)
        if ("weight" not in d) or (not np.isfinite(d["weight"])) or (float(d["weight"]) <= EPS)
    ]

    if edges_to_remove:
        print(f"[INFO] Removing {len(edges_to_remove)} edges with ~zero weights.")
        pns_graph.remove_edges_from(edges_to_remove)

    # Safe inversion
    for _, _, data in pns_graph.edges(data=True):
        w = float(data["weight"])
        assert w > EPS, f"Non-positive weight slipped through: {w}"
        data["distance_weight"] = 1.0 / w

    graph_processing_time = time()  # Time before distance calculation

    nodes = list(pns_graph.nodes())
    graph_distances = dijkstra_all_pairs_matrix_cutoff(pns_graph, weight="distance_weight", cutoff=None)
    max_distance = np.nanmax(graph_distances[np.isfinite(graph_distances)])
    graph_distances[~np.isfinite(graph_distances)] = max_distance
    
    if directed:
        graph_distances = (graph_distances + graph_distances.T) / 2

    betweenness_scores, time_betweenness = compute_edge_betweenness(pns_graph)
    flow_scores, time_flow = compute_edge_flow_centrality(pns_graph)
    node_ci, edge_ci, time_ci = compute_collective_influence(pns_graph)
    kcore_scores, time_kcore = compute_kcore_edges(pns_graph)
    community_scores, time_community = compute_community_bridging_edges(pns_graph)
    
    selected_suns = select_suns(pns_graph, k=None, percent=1.0, method=sun_method, ci_scores=node_ci)

    suns_percent = get_k_from_percent(pns_graph, percent=sun_percent)

    # --- Ensure spatially separated Suns (based on shortest-path distance) ---
    spaced_suns = select_spaced_suns(
        G=pns_graph,
        ranked_nodes=selected_suns,
        k=suns_percent,    
        min_distance=EXPERIMENT_CFG["min_spacing"]      # minimum shortest-path distance between Suns
    )

    top_nodes = spaced_suns  # reuse variable for downstream consistency

    # ===========================================================
    # Influence-highways between selected Suns (top_nodes)
    # ===========================================================
    if INFLUENCE_PATH_CFG["enabled"]:
        best_paths = find_top_influence_paths_between_suns(
            G=pns_graph,
            suns=top_nodes,
            influence_attr=INFLUENCE_PATH_CFG["weight_attr"],
            k_per_pair=INFLUENCE_PATH_CFG["k_per_pair"],
            top_k_global=INFLUENCE_PATH_CFG["top_k_global"]
        )

        influence_edges = set()
        for entry in best_paths:
            path = entry["path"]
            influence_edges.update(zip(path[:-1], path[1:]))

        print(f"[INFO] Highlighting {len(influence_edges)} high-influence edges between Suns.")
    else:
        best_paths = []
        influence_edges = set()



    start_time = time()
    pagerank_scores = nx.pagerank(pns_graph, weight='weight')
    pagerank_time = time() - start_time
    runtime_stats["pagerank_time"] = pagerank_time

    df_comparison = compare_pagerank_and_degree(pns_graph, pagerank_scores)
    df_comparison.to_csv(f"{dataset_name}_pagerank_vs_degree.csv", index=False)

    runtime_stats.update({
        "betweenness_time": time_betweenness,
        "flow_time": time_flow,
        "ci_time": time_ci,
        "kcore_time": time_kcore,
        "community_time": time_community
    })

    pre_dr_time = time()  # Time just before dimensionality reduction
    # Dimensionality Reduction: t-SNE
    start = time()
    tsne = TSNE(n_components=2, metric="precomputed", random_state=random_state, init="random")
    reduced_tsne = tsne.fit_transform(graph_distances)
    reduced_positions_tsne = {n: reduced_tsne[i] for i, n in enumerate(nodes)}

    layout_distances = pairwise_distances(reduced_tsne)

    runtime_stats["tsne_time"] = time() - start
    runtime_stats["graph_processing_time"] = pre_dr_time - graph_processing_time  # Time for graph processing
    runtime_stats["dr_processing_time"] = time() - pre_dr_time  # Time for dimensionality reduction
    print(f"TSNE completed in {time() - start:.2f}s")

    # Calculate Neighborhood Preservation
    neighborhood_preservation_score_tsne = calculate_neighborhood_preservation(graph_distances, layout_distances, k=5)
    print(f"Neighborhood Preservation Score (tsne): {neighborhood_preservation_score_tsne:.4f}")
    runtime_stats["neighborhood_preservation_tsne"] = neighborhood_preservation_score_tsne

    analyze_tsne_output(reduced_positions_tsne, dataset_name)

    #cluster metric
    best_k_tsne, best_sil_score_tsne = find_optimal_clusters(reduced_tsne, max_k=10)
    print(f"Optimal number of clusters: {best_k_tsne}, Silhouette Score: {best_sil_score_tsne:.4f}")
    runtime_stats["silhouette_score_tsne"] = best_sil_score_tsne
    runtime_stats["best_k_tsne"] = best_k_tsne

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    cluster_labels = dbscan.fit_predict(reduced_tsne)
    if len(set(cluster_labels)) > 1:  # Ensure at least 2 clusters
        sil_score_DB_tsne = silhouette_score(reduced_tsne, cluster_labels)
        runtime_stats["silhouette_score_DB_tsne"] = sil_score_DB_tsne
        print(f"DBSCAN Silhouette Score: {sil_score_DB_tsne:.4f}")
    else:
        print("DBSCAN found fewer than 2 clusters, silhouette score not computed.")

    #visual aesthetics metrics
    overlap = calculate_node_overlap(reduced_positions_tsne)
    utilization = calculate_area_utilization(reduced_positions_tsne)
    crossings = count_edge_crossings(reduced_positions_tsne, pns_graph)

    print(f"Node Overlap Count: {overlap}")
    print(f"Area Utilization: {utilization:.4f}")
    print(f"Edge Crossings: {crossings}")

    runtime_stats["node_overlap"] = overlap
    runtime_stats["area_utilization"] = utilization
    runtime_stats["edge_crossings"] = crossings

    # Fairness A: Suns vs Non-Suns
    group_A_suns_tsne = list(top_nodes)
    group_B_others_tsne = [n for n in g.nodes() if n not in top_nodes]

    fairness_suns_vs_rest_tsne = calculate_fairness_metric(
        positions=reduced_positions_tsne,
        graph_distances=graph_distances,
        group_A=group_A_suns_tsne,
        group_B=group_B_others_tsne
    )

    # Fairness B: Top 10% CI vs rest
    ci_sorted_tsne = sorted(node_ci, key=node_ci.get, reverse=True)
    cut_tsne = max(1, int(0.10 * len(ci_sorted_tsne)))   # ensures non-empty
    group_A_top10_tsne = ci_sorted_tsne[:cut_tsne]
    group_B_rest_tsne = ci_sorted_tsne[cut_tsne:]

    fairness_top10_vs_rest_tsne = calculate_fairness_metric(
        positions=reduced_positions_tsne,
        graph_distances=graph_distances,
        group_A=group_A_top10_tsne,
        group_B=group_B_rest_tsne
    )
    runtime_stats["fairness_suns_vs_rest_tsne"] = fairness_suns_vs_rest_tsne
    runtime_stats["fairness_top10_vs_rest_tsne"] = fairness_top10_vs_rest_tsne

    # Fairness C: Solar system vs rest
    solar_nodes = get_all_solar_nodes(
        graph=g,
        suns=top_nodes,
        hops=2
        )
    group_A_solar = list(solar_nodes)
    group_B_rest_solar = [n for n in g.nodes() if n not in solar_nodes]

    fairness_solar_vs_rest_tsne = calculate_fairness_metric(
        positions=reduced_positions_tsne,
        graph_distances=graph_distances,
        group_A=group_A_solar,
        group_B=group_B_rest_solar
    )

    runtime_stats["fairness_solar_vs_rest_tsne"] = fairness_solar_vs_rest_tsne

    # --- t-SNE Gini Stress ---
    node_stress_tsne = compute_node_stress(
        positions=reduced_positions_tsne,
        graph_distances=graph_distances
    )

    gini_stress_tsne = gini_coefficient(list(node_stress_tsne.values()))
    runtime_stats["gini_stress_tsne"] = gini_stress_tsne

    print(f"[GINI STRESS] t-SNE: {gini_stress_tsne:.4f}")

    adjusted_positions = reduced_positions_tsne   
    node_owner = assign_unique_sun_owners(pns_graph, suns=top_nodes, cutoff=None)
    role_sets = compute_role_sets(
        G=pns_graph, pos=adjusted_positions, suns=top_nodes, owners=node_owner,
        include_moons=VIRTUAL_EDGES_CFG.get("include_moons", True)
    )

    if SOLAR_CFG.get("enabled", True):
            # 1) Solar merger
            adjusted_positions = pull_neighbors_toward_source(
                pos=reduced_positions_tsne,
                graph=pns_graph,
                suns=top_nodes,
                strength=SOLAR_CFG["solar_strength"],
                hops=SOLAR_CFG["solar_hops"]
            )
    
    if VIRTUAL_EDGES_CFG.get("enabled", True):
        # 2) Add virtual edges
        virtual_edges, vstats = add_virtual_edges(pns_graph, adjusted_positions, suns=top_nodes, cfg=VIRTUAL_EDGES_CFG, owners = node_owner)
        print(f"[VirtualEdges] Added {vstats['added']} edges across {vstats['groups_touched']} groups "
                f"(strategy={vstats['strategy']}, K={vstats['K']}, w_fake={vstats['w_fake']}, "
                f"planets={vstats['include_planets']}, moons={vstats['include_moons']})")
        
        save_layout_plot_with_virtual_overlay_2(
            pos=adjusted_positions,
            G=pns_graph, 
            dataset_name=dataset_name,
            label="Solar+Virtual_preFA2", 
            output_folder=output_folder,
            tag=VIRTUAL_EDGES_CFG.get("tag", "virtual"),
            highlight_nodes=top_nodes, 
            added_virtual_edges=virtual_edges,
            role_sets=role_sets, 
            edge_lw=1.0, edge_alpha=0.6
        )
        runtime_stats["visualizations"].append(
            f"{output_folder}/{dataset_name}_Solar+Virtual_preFA2.png"
        )

        # --- build the keep sets immediately after adding virtuals, BEFORE FA2 ---
        keep_info = gather_virtual_and_paths_for_viz(
            G=pns_graph,
            pos=adjusted_positions,           # pre-FA2 (t-SNE or Solar-adjusted)
            suns=top_nodes,
            owners=node_owner,                
            cfg=VIRTUAL_EDGES_CFG,
            role_sets=role_sets,
            virtual_tag=VIRTUAL_EDGES_CFG.get("tag", "virtual"),
            include_moons_for_viz=True ,
        )

        # (1) Pre-FA2 view with only virtuals + planet paths
        plot_only_virtuals_and_planet_paths(
            pos=adjusted_positions,  # then later: force_directed_positions
            G=pns_graph,
            dataset_name=dataset_name,
            label="Solar+Virtual_preFA2_ONLY",
            output_folder=output_folder,
            keep_info=keep_info,
            role_sets=role_sets,         
        )

    # Force-Directed Layout
    force_directed_positions = forceatlas2_layout(G=pns_graph, pos=adjusted_positions, weight="weight", 
        max_iter=100, linlog=True, scaling_ratio=2.0, seed=random_state) # , node_mass=node_mass, degree_weight_attribute='weight'
    
    def dump_graph_edges_csv(G, out_path):
            rows = []
            for u, v, d in G.edges(data=True):
                rows.append({
                    "Source": u,
                    "Target": v,
                    "Weight": d.get("weight", 1.0)
                })
            pd.DataFrame(rows).to_csv(out_path, sep=";", index=False)
    
    # (2) Post-FA2 view with only virtuals + planet paths
    plot_only_virtuals_and_planet_paths(
        pos=force_directed_positions,
        G=pns_graph,
        dataset_name=dataset_name,
        label="FA2_ONLY_virtuals_and_planet_paths",
        output_folder=output_folder,
        keep_info=keep_info,
        role_sets=role_sets,
    )
    
    removed = remove_virtual_edges(pns_graph, tag=VIRTUAL_EDGES_CFG["tag"])
    print(f"[VirtualEdges] removed={removed}")

    # ======================================================
    # OPTIONAL: Edge Bundling Visualization 
    # ======================================================
    print(f"[INFO] Performing edge bundling for {dataset_name}...")

    # --- Build node and edge tables based on FA2 layout ---
    nodes_df = (
        pd.DataFrame([
            {"id": str(n), "x": pos[0], "y": pos[1]}
            for n, pos in force_directed_positions.items()
            if np.all(np.isfinite(pos))
        ])
        .set_index("id")
    )

    edges_df = pd.DataFrame([
        {"source": str(u), "target": str(v)}
        for u, v in pns_graph.edges()
        if str(u) in nodes_df.index and str(v) in nodes_df.index
    ])

    # --- Run hammer_bundle ---
    bundled = bd.hammer_bundle(
        nodes_df,
        edges_df,
        initial_bandwidth=0.05,
        decay=0.6,
        iterations=8,
    )

    # --- Normalize schema for all Datashader versions ---
    if isinstance(bundled.index, pd.MultiIndex):
        bundled = bundled.reset_index(names=["edge_id", "point_id"])
    elif "edge_id" not in bundled.columns:
        bundled = bundled.reset_index(drop=True)
        bundled["edge_id"] = np.arange(len(bundled)) // 20  # fallback grouping

    n_edges_bundled = bundled["edge_id"].nunique()
    print(f"[INFO] Bundling complete ({n_edges_bundled} edges).")

    # --- Save bundled results ---
    bundled.to_csv(f"{output_folder}/{dataset_name}_bundled_edges.csv", index=False)
    # Also export for NDlib animation
    control_paths = []
    for edge_id, group in bundled.groupby("edge_id"):
        control_paths.append(list(map(lambda xy: (float(xy[0]), float(xy[1])), zip(group.x, group.y))))

    # ======================================================
    # Plot Bundled Layout (with FA2 nodes)
    # ======================================================
    plt.figure(figsize=(8, 8))

    # Draw bundled curved edges
    for edge_id, group in bundled.groupby("edge_id"):
        plt.plot(group.x, group.y, color="gray", alpha=0.25, linewidth=0.6, zorder=1)

    # Draw all nodes
    xs = [p[0] for p in force_directed_positions.values()]
    ys = [p[1] for p in force_directed_positions.values()]
    plt.scatter(xs, ys, s=10, color="royalblue", alpha=0.8, zorder=2)

    # Highlight suns (top nodes by CI)
    if top_nodes:
        sun_xs = [force_directed_positions[s][0] for s in top_nodes if s in force_directed_positions]
        sun_ys = [force_directed_positions[s][1] for s in top_nodes if s in force_directed_positions]
        plt.scatter(sun_xs, sun_ys, color="gold", edgecolors="black", s=25, zorder=3, label="Top CI Nodes")

    plt.title(f"Edge Bundling Result – {dataset_name}")
    plt.axis("equal")
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{dataset_name}_bundled_plot_full.png", dpi=300)
    plt.close()

    print(f"[INFO] Bundled visualization saved for {dataset_name}.")


    # --- EPB parameters (bind to variables so they get recorded) ---
    k_epb = 2.0
    d_epb = 2.0
    smoothing_epb = 2

    control_point_lists, stats = edge_path_bundling_epb(
        G=pns_graph,
        pos=force_directed_positions,
        d=d_epb,
        k=k_epb,
        smoothing=smoothing_epb
    )

    # Convert force-directed layout positions to NumPy array
    force_directed_array = np.array([force_directed_positions[node] for node in nodes])

    # Calculate Neighborhood Preservation for Force-Directed Layout
    layout_distances_fd = pairwise_distances(force_directed_array)
    neighborhood_preservation_fd = calculate_neighborhood_preservation(graph_distances, layout_distances_fd, k=5)
    print(f"Neighborhood Preservation (Force-Directed): {neighborhood_preservation_fd:.4f}")
    runtime_stats["neighborhood_preservation_fd"] = neighborhood_preservation_fd

    # Calculating silhouette score for Force-Directed Layout
    best_k_fd, best_sil_score_fd = find_optimal_clusters(force_directed_array, max_k=10)
    print(f"Optimal number of clusters: {best_k_fd}, Silhouette Score: {best_sil_score_fd:.4f}")
    runtime_stats["silhouette_score_fd"] = best_sil_score_fd
    runtime_stats["best_k_fd"] = best_k_fd

    cluster_labels_fd = dbscan.fit_predict(force_directed_array)
    if len(set(cluster_labels_fd)) > 1:  # Ensure at least 2 clusters
        sil_score_DB_fd = silhouette_score(force_directed_array, cluster_labels_fd)
        runtime_stats["silhouette_score_DB_fd"] = sil_score_DB_fd
        print(f"DBSCAN Silhouette Score: {sil_score_DB_fd:.4f}")
    else:
        print("DBSCAN found fewer than 2 clusters, silhouette score not computed.")

    # Fairness A: Suns vs Non-Suns for Force-Directed Layout
    group_A_suns = list(top_nodes)
    group_B_others = [n for n in g.nodes() if n not in top_nodes]

    fairness_suns_vs_rest = calculate_fairness_metric(
        positions=force_directed_positions,
        graph_distances=graph_distances,
        group_A=group_A_suns,
        group_B=group_B_others
    )

    # Fairness B: Top 10% CI vs rest for Force-Directed Layout
    ci_sorted = sorted(node_ci, key=node_ci.get, reverse=True)
    cut = max(1, int(0.10 * len(ci_sorted)))   # ensures non-empty
    group_A_top10 = ci_sorted[:cut]
    group_B_rest = ci_sorted[cut:]

    fairness_top10_vs_rest = calculate_fairness_metric(
        positions=force_directed_positions,
        graph_distances=graph_distances,
        group_A=group_A_top10,
        group_B=group_B_rest
    )
    runtime_stats["fairness_suns_vs_rest"] = fairness_suns_vs_rest
    runtime_stats["fairness_top10_vs_rest"] = fairness_top10_vs_rest

    # Fairness C: Solar system vs rest for Force-Directed Layout
    fairness_solar_vs_rest = calculate_fairness_metric(
        positions=force_directed_positions,
        graph_distances=graph_distances,
        group_A=group_A_solar,   
        group_B=group_B_rest_solar
    )

    runtime_stats["fairness_solar_vs_rest"] = fairness_solar_vs_rest

    # --- FA2 Gini Stress ---
    node_stress_fa2 = compute_node_stress(
    positions=force_directed_positions,
    graph_distances=graph_distances
    )

    gini_stress_fa2 = gini_coefficient(list(node_stress_fa2.values()))
    runtime_stats["gini_stress_fa2"] = gini_stress_fa2

    print(f"[GINI STRESS] FA2: {gini_stress_fa2:.4f}")

    save_layout_plot_with_edges(force_directed_positions, pns_graph, dataset_name, "ForceAtlas2", output_folder, highlight_nodes=top_nodes)  
    runtime_stats["visualizations"].append(f"{output_folder}/{dataset_name}_FA2.png")

    save_bundled_edge_plot(control_point_lists, force_directed_positions, dataset_name, "FA2-edge_bundling", output_folder, sun_nodes=top_nodes)   
    runtime_stats["visualizations"].append(f"{output_folder}/{dataset_name}_epb.png")

    save_layout_plot_with_edges(force_directed_positions, pns_graph, dataset_name, "Highways", output_folder, highlight_nodes=top_nodes, highlight_edges=influence_edges)
    runtime_stats["visualizations"].append(f"{output_folder}/{dataset_name}_FA2.png") 

        
    # ======================================================
    # === Export layout + bundled edges for NDlib simulation ===
    # ======================================================
    layout_outdir = os.path.join(output_folder, f"{dataset_name}_layout_for_simulation")
    os.makedirs(layout_outdir, exist_ok=True)

    # 1) Save node coordinates
    with open(os.path.join(layout_outdir, "positions.json"), "w") as f:
        json.dump({n: list(map(float, p)) for n, p in force_directed_positions.items()}, f, indent=2)
    print(f"[EXPORT] Saved node positions for {dataset_name} ({len(force_directed_positions)} nodes).")

    with open(os.path.join(layout_outdir, "bundled_edges.json"), "w") as f:
        json.dump(control_paths, f, indent=2)

    # 2) Save solar role sets (if present)
    if "role_sets" in locals() and role_sets:
        with open(os.path.join(layout_outdir, "roles.json"), "w") as f:
            json.dump({k: list(v) for k, v in role_sets.items()}, f, indent=2)
        print(f"[EXPORT] Saved role sets: {', '.join(role_sets.keys())}.")
    
    dump_graph_edges_csv(
            pns_graph, 
            os.path.join(layout_outdir, "skeleton_edges.csv")
        )

    print(f"[EXPORT] Saved skeleton graph ({pns_graph.number_of_edges()} edges).")

    runtime_stats["total_pipeline_time"] = time() - pipeline_start

    # Save metadata
    # ---- Build a params dictionary to record the run configuration ----
    tsne_params = {
        "n_components": 2,
        "metric": "precomputed",
        "init": "random",
        "perplexity": getattr(tsne, "perplexity", None),
        "random_state": random_state,
    }

    fa2_params = {
        "max_iter": 100,
        "scaling_ratio": 2.0,
        "gravity": 1.0,
        "strong_gravity": False,
        "linlog": True,
        "dissuade_hubs": False,
        "jitter_tolerance": 1.0,
        "weight": "weight",
        "degree_weight_attribute": None,
    }

    thb_cfg = globals().get("TRIMMING_CFG", None)

    solar_cfg = globals().get("SOLAR_CFG", None)

    dbscan_params = {
        "eps": 0.5,
        "min_samples": 5,
    }

    params = {
        "tsne": tsne_params,
        "fa2_baseline": fa2_params,
        "thb_trimming": thb_cfg,
        "dbscan": dbscan_params,
        # "epb": epb_params,
        "solar": solar_cfg
    }

    # ---- Save metadata with hyperparameters ----
    save_metadata(dataset_name, output_folder, None, runtime_stats, params)

    gc.collect()
    del reduced_tsne, layout_distances, force_directed_array
    gc.collect()

    return runtime_stats


def dr_pipeline(datasets, output_folder="baseline_results", random_state=42, plot_with_edges=True):
    os.makedirs(output_folder, exist_ok=True)

    for dataset_name, dataset_info in datasets.items():

        # DEFAULT: run top-CI with 10%
        run_full_pipeline_single_config(
            dataset_name,
            dataset_info,
            sun_method="top_CI",
            sun_percent=0.10,
            output_folder=output_folder,
            random_state=random_state
        )

def run_experiment_grid(datasets, output_folder="experiment_grid", random_state=42):

    os.makedirs(output_folder, exist_ok=True)
    results = []

    for dataset_name, dataset_info in datasets.items():
        for method in EXPERIMENT_CFG["sun_methods"]:
            for pct in EXPERIMENT_CFG["sun_percentages"]:

                config_name = f"{dataset_name}_{method}_{pct}".replace("%","pct")
                config_folder = os.path.join(output_folder, config_name)
                os.makedirs(config_folder, exist_ok=True)

                print(f"\n=== Running {config_name} ===\n")

                stats = run_full_pipeline_single_config(
                    dataset_name=dataset_name,
                    dataset_info=dataset_info,
                    sun_method=method,
                    sun_percent=pct,
                    output_folder=config_folder,
                    random_state=random_state
                )

                stats["dataset"] = dataset_name
                stats["sun_method"] = method
                stats["sun_percent"] = pct
                results.append(stats)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_folder, "experiment_results.csv"), index=False)
    print("\nSaved experiment_results.csv\n")


datasets = {
    # "Rmining": {"filepath": "RM_VAIM_redacted.csv", "directed": False},
    # "school_network": {"filepath": "school_network_converted.csv", "directed": False},
    "InfoVis": {"filepath": "edges_weighted_redacted.csv", "directed": False},
    # "CollegeMessage": {"filepath": "CollegeMessage_weighted_redacted.csv", "directed": True},
    # "BitAlpha": {"filepath": "BitAlpha_redacted.csv", "directed": True},
    # "USAir97": {"filepath": "USAir97.csv", "directed": False},
    # "BitOTC": {"filepath": "BitOtc_weighted_redacted.csv", "directed": True},
    # "PGP": {"filepath": "pgp_giantcompo_weights_redacted.csv", "directed": False},
    # "enron": {"filepath": "enron_edge_list.csv", "directed": False},
    # "conference": {"filepath": "SFHH_conference_edges.csv", "directed": False},
    
    
}

if __name__ == "__main__":
    run_experiment_grid(datasets, output_folder="experiment_grid")