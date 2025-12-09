import math
import networkx as nx
from shapely.geometry import LineString
from shapely.prepared import prep

def build_visibility_graph(obstacles, uav_width, points_of_interest):
    G = nx.Graph()
    margin = uav_width/2 + 2.0
    inflated = [obs.buffer(margin, resolution=2).simplify(2.0, preserve_topology=True) for obs in obstacles]
    nodes = list(points_of_interest)
    for obs in inflated:
        hull = obs.convex_hull
        coords = list(hull.exterior.coords)
        target = 6
        step = max(1, int(len(coords)/target))
        sampled = coords[::step]
        nodes.extend(sampled)
    nodes = list(set(nodes))
    G.add_nodes_from(nodes)
    obs_bounds = [obs.bounds for obs in inflated]
    prepared_obstacles = [prep(obs) for obs in inflated]
    num_nodes = len(nodes)
    print(f"Building visibility graph with {num_nodes} nodes...")
    for i in range(len(nodes)):
        if i % 10 == 0:
            print(f"Processing node {i}/{num_nodes}...", end='\r')
        u = nodes[i]
        for j in range(i + 1, len(nodes)):
            v = nodes[j]
            lx_min = u[0] if u[0] < v[0] else v[0]
            lx_max = u[0] if u[0] > v[0] else v[0]
            ly_min = u[1] if u[1] < v[1] else v[1]
            ly_max = u[1] if u[1] > v[1] else v[1]
            line = None
            intersects = False
            for k, obs in enumerate(inflated):
                ox_min, oy_min, ox_max, oy_max = obs_bounds[k]
                if lx_max < ox_min or lx_min > ox_max or ly_max < oy_min or ly_min > oy_max:
                    continue
                if line is None:
                    line = LineString([u, v])
                if prepared_obstacles[k].intersects(line):
                    if not obs.touches(line):
                        intersects = True
                        break
            if not intersects:
                dist = math.hypot(u[0]-v[0], u[1]-v[1])
                G.add_edge(u, v, weight=dist)
    return G

