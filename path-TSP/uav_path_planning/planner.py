import math
import numpy as np
import networkx as nx
from shapely.geometry import Point

from .visibility import build_visibility_graph
from .coverage import generate_coverage_path, optimize_scan_angle
from .tsp_solver import nearest_neighbor_order, two_opt, orientation_dp, branch_and_bound_order

class UAVPathPlanning:
    def __init__(self, field_bounds, patches, obstacles, uav_width=1.0, turning_radius=1.0):
        self.field = Point(0,0).buffer(1.0)
        self.field = self.field.envelope
        self.field = self.field.from_bounds(field_bounds[0][0], field_bounds[0][1], field_bounds[2][0], field_bounds[2][1])
        self.patches = patches
        self.obstacles = obstacles
        self.uav_width = uav_width
        self.turning_radius = turning_radius
        self.visibility_graph = None

    def build_visibility_graph(self, points_of_interest):
        G = build_visibility_graph(self.obstacles, self.uav_width, points_of_interest)
        self.visibility_graph = G
        return G

    def find_path_avoiding_obstacles(self, start, end):
        if self.visibility_graph is not None:
            s_node = tuple(start) if not isinstance(start, tuple) else start
            e_node = tuple(end) if not isinstance(end, tuple) else end
            if self.visibility_graph.has_node(s_node) and self.visibility_graph.has_node(e_node):
                try:
                    return nx.shortest_path(self.visibility_graph, source=s_node, target=e_node, weight='weight')
                except nx.NetworkXNoPath:
                    pass
        s_node = tuple(start) if not isinstance(start, tuple) else start
        e_node = tuple(end) if not isinstance(end, tuple) else end
        try:
            G = self.build_visibility_graph([s_node, e_node])
            return nx.shortest_path(G, source=s_node, target=e_node, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [start, end]

    def visualize(self, path=None, save_path=None):
        import matplotlib.pyplot as plt
        x = [0, self.field.bounds[2], self.field.bounds[2], 0, 0]
        y = [0, 0, self.field.bounds[3], self.field.bounds[3], 0]
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(x, y, 'k-', linewidth=2, label='Field Boundary')
        for obs in self.obstacles:
            xx, yy = obs.exterior.xy
            ax.fill(xx, yy, 'r', alpha=0.5, label='Obstacle')
        for patch in self.patches:
            xx, yy = patch.exterior.xy
            ax.fill(xx, yy, 'g', alpha=0.5, label='Weed Patch')
        if path:
            path_x, path_y = zip(*path)
            ax.plot(path_x, path_y, 'b--', linewidth=1, label='UAV Path')
            ax.plot(path_x[0], path_y[0], 'bo', label='Start')
            ax.plot(path_x[-1], path_y[-1], 'bx', label='End')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.axis('equal')
        plt.grid(True)
        plt.title('UAV Path Planning for Spot Spraying')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def smooth_path(self, path_points, max_deviation=1.5):
        if len(path_points) < 3:
            return path_points
        smoothed = [path_points[0]]
        for i in range(1, len(path_points) - 1):
            p_prev = np.array(path_points[i-1])
            p_curr = np.array(path_points[i])
            p_next = np.array(path_points[i+1])
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            if len1 < 1e-3 or len2 < 1e-3:
                smoothed.append(path_points[i])
                continue
            u1 = v1 / len1
            u2 = v2 / len2
            dot = np.dot(u1, u2)
            if dot > 0.99:
                smoothed.append(path_points[i])
                continue
            d = min(len1 / 2.5, len2 / 2.5, max_deviation * 3)
            p_start = p_curr - u1 * d
            p_end = p_curr + u2 * d
            num_steps = 5
            for t in np.linspace(0, 1, num_steps):
                pt = (1-t)**2 * p_start + 2*(1-t)*t * p_curr + t**2 * p_end
                smoothed.append(tuple(pt))
        smoothed.append(path_points[-1])
        return smoothed

    def solve_global_optimization(self, start_pos):
        n = len(self.patches)
        patches_data = []
        for patch in self.patches:
            path = optimize_scan_angle(patch, self.uav_width)
            if not path:
                path = [patch.centroid.coords[0], patch.centroid.coords[0]]
            length = sum(np.linalg.norm(np.array(path[k]) - np.array(path[k+1])) for k in range(len(path)-1))
            patches_data.append({'path': path, 'start': path[0], 'end': path[-1], 'len': length})
        all_points = [start_pos]
        for p in patches_data:
            all_points.append(p['start'])
            all_points.append(p['end'])
        all_points = list(set([tuple(p) if not isinstance(p, tuple) else p for p in all_points]))
        
        use_euclidean_tsp = n > 20
        # Always build visibility graph for final path generation to avoid rebuilding it for every segment
        self.build_visibility_graph(all_points)
        print("Visibility graph built. Calculating distance matrix...")
        
        def get_dist(p1, p2):
            if use_euclidean_tsp:
                return np.linalg.norm(np.array(p1) - np.array(p2))
            try:
                s = tuple(p1)
                e = tuple(p2)
                return nx.shortest_path_length(self.visibility_graph, source=s, target=e, weight='weight')
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return np.linalg.norm(np.array(p1) - np.array(p2))
        dist_matrix = np.zeros((n, 2, n, 2))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dist_matrix[i][0][j][0] = get_dist(patches_data[i]['end'], patches_data[j]['start'])
                dist_matrix[i][0][j][1] = get_dist(patches_data[i]['end'], patches_data[j]['end'])
                dist_matrix[i][1][j][0] = get_dist(patches_data[i]['start'], patches_data[j]['start'])
                dist_matrix[i][1][j][1] = get_dist(patches_data[i]['start'], patches_data[j]['end'])
        start_dist = np.zeros((n, 2))
        for i in range(n):
            start_dist[i][0] = get_dist(start_pos, patches_data[i]['start'])
            start_dist[i][1] = get_dist(start_pos, patches_data[i]['end'])
        return_dist = np.zeros((n, 2))
        for i in range(n):
            return_dist[i][0] = get_dist(patches_data[i]['end'], start_pos)
            return_dist[i][1] = get_dist(patches_data[i]['start'], start_pos)
        best_path_indices = []
        # Threshold for using Branch and Bound (e.g. 15). For N <= 15, B&B is feasible and optimal.
        # For N > 15, we use Nearest Neighbor + 2-opt.
        if n > 15:
            centroids = [tuple((self.patches[i].centroid.x, self.patches[i].centroid.y)) for i in range(n)]
            order = nearest_neighbor_order(start_pos, centroids)
            def dist_fn(a_idx, b_idx):
                return get_dist(centroids[a_idx], centroids[b_idx])
            start_cost = lambda i: get_dist(start_pos, centroids[i])
            end_cost = lambda i: get_dist(centroids[i], start_pos)
            order = two_opt(order, lambda a,b: dist_fn(a,b), start_cost, end_cost)
            best_path_indices = orientation_dp(order, patches_data, dist_matrix, start_dist, return_dist)
            print("TSP solved (Heuristic). Constructing final path...")
        else:
            print(f"Using Branch and Bound for TSP (N={n})...")
            centroids = [tuple((self.patches[i].centroid.x, self.patches[i].centroid.y)) for i in range(n)]
            def dist_fn_idx(i, j):
                return get_dist(centroids[i], centroids[j])
            start_cost_fn = lambda i: get_dist(start_pos, centroids[i])
            return_cost_fn = lambda i: get_dist(centroids[i], start_pos)
            
            # Use the Branch and Bound solver from tsp_solver.py
            # Note: This B&B solver optimizes the visitation order of centroids.
            # Orientation is still handled by DP afterwards, or we could integrate it if we had a GTSP solver.
            # The current implementation in tsp_solver.py seems to optimize order based on centroid distances + patch length.
            # Let's verify what branch_and_bound_order expects.
            
            best_order = branch_and_bound_order(start_pos, centroids, patches_data, dist_fn_idx, start_cost_fn, return_cost_fn)
            best_path_indices = orientation_dp(best_order, patches_data, dist_matrix, start_dist, return_dist)
            print("TSP solved (Branch & Bound). Constructing final path...")
        
        full_path = []
        curr_pos = start_pos
        for idx, direction in best_path_indices:
            p_data = patches_data[idx]
            path_segment = p_data['path']
            if direction == 1:
                path_segment = path_segment[::-1]
            target_entry = path_segment[0]
            if np.linalg.norm(np.array(curr_pos) - np.array(target_entry)) > 1e-3:
                transit = self.find_path_avoiding_obstacles(curr_pos, target_entry)
                if len(transit) > 2:
                    transit = self.smooth_path(transit)
                full_path.extend(transit[:-1])
            full_path.extend(path_segment)
            curr_pos = path_segment[-1]
        transit = self.find_path_avoiding_obstacles(curr_pos, start_pos)
        if len(transit) > 2:
            transit = self.smooth_path(transit)
        full_path.extend(transit)
        return full_path

    def plan(self):
        return self.solve_global_optimization((0, 0))
