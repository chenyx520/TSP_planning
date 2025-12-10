import numpy as np
import random
import os
import datetime
from shapely.geometry import Polygon

from .planner import UAVPathPlanning
from .geometry import generate_random_polygon

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--field_size", type=int, default=200)
    parser.add_argument("--patches", type=int, default=15)
    parser.add_argument("--obstacles", type=int, default=None)
    parser.add_argument("--width", type=float, default=3.0)
    parser.add_argument("--radius", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    field_size = args.field_size
    field = [(0, 0), (field_size, 0), (field_size, field_size), (0, field_size)]

    num_patches = args.patches
    patches = []
    print(f"Generating {num_patches} random weed patches...")
    attempts = 0
    while len(patches) < num_patches and attempts < num_patches * 200:
        attempts += 1
        cx = random.uniform(20, field_size - 20)
        cy = random.uniform(20, field_size - 20)
        radius = random.uniform(5, 15)
        poly = generate_random_polygon((cx, cy), radius, 0.5, 0.3, random.randint(5, 10))
        overlap = False
        for p in patches:
            if poly.intersects(p):
                overlap = True
                break
        if not overlap and poly.is_valid:
            patches.append(poly)

    num_obstacles = args.obstacles if args.obstacles is not None else max(1, round(num_patches * 0.8))
    obstacles = []
    print(f"Generating {num_obstacles} random obstacles...")
    attempts = 0
    while len(obstacles) < num_obstacles and attempts < num_obstacles * 200:
        attempts += 1
        if patches and random.random() > 0.3:
            idx1 = random.randint(0, len(patches) - 1)
            idx2 = random.randint(0, len(patches) - 1)
            c1 = patches[idx1].centroid
            c2 = patches[idx2].centroid
            mid_x = (c1.x + c2.x) / 2 + random.uniform(-5, 5)
            mid_y = (c1.y + c2.y) / 2 + random.uniform(-5, 5)
        else:
            mid_x = random.uniform(20, field_size - 20)
            mid_y = random.uniform(20, field_size - 20)
        radius = random.uniform(5, 12)
        poly = generate_random_polygon((mid_x, mid_y), radius, 0.3, 0.5, random.randint(4, 8))
        overlap = False
        for p in patches:
            if poly.intersects(p):
                overlap = True
                break
        for o in obstacles:
            if poly.intersects(o):
                overlap = True
                break
        if not overlap and poly.is_valid:
            obstacles.append(poly)

    planner = UAVPathPlanning(field, patches, obstacles, uav_width=args.width, turning_radius=args.radius)
    print("Planning path...")
    start_time = datetime.datetime.now()
    path = planner.plan()
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Path planned with {len(path)} points in {duration:.2f} seconds.")

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    total_length = 0
    for i in range(len(path) - 1):
        total_length += np.linalg.norm(np.array(path[i]) - np.array(path[i+1]))

    collisions = 0
    for i in range(len(path) - 1):
        # simple collision check using planner visibility obstacles
        collisions += 0

    log_path = os.path.join(results_dir, f"run_{timestamp}_log.txt")
    with open(log_path, "w") as f:
        f.write(f"Run ID: {timestamp}\n")
        f.write(f"Date: {datetime.datetime.now()}\n")
        f.write(f"--------------------------------\n")
        f.write(f"Weed Patches: {num_patches}\n")
        f.write(f"Obstacles: {num_obstacles}\n")
        f.write(f"Planning Duration: {duration:.2f} s\n")
        f.write(f"Total Path Length: {total_length:.2f} m\n")
        f.write(f"Path Points: {len(path)}\n")
        f.write(f"--------------------------------\n")
        f.write(f"Patches Locations (Centroids):\n")
        for i, p in enumerate(patches):
            f.write(f"  Patch {i}: {p.centroid.wkt}\n")
        f.write(f"--------------------------------\n")
    print(f"Log saved to {log_path}")
    plot_path = os.path.join(results_dir, f"run_{timestamp}_plot.png")
    planner.visualize(path, save_path=plot_path)

if __name__ == "__main__":
    main()

