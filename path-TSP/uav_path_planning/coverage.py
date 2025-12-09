import numpy as np
from shapely.geometry import Point, LineString
from shapely import affinity

def generate_coverage_path(polygon, uav_width, angle):
    rotated_poly = affinity.rotate(polygon, -angle, origin='centroid')
    min_x, min_y, max_x, max_y = rotated_poly.bounds
    headland_width = uav_width / 2
    headland_poly = rotated_poly.buffer(-headland_width)
    if headland_poly.is_empty:
        scan_poly = rotated_poly
        use_headland = False
    else:
        scan_poly = headland_poly
        use_headland = True
    path_points = []
    y = min_y + headland_width + uav_width / 2
    direction = 1
    scan_min_x, scan_min_y, scan_max_x, scan_max_y = scan_poly.bounds
    while y < scan_max_y:
        line = LineString([(scan_min_x - 10, y), (scan_max_x + 10, y)])
        intersection = scan_poly.intersection(line)
        if intersection.is_empty:
            y += uav_width
            continue
        points = []
        if intersection.geom_type == 'LineString':
            points.extend(list(intersection.coords))
        elif intersection.geom_type == 'MultiLineString':
            for geom in intersection.geoms:
                points.extend(list(geom.coords))
        points.sort(key=lambda p: p[0])
        path_points.extend(points if direction == 1 else points[::-1])
        direction *= -1
        y += uav_width
    final_rotated_path = []
    if use_headland:
        if headland_poly.geom_type == 'Polygon':
            headland_coords = list(headland_poly.exterior.coords)
        elif headland_poly.geom_type == 'MultiPolygon':
            headland_coords = list(max(headland_poly.geoms, key=lambda a: a.area).exterior.coords)
        else:
            headland_coords = []
        final_rotated_path.extend(headland_coords)
    final_rotated_path.extend(path_points)
    final_path = []
    for p in final_rotated_path:
        pt = Point(p)
        rotated_pt = affinity.rotate(pt, angle, origin=polygon.centroid)
        final_path.append((rotated_pt.x, rotated_pt.y))
    return final_path

def optimize_scan_angle(patch, uav_width):
    best_path = []
    min_cost = float('inf')
    for angle in range(0, 180, 15):
        path = generate_coverage_path(patch, uav_width, angle)
        if not path:
            continue
        length = 0
        for i in range(len(path) - 1):
            length += np.linalg.norm(np.array(path[i]) - np.array(path[i+1]))
        if length < min_cost:
            min_cost = length
            best_path = path
    return best_path

