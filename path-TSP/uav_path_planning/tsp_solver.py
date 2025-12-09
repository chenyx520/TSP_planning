import numpy as np

def nearest_neighbor_order(start_pos, centroids):
    n = len(centroids)
    unvisited = set(range(n))
    order = []
    cur = min(unvisited, key=lambda i: _dist(start_pos, centroids[i]))
    order.append(cur)
    unvisited.remove(cur)
    while unvisited:
        nxt = min(unvisited, key=lambda i: _dist(centroids[cur], centroids[i]))
        order.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return order

def two_opt(order, dist_fn, start_cost, end_cost):
    n = len(order)
    def route_length(o):
        if not o:
            return 0.0
        s = start_cost(o[0])
        s += sum(dist_fn(o[k], o[k+1]) for k in range(n-1))
        s += end_cost(o[-1])
        return s
    improved = True
    while improved:
        improved = False
        for i in range(0, n-2):
            for j in range(i+2, n):
                new_order = order[:i+1] + order[i+1:j+1][::-1] + order[j+1:]
                if route_length(new_order) + 1e-6 < route_length(order):
                    order = new_order
                    improved = True
    return order

def orientation_dp(order, patches_data, dist_matrix, start_dist, return_dist):
    n = len(order)
    dp = np.full((n, 2), float('inf'))
    prev = [[-1, -1] for _ in range(n)]
    first = order[0]
    dp[0][0] = patches_data[first]['len'] + start_dist[first][0]
    dp[0][1] = patches_data[first]['len'] + start_dist[first][1]
    for i in range(1, n):
        a = order[i-1]
        b = order[i]
        for dprev in (0,1):
            for dcur in (0,1):
                c = dp[i-1][dprev] + patches_data[b]['len'] + dist_matrix[a][dprev][b][0 if dcur==0 else 1]
                if c < dp[i][dcur]:
                    dp[i][dcur] = c
                    prev[i][dcur] = dprev
    last_idx = order[-1]
    end_dir = 0 if dp[n-1][0] + return_dist[last_idx][0] <= dp[n-1][1] + return_dist[last_idx][1] else 1
    dirs = [0]*n
    dirs[n-1] = end_dir
    for i in range(n-1,0,-1):
        dirs[i-1] = prev[i][dirs[i]]
    return [(order[i], dirs[i]) for i in range(n)]

def _dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def branch_and_bound_order(start_pos, centroids, patches_data, dist_fn, start_cost, return_cost):
    n = len(centroids)
    dist_cc = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist_cc[i][j] = 0.0 if i == j else dist_fn(i, j)
    start_d = [start_cost(i) for i in range(n)]
    min_return_all = min(return_cost(i) for i in range(n))
    min_in_cost = []
    for j in range(n):
        best_in = start_d[j]
        for i in range(n):
            if i == j:
                continue
            if dist_cc[i][j] < best_in:
                best_in = dist_cc[i][j]
        min_in_cost.append(best_in)

    import heapq
    def lower_bound(cost_so_far, visited_mask):
        lb = cost_so_far
        rem_len = 0.0
        rem_in = 0.0
        for k in range(n):
            if not (visited_mask & (1 << k)):
                rem_len += patches_data[k]['len']
                rem_in += min_in_cost[k]
        lb += rem_len + rem_in + min_return_all
        return lb

    best_cost = float('inf')
    best_order = []
    pq = []  # (lb, cost, visited_mask, last_idx, order)
    for i in range(n):
        mask = (1 << i)
        cost0 = start_d[i] + patches_data[i]['len']
        lb0 = lower_bound(cost0, mask)
        heapq.heappush(pq, (lb0, cost0, mask, i, [i]))

    while pq:
        lb, cost, mask, last, order = heapq.heappop(pq)
        if lb >= best_cost:
            continue
        if mask == (1 << n) - 1:
            final_cost = cost + return_cost(last)
            if final_cost < best_cost:
                best_cost = final_cost
                best_order = order
            continue
        for j in range(n):
            if not (mask & (1 << j)):
                new_mask = mask | (1 << j)
                new_cost = cost + dist_cc[last][j] + patches_data[j]['len']
                new_lb = lower_bound(new_cost, new_mask)
                if new_lb < best_cost:
                    heapq.heappush(pq, (new_lb, new_cost, new_mask, j, order + [j]))

    return best_order if best_order else list(range(n))
