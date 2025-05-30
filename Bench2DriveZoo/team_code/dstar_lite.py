import math
import numpy as np
import matplotlib.pyplot as plt
import random


class Node:
    def __init__(self, x: int = 0, y: int = 0, cost: float = 0.0):
        self.x = x
        self.y = y
        self.cost = cost


class DStarLite:
    motions = [
        Node(1, 0, 1), Node(0, 1, 1), Node(-1, 0, 1), Node(0, -1, 1),
        Node(1, 1, math.sqrt(2)), Node(1, -1, math.sqrt(2)),
        Node(-1, 1, math.sqrt(2)), Node(-1, -1, math.sqrt(2))
    ]

    def __init__(self, width: int, height: int):
        self.x_max = width
        self.y_max = height
        self.g = np.full((width, height), float('inf'))
        self.rhs = np.full((width, height), float('inf'))
        self.obstacles = set()
        self.U = []
        self.km = 0.0
        self.start = Node()
        self.goal = Node()
        self.initial_start = Node(self.start.x, self.start.y)
        self.last_direction = None 

    def is_valid(self, node: Node):
        return 0 <= node.x < self.x_max and 0 <= node.y < self.y_max

    def is_obstacle(self, node: Node):
        return (node.x, node.y) in self.obstacles

    def c(self, node1: Node, node2: Node):
        if self.is_obstacle(node2):
            return math.inf
        delta = Node(node2.x - node1.x, node2.y - node1.y)
        for motion in self.motions:
            if (motion.x == delta.x) and (motion.y == delta.y):
                return motion.cost
        return math.inf

    def h(self, s: Node):
        dx = abs(self.initial_start.x - s.x)
        dy = abs(self.initial_start.y - s.y)
        D = 1
        D2 = math.sqrt(2)
        base_heuristic = D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
        return base_heuristic

    def direction_change_penalty(self, s: Node):
        if self.last_direction is None:
            return 0
        new_direction = (s.x - self.start.x, s.y - self.start.y)
        if new_direction == (0, 0):
            return 0
        last_dx, last_dy = self.last_direction
        ndx, ndy = new_direction
        dot_product = last_dx * ndx + last_dy * ndy
        norm_last = math.hypot(last_dx, last_dy)
        norm_new = math.hypot(ndx, ndy)
        if norm_last != 0 and norm_new != 0:
            cosine_similarity = dot_product / (norm_last * norm_new)
            cosine_similarity = max(min(cosine_similarity, 1.0), -1.0)
            return 1 - cosine_similarity 
        else:
            return 0

    def path_deviation_penalty(self, s: Node, a_star_path):
        min_dist = min(math.hypot(s.x - px, s.y - py)
                       for (px, py) in a_star_path)
        return min_dist 

    def calculate_key(self, s: Node):
        min_val = min(self.g[s.x][s.y], self.rhs[s.x][s.y])
        return (min_val + self.h(s) + self.km, min_val)

    def get_neighbours(self, node: Node):
        return [Node(node.x + m.x, node.y + m.y) for m in self.motions if
                self.is_valid(Node(node.x + m.x, node.y + m.y))]

    def initialize(self, start: Node, goal: Node):
        self.start = start
        self.goal = goal

    def update_vertex(self, u: Node):
        if not (u.x == self.goal.x and u.y == self.goal.y):
            self.rhs[u.x][u.y] = min([self.c(u, s) + self.g[s.x][s.y]
                                      for s in self.get_neighbours(u)] or [float('inf')])
        self.U = [(n, k) for n, k in self.U if not (n.x == u.x and n.y == u.y)]
        if self.g[u.x][u.y] != self.rhs[u.x][u.y]:
            self.U.append((u, self.calculate_key(u)))
        self.U.sort(key=lambda x: x[1])

    def compare_keys(self, k1, k2):
        return k1[0] < k2[0] or (k1[0] == k2[0] and k1[1] < k2[1])

    def compute_shortest_path(self):
        while self.U and (
                self.compare_keys(self.U[0][1], self.calculate_key(self.start)) or
                self.rhs[self.start.x][self.start.y] != self.g[self.start.x][self.start.y]
        ):
            u, k_old = self.U.pop(0)
            if self.compare_keys(k_old, self.calculate_key(u)):
                self.U.append((u, self.calculate_key(u)))
            elif self.g[u.x][u.y] > self.rhs[u.x][u.y]:
                self.g[u.x][u.y] = self.rhs[u.x][u.y]
                for s in self.get_neighbours(u):
                    self.update_vertex(s)
            else:
                self.g[u.x][u.y] = float('inf')
                for s in self.get_neighbours(u) + [u]:
                    self.update_vertex(s)
            self.U.sort(key=lambda x: x[1])

    def inject_a_star_path(self, path_tuples):
        total_cost = 0.0
        path_nodes = [Node(x, y) for (x, y) in path_tuples]
        for i in reversed(range(len(path_nodes))):
            node = path_nodes[i]
            self.g[node.x][node.y] = total_cost
            self.rhs[node.x][node.y] = total_cost
            if i > 0:
                cost = self.c(path_nodes[i - 1], node)
                total_cost += cost if cost != math.inf else 1e9

    def sense_and_update_obstacles(self, obstacles, last_node: Node):
        changed = []
        for (ox, oy) in obstacles:
            if (ox, oy) not in self.obstacles:
                self.obstacles.add((ox, oy))
                changed.append(Node(ox, oy))

        self.km += self.h(last_node)
        last_node = Node(self.start.x, self.start.y)

        for u in changed:
            self.rhs[u.x][u.y] = math.inf
            self.g[u.x][u.y] = math.inf
            self.update_vertex(u)
            for s in self.get_neighbours(u):
                self.update_vertex(s)
        self.compute_shortest_path()
        return last_node

    def visualize_path(self, original_path, new_path, obstacles):
        grid = np.ones((self.y_max, self.x_max))
        for ox, oy in obstacles:
            grid[oy][ox] = 0

        fig, ax = plt.subplots()
        ax.imshow(grid, cmap='gray_r')

        if original_path:
            ox, oy = zip(*original_path)
            ax.plot(ox, oy, color='blue', linewidth=2, label='A* Path')

        if new_path:
            nx, ny = zip(*[(n.x, n.y) for n in new_path])
            ax.plot(nx, ny, color='red', linewidth=2, label='D* Lite Path')

        ax.plot(original_path[0][0], original_path[0]
        [1], 'go', markersize=10, label='Start')
        ax.plot(original_path[-1][0], original_path[-1]
        [1], 'ro', markersize=10, label='Goal')

        ax.set_xticks(np.arange(-0.5, self.x_max, 1))
        ax.set_yticks(np.arange(-0.5, self.y_max, 1))
        ax.grid(True, which='both')
        ax.legend()
        ax.set_title("D* Lite with A* Init")
        plt.gca().invert_yaxis()
        plt.show()

    def run_navigation(self, a_star_path, obstacle_list):
        last = self.start
        if obstacle_list:
            last = self.sense_and_update_obstacles(obstacle_list, last)

        path = [self.start]
        while not (self.start.x == self.goal.x and self.start.y == self.goal.y):
            successors = [s for s in self.get_neighbours(self.start)
                        if not self.is_obstacle(s) and self.g[s.x][s.y] != float('inf')]
            if not successors:
                print("[DEBUG] Dead end")
                break

            next_node = min(successors, key=lambda s:
                            self.c(self.start, s) +
                            self.g[s.x][s.y] +
                            self.direction_change_penalty(s) + 
                            self.path_deviation_penalty(s, a_star_path))

            self.last_direction = (
                next_node.x - self.start.x, next_node.y - self.start.y)
            dx = int(math.copysign(1, self.last_direction[0])) if self.last_direction[0] != 0 else 0
            dy = int(math.copysign(1, self.last_direction[1])) if self.last_direction[1] != 0 else 0
            self.last_direction = (dx, dy)
            self.start = next_node
            path.append(self.start)

        return path
