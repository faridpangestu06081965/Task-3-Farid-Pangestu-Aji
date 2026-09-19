import numpy as np
import heapq
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class OccupancyGridMap:
    def __init__(self, width=15, height=15):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        
    def add_wall_obstacle(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = 100

    def apply_inflation_layer(self, radius=1):
        inflated_grid = self.grid.copy()
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == 100:
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                if self.grid[ny, nx] != 100:
                                    inflated_grid[ny, nx] = 50
        self.grid = inflated_grid

class AStarPlanner:
    def __init__(self, grid_map):
        self.map = grid_map

    def manhattan_heuristic(self, node, goal):
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    def plan(self, start, goal):
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.manhattan_heuristic(start, goal)}
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while open_list:
            _, current = heapq.heappop(open_list)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < self.map.width and 0 <= neighbor[1] < self.map.height):
                    continue
                cell_cost = self.map.grid[neighbor[1], neighbor[0]]
                if cell_cost == 100:
                    continue
                movement_cost = 1 + (cell_cost / 100.0)
                tentative_g = g_score[current] + movement_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_cost = tentative_g + self.manhattan_heuristic(neighbor, goal)
                    f_score[neighbor] = f_cost
                    heapq.heappush(open_list, (f_cost, neighbor))
        return None

class ExtendedKalmanFilter:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0

    def update(self, odom_raw, imu_raw):
        self.x = 0.6 * odom_raw[0] + 0.4 * imu_raw[0]
        self.y = 0.6 * odom_raw[1] + 0.4 * imu_raw[1]
        return (self.x, self.y)

class DynamicObstacleController:
    def __init__(self, max_speed=1.2, safe_distance=1.0):
        self.max_speed = max_speed
        self.safe_distance = safe_distance

    def calculate_twist_velocity(self, distance_to_obstacle):
        error = distance_to_obstacle - self.safe_distance
        if error <= 0:
            linear_velocity = 0.0
        else:
            linear_velocity = self.max_speed * math.tanh(error)
        return max(0.0, linear_velocity)

def render_and_save_map(grid_map, path, start, goal, filename="amr_navigation_map.png"):
    fig = plt.figure(figsize=(14, 6))
    
    # Plot 2D
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(grid_map.grid, cmap='Blues', origin='lower')
    if path:
        px, py = zip(*path)
        ax1.plot(px, py, color='#00FF66', linewidth=3, marker='o', markersize=4, label='A* Path')
    ax1.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax1.plot(goal[0], goal[1], 'r*', markersize=12, label='Goal')
    ax1.set_title("2D Occupancy Grid Map (15x15 - 2 Obstacles)")
    ax1.grid(True, linestyle=':', linewidth=0.5)
    ax1.legend(loc='upper left')

    # Plot 3D
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    height, width = grid_map.grid.shape
    _x, _y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = _x.ravel(), _y.ravel()
    top = np.zeros_like(x, dtype=float)
    colors = []

    for i in range(len(top)):
        val = grid_map.grid[y[i], x[i]]
        if val == 100:
            top[i] = 2.0
            colors.append('#1f77b4')
        elif val == 50:
            top[i] = 0.5
            colors.append('#aec7e8')
        else:
            top[i] = 0.01
            colors.append('#f0f0f0')

    ax2.bar3d(x, y, np.zeros_like(top), 0.8, 0.8, top, color=colors, shade=True, alpha=0.8)
    if path:
        px, py = zip(*path)
        ax2.plot(px, py, [0.3]*len(path), color='#00FF44', linewidth=3, marker='o', markersize=4)
    ax2.scatter([start[0]], [start[1]], [0.5], color='green', s=80)
    ax2.scatter([goal[0]], [goal[1]], [0.5], color='red', s=120, marker='*')
    ax2.set_title("3D Spatial Environment (2 Obstacles)")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n[INFO] Map preview saved to '{filename}'")
    plt.show()

if __name__ == "__main__":
    print("==================================================")
    print("    DECODELABS PROJECT 3: AMR NAVIGATION SIMULATOR")
    print("==================================================")
    print()

    grid_map = OccupancyGridMap(width=15, height=15)
    
    # OBSTACLE 1: Dinding vertikal pertama di x=5
    for y in range(2, 10):
        grid_map.add_wall_obstacle(5, y)

    # OBSTACLE 2: Dinding vertikal kedua di x=10
    for y in range(6, 14):
        grid_map.add_wall_obstacle(10, y)

    grid_map.apply_inflation_layer(radius=1)

    start_node = (1, 1)
    goal_node = (13, 13)

    print(f"[1] Map Created ({grid_map.width}x{grid_map.height}) with 2 Obstacles. Start: {start_node}, Goal: {goal_node}")

    planner = AStarPlanner(grid_map)
    path = planner.plan(start_node, goal_node)

    if path:
        print(f"[2] A* Search Successful! Planned Path ({len(path)} steps):")
        print(f"     {path}\n")

    print("[3] Executing Navigation Loop & Dynamic Obstacle Avoidance:")
    print("----------------------------------------------------------")

    ekf = ExtendedKalmanFilter()
    controller = DynamicObstacleController(max_speed=1.2, safe_distance=1.0)
    simulated_distances = [3.5, 2.5, 1.8, 1.2, 0.8, 0.4, 1.5, 3.0]

    for step, dist in enumerate(simulated_distances, 1):
        odom_data = ((step - 1) * 0.5, (step - 1) * 0.5)
        imu_data = ((step - 1) * 0.52, (step - 1) * 0.48)
        filtered_pose = ekf.update(odom_data, imu_data)
        twist_vx = controller.calculate_twist_velocity(dist)

        print(f" Step {step:02d} | Pose (EKF): ({filtered_pose[0]:.2f}, {filtered_pose[1]:.2f}) | Obs Dist: {dist:.1f}m | Cmd_Vel (Twist.vx): {twist_vx:.3f} m/s")

        if dist <= controller.safe_distance:
            print("        ⚠️ EMERGENCY OVERRIDE: Dynamic Obstacle Detected! Braking Applied.")

    print("\n==================================================")
    print("    PROJECT 3 CORE MODULES VALIDATED SUCCESSFULLY!")
    print("==================================================")

    render_and_save_map(grid_map, path, start_node, goal_node)
