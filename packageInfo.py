import random
import numpy as np
import heapq

class Package:
    def __init__(self, pickup, dropoff, length=1, width=1, height=1, weight=1):
        self.pickup = pickup
        self.dropoff = dropoff
        self.length = length
        self.width = width
        self.height = height
        self.weight = weight

    def volume(self):
        return self.length * self.width * self.height

class Car:
    def __init__(self, id, start):
        self.id = id
        self.start = start
        self.package = None
        self.path = []
        self.current_index = 0
        self.color = np.random.rand(3,)

    def assign_package(self, package, hroads, vroads, shelves):
        self.package = package
        _, _, _, full_path = build_full_path(
            self.start, package.pickup, package.dropoff, hroads, vroads, shelves
        )
        self.path = full_path

def generate_random_pickup(grid_size, start, end, shelf_locations, existing_pickups=[]):
    available_locations = [loc for loc in shelf_locations if loc != start and loc != end and loc not in existing_pickups]
    if not available_locations:
        raise ValueError("No valid pickup locations available.")
    return random.choice(available_locations)

def a_star_search(start, goal, hroads, vroads, shelves):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: manhattan_distance(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        x, y = current
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

        for nx, ny in neighbors:
            if 0 <= nx < 10 and 0 <= ny < 10 and (nx, ny):
                if nx > x:
                    cost = hroads[x, y]
                elif nx < x:
                    cost = hroads[nx, y]
                elif ny > y:
                    cost = vroads[x, y]
                else:
                    cost = vroads[x, ny]

                g = g_score[current] + cost
                if (nx, ny) not in g_score or g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = g
                    f_score[(nx, ny)] = g + manhattan_distance((nx, ny), goal)
                    heapq.heappush(open_set, (f_score[(nx, ny)], (nx, ny)))
                    came_from[(nx, ny)] = current

    return []

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def build_full_path(start, pickup, dropoff, hroads, vroads, shelves):
    path_pickup = a_star_search(start, pickup, hroads, vroads, shelves)
    path_delivery = a_star_search(pickup, dropoff, hroads, vroads, shelves)
    path_return = a_star_search(dropoff, start, hroads, vroads, shelves)
    full_path = path_pickup + path_delivery[1:] + path_return[1:]
    return path_pickup, path_delivery, path_return, full_path

def setup_multiple_cars_and_packages(num_cars, shelf_locations, hroads, vroads, shelves):
    cars = [Car(id=i, start=(0, i)) for i in range(num_cars)]
    packages = []
    used_locations = []

    for car in cars:
        pickup = generate_random_pickup(10, car.start, (0, 0), shelf_locations, used_locations)
        dropoff = generate_random_pickup(10, pickup, car.start, shelf_locations, used_locations + [pickup])
        package = Package(pickup=pickup, dropoff=dropoff)
        packages.append(package)
        used_locations.extend([pickup, dropoff])
        car.assign_package(package, hroads, vroads, shelves)

    return cars, packages

def avoid_collision(*paths):
    max_len = max(len(path) for path in paths)  
    
    paths = [list(path) for path in paths]
    
    for t in range(max_len):
        for i, path in enumerate(paths):
            if t < len(path):  
                current_position = path[t]
                
                for j in range(i + 1, len(paths)):
                    if t < len(paths[j]) and current_position == paths[j][t]:
                        print(f"Collision detected between car {i+1} and car {j+1} at timestep {t}")
                        paths[i] = paths[i][:t] + [paths[i][t-1]] + paths[i][t+1:]  
                        paths[j] = paths[j][:t] + [paths[j][t-1]] + paths[j][t+1:]  
    
    return paths

def find_available_position(paths, occupied_positions, path, t, position):
    possible_moves = [
        (position[0] + 1, position[1]),  # Move right
        (position[0] - 1, position[1]),  # Move left
        (position[0], position[1] + 1),  # Move up
        (position[0], position[1] - 1),  # Move down
    ]
    
    for move in possible_moves:
        if (0 <= move[0] < 10 and 0 <= move[1] < 10 and  
            move not in occupied_positions and  
            move not in [p[t] for p in paths if t < len(p)]): 
            return move  
    
    return position  

# Function to calculate total time based on the length of the paths
def calculate_total_time(paths):
    total_time = 0
    for path in paths:
        total_time += len(path) 
    return total_time
