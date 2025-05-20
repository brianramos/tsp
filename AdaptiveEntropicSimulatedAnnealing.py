import math
import random
import time
import os
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist

class City:
    def __init__(self, x, y, name=None):
        self.x = x
        self.y = y
        self.name = name if name else str(id(self))
        self.visited = False
        self.entropy = 1.0

def calculate_initial_entropies(cities, density_radius_base=20, k_nearest=5):
    distances = cdist([[city.x, city.y] for city in cities], [[city.x, city.y] for city in cities])
    np.fill_diagonal(distances, np.inf)

    initial_entropies = {}
    for i, city in enumerate(cities):
        nearest_distances = np.sort(distances[i])
        density_radius = density_radius_base * np.mean(nearest_distances[:min(k_nearest, len(nearest_distances))])
        initial_entropies[city] = calculate_density(city, cities, density_radius, distances=distances[i]) # Pass distances here
    return initial_entropies

def calculate_density(city, cities, radius, distances=None):
    if distances is None:  # Calculate distances if not provided
        distances = [euclidean_distance(city, other) for other in cities if other != city]
        count = np.sum(np.array(distances) <= radius) # Convert to numpy array for comparison
    else: # Use pre-calculated distances if provided
        count = np.sum(distances <= radius)
    return count

def calculate_global_entropy(cities, unvisited_cities, grid_resolution=None, use_weighted_entropy=False):
    # Optimization: Use bounding box for grid dimensions
    min_x = min(city.x for city in cities)
    min_y = min(city.y for city in cities)
    width = max(city.x for city in cities) - min_x
    height = max(city.y for city in cities) - min_y

    # Adaptive grid resolution based on city distribution
    k = 5  # Number of nearest neighbors to consider
    distances = cdist([[city.x, city.y] for city in unvisited_cities], [[city.x, city.y] for city in unvisited_cities])
    np.fill_diagonal(distances, np.inf)
    mean_nearest_distance = np.mean(np.sort(distances)[:, :k])
    grid_resolution = int(max(width, height) / mean_nearest_distance) + 1  # Adapt resolution

    num_cells = grid_resolution
    grid = np.zeros((num_cells, num_cells))
    cell_width = width / num_cells
    cell_height = height / num_cells

    for city in unvisited_cities:
        x = int((city.x - min_x) / cell_width)
        y = int((city.y - min_y) / cell_height)
        x = min(x, num_cells - 1)  # Ensure within bounds
        y = min(y, num_cells - 1)

        if use_weighted_entropy:
            cell_center_x = x * cell_width + cell_width / 2 + min_x
            cell_center_y = y * cell_height + cell_height / 2 + min_y
            weight = 1 / (1 + euclidean_distance(city, City(cell_center_x, cell_center_y)))
            grid[x, y] += weight
        else:
            grid[x, y] += 1

    probabilities = grid[grid > 0] / len(unvisited_cities)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_directional_entropy(city, unvisited_cities, num_directions=4):
    directional_entropies = [0.0] * num_directions
    for other_city in unvisited_cities:
        if other_city != city:
            angle = math.atan2(other_city.y - city.y, other_city.x - city.x)
            direction = int(angle / (2 * math.pi) * num_directions) % num_directions
            directional_entropies[direction] += 1

    probabilities = [count / len(unvisited_cities) for count in directional_entropies]
    entropies = [-p * math.log2(p) if p > 0 else 0 for p in probabilities]
    return entropies

def euclidean_distance(city1, city2):
    return math.sqrt((city1.x - city2.x)**2 + (city1.y - city2.y)**2)

def path_length(path):
    length = 0
    for i in range(len(path) - 1):
        length += euclidean_distance(path[i], path[i+1])
    length += euclidean_distance(path[-1], path[0])
    return length

def nearest_neighbor(cities):
    num_cities = len(cities)
    path = []
    current_city = random.choice(cities)
    current_city.visited = True
    path.append(current_city)

    for _ in range(num_cities - 1):
        min_distance = float('inf')
        next_city = None
        for city in cities:
            if not city.visited:
                distance = euclidean_distance(current_city, city)
                if distance < min_distance:
                    min_distance = distance
                    next_city = city

        if next_city:
            next_city.visited = True
            path.append(next_city)
            current_city = next_city
    return path

def two_opt(cities):
  num_cities = len(cities)
  path = cities.copy()
  improved = True
  while improved:
    improved = False
    for i in range(1, num_cities - 2):
        for j in range(i + 1, num_cities):
            if j - i == 1: continue  # Skip adjacent swaps
            new_path = path[:i] + path[i:j][::-1] + path[j:]
            if path_length(new_path) < path_length(path):
                path = new_path
                improved = True
                break # Exit inner loop early if improvement found
        if improved:
            break # Exit outer loop early if improvement found
  return path


def adaptive_contract_bubble(city, stretch, bubble_sigma, density_radius, cities=None, influence_fields=None):
    density = calculate_density(city, cities, density_radius)
    adjusted_bubble_strength = bubble_sigma * (1 - density / len(cities))
    influence_factor = 1.0
    if influence_fields is not None:
        influence_factor = influence_fields[city]
    city.entropy = max(0, city.entropy - (0.1 + 0.05 * stretch) * adjusted_bubble_strength * influence_factor)

def entropic_distance(city1, city2, stretch, initial_distances, global_entropy, directional_entropies):
    initial_distance = initial_distances[(city1, city2)]
    entropy_factor = 1 / (city1.entropy * city2.entropy + 1e-10) # Added small constant to avoid division by zero

    angle = math.atan2(city2.y - city1.y, city2.x - city1.x)
    direction = int(angle / (2 * math.pi) * len(directional_entropies)) % len(directional_entropies)
    directional_entropy_factor = 1 / (1 + directional_entropies[direction])

    stretch_factor = 1 + 0.1 * stretch
    global_entropy_weight = 1 + (global_entropy * 0.2)
    return initial_distance * entropy_factor * stretch_factor / global_entropy_weight * directional_entropy_factor

def traveling_salesperson_entropic_adaptive(cities, start_city, bubble_sigma, density_radius_base, grid_resolution):
    path = []
    current_city = start_city
    current_city.visited = True
    path.append(current_city)

    unvisited_cities = [city for city in cities if city != current_city]
    initial_entropies = calculate_initial_entropies(cities, density_radius_base)
    for city, entropy in initial_entropies.items():
        city.entropy = entropy

    initial_distances = {}
    for c1 in cities:
        for c2 in cities:
            initial_distances[(c1, c2)] = euclidean_distance(c1, c2)

    influence_fields = defaultdict(lambda: 1.0)
    total_stretch = 0

    while unvisited_cities:
        global_entropy = calculate_global_entropy(cities, unvisited_cities)
        directional_entropies = calculate_directional_entropy(current_city, unvisited_cities)

        distances = {city: entropic_distance(current_city, city, total_stretch, initial_distances, global_entropy, directional_entropies) for city in unvisited_cities}
        nearest_cities = sorted(distances, key=distances.get)
        next_city = nearest_cities[0]

        next_city.visited = True
        path.append(next_city)
        unvisited_cities.remove(next_city)

        adaptive_contract_bubble(next_city, total_stretch, bubble_sigma, density_radius_base, cities=cities, influence_fields=influence_fields)
        current_city = next_city

    path_length_val = path_length(path)
    for city in cities:
        city.visited = False # Reset visited status
    return path, path_length_val

def traveling_salesperson_entropic_brute_start(cities, initial_params=(0.5, 20, 12)):
    best_path = None
    min_length = float('inf')

    for start_city in cities:
        path, length = traveling_salesperson_entropic_adaptive(cities.copy(), start_city, *initial_params)
        if length < min_length:
            min_length = length
            best_path = path.copy()

    return best_path

def reset_cities(cities):
    for city in cities:
        city.visited = False
        city.entropy = 1.0

def read_tsp_file(filepath):
    cities = []
    reading_coords = False
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip().startswith("NODE_COORD_SECTION"):
                reading_coords = True
                continue
            elif line.strip().startswith("EOF") or line.strip().startswith("DISPLAY_DATA_SECTION"):
                reading_coords = False
            elif reading_coords and line.strip() and not line.strip().startswith("EOF"):
                parts = line.split()
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                    cities.append(City(x, y)) # City name not used, so simplified
                except (ValueError, IndexError):
                    print(f"Skipping invalid line: {line.strip()}")

    return cities

def test_all_tsp_files(directory="."):
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith(".tsp"):
            filepath = os.path.join(directory, filename)
            print(f"Testing {filename}...")
            cities = read_tsp_file(filepath)

            results[filename] = {}

            start_time = time.time()
            path = traveling_salesperson_entropic_brute_start(cities.copy())
            length = path_length(path)
            results[filename]["entropic"] = {"length": length, "time": time.time() - start_time}


            start_time = time.time()
            reset_cities(cities)
            path = nearest_neighbor(cities.copy())
            results[filename]["nearest_neighbor"] = {"length": path_length(path), "time": time.time() - start_time}


            start_time = time.time()
            reset_cities(cities)
            path = two_opt(cities.copy())
            results[filename]["two_opt"] = {"length": path_length(path), "time": time.time() - start_time}

            for algorithm in ["entropic", "nearest_neighbor", "two_opt"]:
                print(f"    {algorithm}: Length = {results[filename][algorithm]['length']}, Time = {results[filename][algorithm]['time']:.2f}s")

    return results

results = test_all_tsp_files()

for filename, algorithms in results.items():
    print(f"\nSummary for {filename}:")
    for algorithm, data in algorithms.items():
        print(f"    {algorithm}: Length = {data['length']}, Time = {data['time']:.2f}s")
