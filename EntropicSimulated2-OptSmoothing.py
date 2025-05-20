import math
import random
import time
import os
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.special import softmax  # Import softmax

class City:
    def __init__(self, x, y, name=None):
        self.x = x
        self.y = y
        self.name = name if name else str(id(self))
        self.visited = False
        self.entropy = 1.0

def euclidean_distance(city1, city2):
    return math.sqrt((city1.x - city2.x) ** 2 + (city1.y - city2.y) ** 2)

def path_length(path):
    length = 0
    for i in range(len(path) - 1):
        length += euclidean_distance(path[i], path[i + 1])
    length += euclidean_distance(path[-1], path[0])
    return length

def nearest_neighbor(cities, start_city=None):
    num_cities = len(cities)
    path = []

    # If start_city is not provided, choose a random one
    if start_city is None:
        start_city = random.choice(cities)

    current_city = start_city
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
                if j - i == 1:
                    continue  # Skip adjacent swaps
                new_path = path[:i] + path[i:j][::-1] + path[j:]
                if path_length(new_path) < path_length(path):
                    path = new_path
                    improved = True
                    break  # Exit inner loop early if improvement found
            if improved:
                break  # Exit outer loop early if improvement found
    return path

def two_opt_swap(route, i, k):
    """Performs a 2-opt swap on a route."""
    new_route = route[:i] + route[i:k + 1][::-1] + route[k + 1:]
    return new_route

def two_opt_entropic_smoothing(distance_matrix, initial_route, max_iterations=1000, initial_temperature=10.0, cooling_rate=0.995, min_temperature=0.1):
    """
    2-opt with entropic sampling for smoothing.

    Args:
        distance_matrix: NxN matrix of distances between cities.
        initial_route: Starting route (list of city indices).
        max_iterations: Maximum number of iterations.
        initial_temperature: Starting temperature for simulated annealing.
        cooling_rate: Rate at which the temperature decreases.
        min_temperature: Minimum temperature.

    Returns:
        Improved route.
    """
    current_route = initial_route.copy()
    current_distance = calculate_distance(current_route, distance_matrix)
    temperature = initial_temperature
    best_route = current_route.copy()
    best_distance = current_distance

    for iteration in range(max_iterations):
        n = len(current_route)
        distances_to_sample = []
        swap_indices = []

        # Generate potential swaps and their distances
        for _ in range(n):  # Consider 'n' potential swaps each iteration
            i = random.randint(0, n - 2)
            k = random.randint(i + 1, n - 1)
            new_route = two_opt_swap(current_route, i, k)
            new_distance = calculate_distance(new_route, distance_matrix)
            distances_to_sample.append(new_distance - current_distance)  # Store *difference* in distance
            swap_indices.append((i, k))

        # Entropic sampling:  Use softmax on *negative* distance differences (to favor improvements)
        selected_index = entropic_sampling(distances_to_sample, temperature)
        i, k = swap_indices[selected_index]

        # Apply the selected swap
        new_route = two_opt_swap(current_route, i, k)
        new_distance = calculate_distance(new_route, distance_matrix)

        # Metropolis criterion (always accept improvements)
        if new_distance < current_distance:
            current_route = new_route
            current_distance = new_distance
            if new_distance < best_distance:
                best_distance = new_distance
                best_route = current_route.copy()
        else: #Accepting worse solutions with a certain probability
            delta = new_distance - current_distance
            if random.random() < np.exp(-delta / temperature):
                current_route = new_route
                current_distance = new_distance

        # Cool down
        temperature = max(min_temperature, temperature * cooling_rate)

    return best_route

def calculate_distance(route, distance_matrix):
    """Calculates the total distance of a route."""
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i], route[i + 1]]
    total_distance += distance_matrix[route[-1], route[0]]  # Return to the starting point
    return total_distance

def entropic_sampling(distances, temperature=1.0):
    """Samples a pair of indices (i, k) based on their distances using softmax (entropic sampling)."""
    probabilities = softmax(-np.array(distances) / temperature)
    cumulative_probabilities = np.cumsum(probabilities)
    rand_val = random.random()
    for i in range(len(cumulative_probabilities)):
        if rand_val <= cumulative_probabilities[i]:
            return i  # Return the index chosen
    return len(cumulative_probabilities) - 1  # Should not happen, but safeguard.

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
                    cities.append(City(x, y))  # City name not used, so simplified
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
            # Create distance matrix for 2-opt
            distance_matrix = cdist([[city.x, city.y] for city in cities], [[city.x, city.y] for city in cities])

            results[filename] = {}

            # Find best NN path for 2-opt initialization
            best_nn_path = None
            best_nn_length = float('inf')
            for start_city in cities:
                reset_cities(cities)
                nn_path = nearest_neighbor(cities.copy(), start_city)
                nn_length = path_length(nn_path)
                if nn_length < best_nn_length:
                    best_nn_length = nn_length
                    best_nn_path = nn_path

            start_time = time.time()
            reset_cities(cities)
            # Convert best NN path to indices for 2-opt
            initial_route_indices = [cities.index(city) for city in best_nn_path]
            smoothed_route = two_opt_entropic_smoothing(distance_matrix, initial_route_indices)
            # Convert indices back to City objects
            smoothed_cities = [cities[i] for i in smoothed_route]
            results[filename]["two_opt_entropic"] = {"length": path_length(smoothed_cities), "time": time.time() - start_time}

            start_time = time.time()
            reset_cities(cities)
            path = nearest_neighbor(cities.copy())
            results[filename]["nearest_neighbor"] = {"length": path_length(path), "time": time.time() - start_time}

            start_time = time.time()
            reset_cities(cities)
            path = two_opt(cities.copy())
            results[filename]["two_opt"] = {"length": path_length(path), "time": time.time() - start_time}

            # for algorithm in ["entropic", "two_opt_entropic", "nearest_neighbor", "two_opt"]:
            for algorithm in ["two_opt_entropic", "nearest_neighbor", "two_opt"]:
                print(f"    {algorithm}: Length = {results[filename][algorithm]['length']}, Time = {results[filename][algorithm]['time']:.2f}s")
    return results

results = test_all_tsp_files()

for filename, algorithms in results.items():
    print(f"\nSummary for {filename}:")
    for algorithm, data in algorithms.items():
        print(f"    {algorithm}: Length = {data['length']}, Time = {data['time']:.2f}s")
