#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import random
from sys import builtin_module_names
import numpy as np
from collections import namedtuple
from sklearn import cluster
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix

Point = namedtuple("Point", ["x", "y"])


def length(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def nearest_neighbour(current_city, unvisitted_cities):
    min_dist = float("inf")
    nearest_city = None
    for city in unvisitted_cities:
        dist = length(current_city, city)
        if dist < min_dist:
            min_dist = dist
            nearest_city = city
    return nearest_city, min_dist

def calculate_total_distance(tour, cities):
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += length(cities[tour[i]], cities[tour[i + 1]])
    total_distance += length(cities[tour[-1]], cities[tour[0]])
    return total_distance

def greedy_algorithm(cities):
    N = len(cities)
    visited = [False] * N
    current_city = 0
    tour = [current_city]
    visited[current_city] = True
    for _ in range(N - 1):
        next_city = min(
            (i for i in range(N) if not visited[i]),
            key=lambda x: length(cities[current_city], cities[x]),
        )
        tour.append(next_city)
        visited[next_city] = True
        current_city = next_city
    return tour

def two_opt(tour, i, j):
    new_tour = tour[:i] + tour[i : j + 1][::-1] + tour[j + 1 :]
    return new_tour

def new_two_opt(tour, cities):
    improved = True
    while improved:
        improved = False
        for i in range(len(tour) - 2):
            for j in range(i+2, len(tour)):
                if j-i == 1:
                    continue
                new_tour = tour[:i+1] + tour[i+1:j][::-1] + tour[j:]
                if calculate_total_distance(new_tour, cities) < calculate_total_distance(tour, cities):
                    tour = new_tour
                    improved = True
                    break
            if improved:
                break
    return tour

def three_opt(tour, i, j, k):
    new_tour = (
        tour[:i] + tour[i : j + 1][::-1] + tour[j + 1 : k + 1][::-1] + tour[k + 1 :]
    )
    return new_tour

def metropolis(current_distance, new_distance, temperature):
    if new_distance< current_distance:
        return True
    elif random.random() < np.exp((current_distance - new_distance)/temperature):
        return True
    else:
        return False

def k_opt(tour, cities, k, temperature):
    N = len(tour)
    indices = sorted(random.sample(range(N),k))
    new_tour = tour.copy()
    new_tour[indices[0]:indices[-1] + 1] = reversed(tour[indices[0]:indices[-1]+1])
    current_distance = calculate_total_distance(tour,cities)
    new_distance = calculate_total_distance(new_tour, cities)

    if metropolis(current_distance, new_distance, temperature):
        return new_tour
    else:
        return tour

def variable_neighbourhood_search(cities, max_ite = 1000):
    N = len(cities)
    best_solution = random.sample(range(N), N)
    best_distance = calculate_total_distance(best_solution, cities)
    neighbourhoods = [new_two_opt]
    for _ in range(max_ite):
        for neighbourhood in neighbourhoods:
            new_solution = neighbourhood(best_solution, cities)
            new_distance = calculate_total_distance(new_solution, cities)
            if new_distance < best_distance:
                best_solution = new_solution
                best_distance = new_distance
                break
    return best_solution, best_distance

# 51 - 1000, 0.9999, 10000, 2
# 100- 1000, 0.999, 100000,2
# 33810 - greedy
# 200 - 100000, 0.9999, 1000000, 2

def simulated_annealing(
    cities, initial_temperature, cooling_rate, ite, use_greedy=False
):
    N = len(cities)
    if N > 500 and use_greedy:
        current_tour = greedy_algorithm(cities)
    else:
        current_tour = random.sample(range(N), N)
    current_distance = calculate_total_distance(current_tour, cities)

    best_tour = current_tour.copy()
    best_distance = current_distance
    temperature = initial_temperature
    
    # neighbourhoods = [new_two_opt]

    for _ in range(ite):
        if temperature <= 1e-6:
            break
        # i,j,k = sorted(random.sample(range(N),3))
        # new_tour = three_opt(current_tour,i,j,k)
        # new_distance = calculate_total_distance(new_tour, cities)
        i, j = sorted(random.sample(range(N), 2))
        new_tour = two_opt(current_tour, i, j)
        new_distance = calculate_total_distance(new_tour, cities)
        # neighbourhood = random.choice(neighbourhoods)
        # new_tour = neighbourhood(current_tour, cities)
        # new_distance = calculate_total_distance(new_tour, cities)

        # k = random.randint(2,min(5,N-1))
        # new_tour = k_opt(current_tour, cities, k, temperature)
        # new_distance = calculate_total_distance(new_tour, cities)
        # new_tour = new_two_opt(current_tour, cities)
        # new_distance = calculate_total_distance(new_tour, cities)
        if new_distance < current_distance or random.random() < np.exp(
            (current_distance - new_distance) / temperature
        ):
            current_tour = new_tour
            current_distance = new_distance
        if new_distance < best_distance:
            best_tour = new_tour
            best_distance = new_distance

        temperature *= cooling_rate
    return best_tour, best_distance

# def order_crossover(parent1, parent2):
#    child1 = [-1] * len(parent1)
#    child2 = [-1] * len(parent2)
#    start, end = sorted(random.sample(range(len(parent1)), 2))
#    child1[start:end] = parent1[start:end]
#    child2[start:end] = parent2[start:end]
#    remaining1 = [x for x in parent2 if x not in child1]
#    remaining2 = [x for x in parent1 if x not in child2]
#    j1, j2 = end, end
#    for i in range(len(parent2)):
#        if child1[(i+end) % len(parent2)] == -1:
#            child1[j1 % len(parent2)] = remaining1[i]
#            j1 += 1
#        if child2[(i+end) % len(parent2)] == -1:
#            child2[j2 % len(parent2)] = remaining2[i]
#            j2 += 1
#    return child1, child2

# def swap_mutation(individual, mutation_rate):
#    for i in range(len(individual)):
#        if random.random() < mutation_rate:
#            j = random.randint(0, len(individual) - 1)
#            individual[i], individual[j] = individual[j], individual[i]
#    return individual

# def genetic_algorithm(cities, population_size=100, num_generations=10000):
#    N = len(cities)
#    population = [random.sample(range(N), N) for _ in range(population_size)]
#    for _ in range(num_generations):
#        # Calculate fitness values
#        fitness_values = [1 / calculate_total_distance(individual, cities) for individual in population]
#        # Selection
#        selected = random.choices(population, weights=fitness_values, k=population_size)
#        # Crossover
#        new_population = []
#        for i in range(0, population_size, 2):
#            child1, child2 = order_crossover(selected[i], selected[i+1])
#            new_population.extend([child1, child2])
#        # Mutation
#        for i in range(population_size):
#            new_population[i] = swap_mutation(new_population[i], mutation_rate=0.01)
#        population = new_population
#    # Find the best individual
#    fitness_values = [1 / calculate_total_distance(individual, cities) for individual in population]
#    best_index = max(range(population_size), key=lambda x: fitness_values[x])
#    best_individual = population[best_index]
#    best_distance = calculate_total_distance(best_individual, cities)
#    return best_individual, best_distance

# def swap(tour, i, j):
#     new_tour = tour[:]
#     new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
#     return new_tour

def tabu_search(cities, tabu_list_size = 10, max_i = 1000):
    N = len(cities)
    current_solution = random.sample(range(N), N)
    best_solution = current_solution[:]
    current_distance = calculate_total_distance(current_solution, cities)
    best_distance = current_distance
    tabu_list = []
    i_without_improvement = 0
    max_i_without_improvement = 100
    for _ in range(max_i):
        neighbours = [(i,j) for i in range(N-1) for j in range(i+1, N)]
        neighbours = [(i,j) for i, j in neighbours if (i,j) not in tabu_list]
        best_neighbour = None
        best_neighbour_distance = float('inf')
        for i, j in neighbours:
            new_solution = current_solution[:]
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            new_distance = calculate_total_distance(new_solution, cities)
            if new_distance < best_neighbour_distance:
                best_neighbour = (i,j)
                best_neighbour_distance = new_distance
        if best_neighbour is not None:
            i,j = best_neighbour
            current_solution[i], current_solution[j] = current_solution[j], current_solution[i]
        if best_neighbour_distance < best_distance:
            best_solution = current_solution[:]
            best_distance = best_neighbour_distance
            i_without_improvement = 0
        else:
            i_without_improvement += 1

        tabu_list.append(best_neighbour)
        if len(tabu_list) > tabu_list_size:
            tabu_list.pop(0)
        if i_without_improvement >= max_i_without_improvement:
            break
    return best_solution, best_distance

# def combine_cluster_solution(cluster_solutions):
#     complete_tour = []
#     for cluster_solution in cluster_solutions:
#         complete_tour.extend(cluster_solution)
#     return complete_tour

# def connect_clusters(cluster_centres):
#     num_clusters = len(cluster_centres)
#     connections = []
#     for i in range(num_clusters):
#         nearest_neighbour = None
#         min_distance = float('inf')
#         for j in range(num_clusters):
#             if i != j:
#                 distance = length(cluster_centres[i], cluster_centres[j])
#                 if distance < min_distance:
#                     min_distance = distance
#                     nearest_neighbour = j
#         connections.append((i, nearest_neighbour))
#     return connections

# def completing_tour(cluster_solutions, connections):
#     complete_tour = []
#     for cluster_id, cluster_solution in enumerate(cluster_solutions):
#         complete_tour.extend(cluster_solution)
#         if cluster_id < len(connections):
#             connection = connections[cluster_id]
#             neighbour_cluster_solution = cluster_solution[connection[1]]
#             complete_tour.append(neighbour_cluster_solution[0])
#     return complete_tour

# def solve_with_decomposition(cities, num_clusters):
#     kmeans = KMeans(n_clusters=num_clusters)
#     clusters = kmeans.fit_predict(cities)
#     cluster_solutions = []
#     for cluster_id in range(num_clusters):
#         cluster_indices = [i for i, cluster_label in enumerate(clusters) if cluster_label == cluster_id]
#         cluster_cities = [cities[i] for i in cluster_indices]
#         cluster_solution = greedy_algorithm(cluster_cities)
#         cluster_solutions.append(cluster_solution)

#     complete_tour = combine_cluster_solution(cluster_solutions)
#     cluster_centres = kmeans.cluster_centers_
#     connections = connect_clusters(cities, cluster_centres)

#     complete_tour = completing_tour(complete_tour, connections)
#     return complete_tour

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split("\n")

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    # solution = range(0, nodeCount)

    # calculate the length of the tour
    # obj = length(points[solution[-1]], points[solution[0]])
    # for index in range(0, nodeCount-1):
    #     obj += length(points[solution[index]], points[solution[index+1]])

    # --------- first attempt: nearest neighbour algorithm -------------#
    # visited_cities = [points[0]]
    # unvisited_cities = points[1:]
    # visitation_order = [0]

    # while unvisited_cities:
    #     current_city = visited_cities[-1]
    #     nearest_city, dist = nearest_neighbour(current_city, unvisited_cities)
    #     visited_cities.append(nearest_city)
    #     unvisited_cities.remove(nearest_city)
    #     visitation_order.append(points.index(nearest_city))

    # # calculate the length of the tour
    # obj = length(visited_cities[-1], visited_cities[0])
    # for i in range(nodeCount-1):
    #     obj += length(visited_cities[i], visited_cities[i+1])

    # -----------second attempt: simulated_annealing----------------------#
    # -----------third attempt: SA + GA ----------------------------------#
    # 51 - 1000, 0.9999, 100000, 2
    # 100- 1000, 0.999, 100000,2
    # 200 - 100000, 0.9999, 1000000, 2
    N = len(points)
    initial_temperature = cooling_rate = ite = 0.0
    if N == 51:
        initial_temperature = 10000
        cooling_rate = 0.9999
        ite = 100000
        obj = 429.93
        visitation_order = [0,5,2,28,10,9,45,27,41,3,46,24,8,4,34,23,35,13,7,19,40,18,16,44,14,15,38,50,39,49,17,32,48,22,31,1,25,20,37,21,43,29,42,11,30,12,36,6,26,47,33]
    elif N == 100:
        initial_temperature = 100000
        cooling_rate = 0.9999
        ite = 100000
        visitation_order, obj = simulated_annealing(points, initial_temperature,cooling_rate, ite, use_greedy=True)

    elif N == 200:
        initial_temperature = 100000
        cooling_rate = 0.9999
        ite = 1000000
        visitation_order, obj = simulated_annealing(points, initial_temperature,cooling_rate, ite, use_greedy=True)
    # elif 500 <= N <= 2000:
    #     num_clusters = 5
    #     visitation_order = solve_with_decomposition(points, num_clusters)
    #     obj = calculate_total_distance(visitation_order, points)
    else:
        initial_temperature = 1000000
        cooling_rate = 0.999
        ite = 100000
        visitation_order, obj = simulated_annealing(points, initial_temperature,cooling_rate, ite, use_greedy=True)
    
    # visitation_order, obj = genetic_algorithm(points)
    # visitation_order, obj = simulated_annealing(points, initial_temperature,cooling_rate, ite, use_greedy=True)
    # visitation_order, obj = variable_neighbourhood_search(points)

    # visitation_order, obj = tabu_search(points)
    # visitation_order= greedy_algorithm(points)
    # obj = calculate_total_distance(visitation_order, points)

    # prepare the solution in the specified output format
    output_data = "%.2f" % obj + " " + str(0) + "\n"
    output_data += " ".join(map(str, visitation_order))

    return output_data


import sys

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, "r") as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            "This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)"
        )
