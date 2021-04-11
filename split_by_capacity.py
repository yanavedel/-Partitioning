"""Algorithms for partitioning data.

One-dimensional case. Partitioning by one feature.
"""
from __future__ import division

import collections
import itertools
import random

import pulp


def split_by_capacity(vehicles, k):
    """Split vehicles by capacities.

    Splitting set of vehicles into `k` subsets with almost equal (equal if
    possible) sums of capacities.

    Formulate and solve integer linear programming problem (ILP).
    diff --> min
    subject to
    ___________abs(sum_i(alpha[i][k_1] * c[i]) - sum_i(alpha[i][k_2] * c[i])) >= diff
                                                for all (k_1, k_2) from Combinations(k,2)
    ___________sum_j(alpha[i][j]) = nmbrs[i] for all i from 0 to len(vectors)
    ___________0 <= alpha[i][j] <= nmbrs[i] for all i from 0 to len(vectors)

    Args:
        vehicles (list[dict]): List of vehicles. Each vehicle is a dict with
            must have  "weight_capacity" and "initial_cost" keys.
        k (int): number of clusters to split in

    Returns:
        list[list[dict]]: List of `k` lists of vehicles.
    """

    # Preprocess dataset
    grouped = collections.defaultdict(list)
    for vehicle in vehicles:
        grouped[(vehicle["weight_capacity"], vehicle["initial_cost"])].append(vehicle)

    feature_groups, vehicles_by_features = zip(*grouped.items())
    vehicles_by_features = list(vehicles_by_features)
    capacities, _ = zip(*feature_groups)

    # Definition of problem and variables
    problem = pulp.LpProblem("Partitioning Problem", pulp.LpMinimize)
    delta = pulp.LpVariable("Delta", 0, None, pulp.LpInteger)
    problem += delta

    # Definition of constraints
    amounts_per_feature = []
    for i, vehicles_with_feature in enumerate(vehicles_by_features):
        amount = len(vehicles_with_feature)
        # Variable name is important here, because order matters and PuLP sorts
        # variables by name. We don't expect to have more than 999 clusters here.
        amount_per_feature = [
            pulp.LpVariable("Var{:03d}{}".format(cluster_index, i), 0, amount, pulp.LpInteger)
            for cluster_index in range(k)
        ]
        amounts_per_feature.append(amount_per_feature)
        problem += pulp.lpSum(amount_per_feature) == amount

    for k1, k2 in itertools.combinations(range(k), 2):
        constraint = sum(
            capacity * (amount_per_feature[k1] - amount_per_feature[k2])
            for capacity, amount_per_feature in zip(capacities, amounts_per_feature)
        )
        problem += constraint - delta <= 0

        constraint = sum(
            capacity * (amount_per_feature[k2] - amount_per_feature[k1])
            for capacity, amount_per_feature in zip(capacities, amounts_per_feature)
        )
        problem += constraint - delta <= 0

    # Problem solution
    problem.solve()
    solutions = [v.varValue for v in problem.variables()[1:]]

    # Representation of solution
    partitioning = collections.defaultdict(list)

    for i, solution in enumerate(solutions):
        solution = int(solution)
        cluster_index, feature_index = divmod(i, len(feature_groups))
        vehicles = vehicles_by_features[feature_index][:solution]
        vehicles_by_features[feature_index] = vehicles_by_features[feature_index][solution:]
        partitioning[cluster_index].extend(vehicles)

    return list(partitioning.values())


def vehicles_splitting(vehicles, stop_clusters, routes, k):
    """Split vehicles by capacities.

    Splitting set of vehicles into `k` subsets with almost equal (equal if
    possible) sums of capacities. Contains both cases when we have only one
    type of vehicles and when we have different types. If we have one type
    of vehicles, we split them according to distances from depo per cluster.
    If we have different types of vehicles, we use previous split_by_capacities()
    approach.

    Args:
        vehicles (list[dict]): List of vehicles. Each vehicle is a dict with
            must have  "weight_capacity" and "initial_cost" keys.
        stop_clusters(dict[dict]): Clusters of stops. Must have ids and field
            "stops" in each "cluster"
        routes(list[dict]): List of routes
        k (int): number of clusters to split in

    Returns:
        list[list[dict]]: List of `k` lists of vehicles.
    """
    # Preprocess dataset
    grouped = collections.defaultdict(list)
    for vehicle in vehicles:
        grouped[(vehicle["weight_capacity"], vehicle["initial_cost"])].append(vehicle)

    feature_groups = list(grouped.keys())

    if len(feature_groups) > 1:
        return split_by_capacity(vehicles, k)

    depot = routes[0]["stops"][0]["point"]
    centroids = [
        stop_cluster["cluster"]["centroid"][0] for stop_cluster in stop_clusters.values()
    ]
    distances = [
        ((depot[0] - centroid[0]) ** 2 + (depot[0] - centroid[0]) ** 2) ** 0.5
        for centroid in centroids
    ]

    amount_per_cluster = [int((len(vehicles) * dist) / sum(distances)) for dist in distances]
    if sum(amount_per_cluster) < len(vehicles):
        diff = len(vehicles) - sum(amount_per_cluster)
        amount_per_cluster[amount_per_cluster.index(min(amount_per_cluster))] += diff

    partitioning = []

    for amount in amount_per_cluster:
        partitioning.append([vehicles[0] for _ in range(amount)])

    return partitioning


def add_vehicle(vehicle, planning_state):
    """Add new vehicle to clusters.

    Adding new vehicle to one of existing clusters. Changes the initial dict
    of clusters by adding a new vehicle.

    Args:
        vehicle (dict): Vehicle to add. Is a dict with
            must have  "weight_capacity" and "initial_cost" keys.
        planning_state (dict): Dict of `k` planning_states with uuids.
    """
    clusters_without_vehicle = []
    amounts_of_vehicle_per_cluster = []

    for cluster in planning_state.values():
        occurrences = sum(
            cl_vehicle["weight_capacity"] == vehicle["weight_capacity"]
            and cl_vehicle["initial_cost"] == vehicle["initial_cost"]
            for cl_vehicle in cluster.vehicles
        )

        if not occurrences:
            clusters_without_vehicle.append(cluster)
        else:
            amounts_of_vehicle_per_cluster.append((cluster, occurrences))

    if clusters_without_vehicle:
        cluster = random.choice(clusters_without_vehicle)
    else:
        min_amount = min(amount for _, amount in amounts_of_vehicle_per_cluster)
        cluster = random.choice(
            [cluster for cluster, amount in amounts_of_vehicle_per_cluster if amount == min_amount]
        )

    cluster.vehicles.add(vehicle)


def split_uniformly(vehicles, num_of_clusters):
    """Splits vehicles uniformly into num_of_clusters clusters.

    Args:
        vehicles (list[dict]): Iterable of dicts that correspond to vehicles.
        num_of_clusters(int): Number of clusters to split the routes into.

    Returns:
        list[list[dict]]: Vehicles split in clusters.

    v2.1
    """
    clustered_vehicles = []
    for i in range(num_of_clusters):
        clustered_vehicles.append(vehicles[i::num_of_clusters])
    return clustered_vehicles


# Uncomment the following for run

if __name__ == "__main__":

    from kwargs_to_play_with import kwargs
    from data_demo80 import stops

    st = stops
    rs = kwargs['routes']
    vs = kwargs['vehicles']

    clustered_vehicles = split_by_capacity(vs, 4)
