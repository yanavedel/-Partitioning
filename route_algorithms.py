"""Algorithms for routes clustering."""
from __future__ import division

import collections
import random


def split_uniformly(stops, vehicles, routes, num_of_clusters):
    """Split routes uniformly into num_of_clusters clusters.

    Args:
        stops (dict[str, dict]): A dict with keys representing ids of clusters(child-buckets).
               Each of those keys has a value of this example: {
               "cluster": {
                   "centroid": [(4.447033316605503, 51.18522441958278)],
                   "id": "ebcd5d5f-8fb4-4f82-bc00-5264d7bac764",
                   "area": "",
               },
               "stops": [
                   {}....
               ]}
        vehicles (list[list[dict]]): List of lists of vehicles that represent the partitions of
               the vehicles.
        routes (list[dict]): Iterable of dicts that correspond to routes.
        num_of_clusters(int): Number of clusters to split the routes into.

    Returns:
        dict : A dict with keys as follows:
            "clusters": a list with ids of clusters(child-buckets)
            "routes": a list of hashable routes
            "vehicles": a list of hashable vehicles
            All of them are related to each other based on index in their lists.

    r1.1
    """
    clustered_routes = []
    for i in range(num_of_clusters):
        clustered_routes.append(routes[i::num_of_clusters])
    return {"clusters": list(stops.keys()), "routes": clustered_routes, "vehicles": vehicles}


def add_route(route, planning_state):
    """Add new route to clusters.

    Adding new route to one of existing clusters. Changes the initial dict
    of clusters by adding a new route.

    Args:
        route (dict): Route to add. Is a dict with
            must have  "windows" keys.
        planning_state (dict): Dict of `k` planning_states with uuids.
    """
    clusters_without_route = []
    amounts_of_route_per_cluster = []

    for cluster in planning_state.values():
        occurrences = sum(
            cluster_route["windows"] == route["windows"] for cluster_route in cluster.routes
        )
        if not occurrences:
            clusters_without_route.append(cluster)
        else:
            amounts_of_route_per_cluster.append((cluster, occurrences))

    if clusters_without_route:
        cluster = random.choice(clusters_without_route)
    else:
        min_amount = min(amount for _, amount in amounts_of_route_per_cluster)
        cluster = random.choice(
            [cluster for cluster, amount in amounts_of_route_per_cluster if amount == min_amount]
        )

    cluster.routes.add(route)


def partition_routes(stops, vehicles, routes):
    """Partition routes.

    According to the Euclidean distance of centroid of each cluster of stops to the depot,
    it maps vehicles and routes randomly and uniformly to each cluster such that the number
    of vehicles is less than the number of routes in each cluster. Moreover it takes into
    account the restriction that more vehicles belong to more distant from depot clusters.

    Args:
        stops (dict[str, dict]): A dict with keys representing ids of clusters(child-buckets).
               Each of those keys has a value of this example: {
               "cluster": {
                   "centroid": [(4.447033316605503, 51.18522441958278)],
                   "id": "ebcd5d5f-8fb4-4f82-bc00-5264d7bac764",
                   "area": "",
               },
               "stops": [
                   {}....
               ]}
        vehicles (list[list[dict]]): List of lists of vehicles that represent the partitions of
               the vehicles.
        routes (list[dict]]: iterable of Hashable dicts that represent routes.

    Returns:
        dict : A dict with keys as follows:
            "clusters": a list with ids of clusters(child-buckets)
            "routes": a list of hashable routes
            "vehicles": a list of hashable vehicles
            All of them are related to each other based on index in their lists.

    v3.1 & r2.1.
    """
    vehicles = sorted(vehicles, key=len, reverse=True)

    required_routes_per_cluster = [len(vehicles_of_cluster) for vehicles_of_cluster in vehicles]

    total_required = sum(required_routes_per_cluster)
    excess_routes = len(routes) - total_required

    num_routes_to_add = []
    for num in required_routes_per_cluster:
        additional_routes = int(round((num / total_required) * excess_routes))
        num_routes_to_add.append(num + additional_routes)

    routes_per_cluster = []
    clustered_so_far = 0
    for needed in num_routes_to_add:
        routes_per_cluster.append(routes[clustered_so_far: clustered_so_far + needed])
        clustered_so_far += needed

    remaining_routes = routes[clustered_so_far:]
    routes_per_cluster[-1].extend(remaining_routes)

    # Routes were clustered taking number of vehicles for cluster in
    # consideration. Now we want distribute these clustered routes & vehicles
    # in such way that clusters with longest distance from depot have the
    # biggest number of routes & vehicles, thus we return cluster ids sorted by
    # distance from depot to cluster centroid.
    depot_lat, depot_lon = routes[0]["stops"][0]["point"]

    def distance_from_depot(cluster_id):
        """Calculate squared distance from cluster centroid to depot.

        Squared distance is ok, because it is only used for sorting.
        """
        centroid_lon, centroid_lat = stops[cluster_id]["cluster"]["centroid"][0]
        return (depot_lon - centroid_lon) ** 2 + (depot_lat - centroid_lat) ** 2

    clusters_ids = sorted(stops.keys(), key=distance_from_depot, reverse=True)

    res = {"clusters": clusters_ids, "routes": routes_per_cluster, "vehicles": vehicles}

    return res


def smarter_routes_partitioning(stop_clusters, vehicle_clusters, routes):
    """Smart partitioning of routes.

    Including case when we have less routes than vehicles.

    Args:
        stop_clusters(dict[dict]): Clusters of stops. Must have ids and field
        "stops" in each "cluster"
        vehicle_clusters(list[dict]): Clusters of vehicles
        routes(list[dict]): Routes to clusterize

    Return:
        (dict): Dictionary with fields "clusters_ids", "routes" and "vehicles"
    """
    if len(routes) >= sum(len(vehicles) for vehicles in vehicle_clusters):
        return partition_routes(stop_clusters, vehicle_clusters, routes)

    depot = routes[0]["stops"][0]["point"]
    centroids = [stop_cluster["cluster"]["centroid"][0] for stop_cluster in stop_clusters.values()]
    distances = [
        ((depot[0] - centroid[0]) ** 2 + (depot[0] - centroid[0]) ** 2) ** 0.5
        for centroid in centroids
    ]

    # Preprocess vehicles dataset
    vehicle_type_per_cluster = []
    for cluster in vehicle_clusters:
        grouped = collections.defaultdict(list)
        for vehicle in cluster:
            grouped[(vehicle["weight_capacity"], vehicle["initial_cost"])].append(vehicle)
        vehicles_by_features = list(grouped.values())
        vehicle_type_per_cluster.append(vehicles_by_features)

    ratios = [
        len(routes)
        * len(cluster)
        / sum(len(vehicle_cluster) for vehicle_cluster in vehicle_clusters)
        for cluster in vehicle_type_per_cluster
    ]

    cut_vehicles_clusters = [[] for _ in vehicle_clusters]
    leftovers = [[] for _ in vehicle_clusters]
    for vehicles, ratio in zip(vehicle_type_per_cluster, ratios):
        for vehicle_type in vehicles:
            cut_vehicles_clusters.append(vehicle_type[: int(ratio * len(vehicle_type))])
            leftovers.append(vehicle_type[int(ratio * len(vehicle_type)):])

    leftovers = sorted(leftovers, key=len)
    sorted_distances = sorted(distances)
    leftovers = [leftovers[distances.index(dist)] for dist in sorted_distances]

    vehicle_clusters = [
        cut_cluster + leftover for cut_cluster, leftover in zip(cut_vehicles_clusters, leftovers)
    ]
    clusters_ids = list(stop_clusters.keys())
    routes_clusters = []
    for i in range(len(cut_vehicles_clusters)):
        start_index = len(cut_vehicles_clusters[i - 1])
        end_index = len(cut_vehicles_clusters[i])
        if i == 0:
            routes_clusters.append(routes[:len(cut_vehicles_clusters[i])])
        else:
            routes_clusters.append(routes_clusters[start_index: start_index + end_index])

    if sum(len(cluster) for cluster in routes_clusters) < len(routes):
        diff = len(routes) - sum(len(cluster) for cluster in routes_clusters)
        routes_clusters[-1] += routes[-diff:]

    return {"clusters": clusters_ids, "routes": routes_clusters, "vehicles": vehicle_clusters}


# Uncomment the following for run

#if __name__ == "__main__":
#
#    from kwargs_to_play_with import kwargs
#    from PNL_BE_Demons80_Stops import stops
#
#    st = stops
#    rs = kwargs['routes']
#    vs = kwargs['vehicles']
#
#    split_uniformly(st, vs, rs, 4)
