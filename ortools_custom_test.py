"""Simple Pickup Delivery Problem (PDP)."""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data["distance_matrix"] = [
        # fmt: off
        [0, 10, 10, 10],
        [10, 0, 10, 10],
        [10, 10, 0, 10],
        [10, 10, 10, 0],
        # fmt: on
    ]
    data["pickups_deliveries"] = [
        [1, 3],
        [2, 3],
    ]
    data["num_vehicles"] = 2
    data["vehicle_capacities"] = [2, 2]
    data["demands"] = [0, 1, 1, -1]
    data["depot"] = 0
    data["distance_matrix"], data["pickups_deliveries"], data["demands"] = duplicate_nodes(data["distance_matrix"], data["pickups_deliveries"], data["demands"])
    return data


def duplicate_nodes(distance_matrix, pickups_deliveries, demands):
    duplicates = set()
    duplicate_found = False
    dummies = []
    for i in range(len(pickups_deliveries)):
        pickup, delivery = pickups_deliveries[i][0], pickups_deliveries[i][1]
        if delivery not in duplicates:
            duplicates.add(delivery)
        else:
            duplicate_found = True
            # duplicate distances of location
            dummies.append(distance_matrix[delivery])
            # change index of delivery site to dummy's index
            pickups_deliveries[i][1] = len(distance_matrix) + len(dummies) - 1
            demands.append(demands[delivery])

    if duplicate_found:
        distance_array = np.array(distance_matrix)
        dummies_rows = np.array(dummies)
        dummies_cols = dummies_rows.T
        distance_array_merged = np.zeros((distance_array.shape[0]+dummies_rows.shape[0], distance_array.shape[1]+dummies_cols.shape[1]))
        distance_array_merged[:distance_array.shape[0], :distance_array.shape[1]] = distance_array
        distance_array_merged[distance_array.shape[0]:, :dummies_rows.shape[1]] = dummies_rows
        distance_array_merged[:dummies_cols.shape[0], distance_array.shape[1]:] = dummies_cols

        distance_matrix = np.round(distance_array_merged).astype(int).tolist() # round and convert to ints because doubles didn't work

    # for row in distance_matrix:
    #     print(row)

    # for row in pickups_deliveries:
    #     print(row)

    # for row in demands:
    #     print(row)

    return distance_matrix, pickups_deliveries, demands


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f"Objective: {solution.ObjectiveValue()}")
    total_distance = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += f" {manager.IndexToNode(index)} -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f"{manager.IndexToNode(index)}\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        print(plan_output)
        total_distance += route_distance
    print(f"Total Distance of all routes: {total_distance}m")


def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Define cost of each arc.
    def distance_callback(from_index, to_index):
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = "Distance"
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        5000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name,
    )
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0, # slack
        data["vehicle_capacities"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )
    capacity_dimension = routing.GetDimensionOrDie("Capacity")


    # Define Transportation Requests.
    for request in data["pickups_deliveries"]:
        pickup_index = manager.NodeToIndex(request[0])
        delivery_index = manager.NodeToIndex(request[1])
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        routing.solver().Add(
            routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index)
        )
        routing.solver().Add(
            distance_dimension.CumulVar(pickup_index)
            <= distance_dimension.CumulVar(delivery_index)
        )

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)


if __name__ == "__main__":
    main()