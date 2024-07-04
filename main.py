from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import pandas as pd
import urllib
import matrix
import api
from tabulate import tabulate

def create_test_data_model():
    """Stores the data for the problem."""
    data = {}
    data["distance_matrix"] = [
        [   0, 100, 1000, 100, 1000],
        [ 100,   0,   50,  20,   50],
        [1000,  50,    0,  50,    0],
        [ 100,  20,   50,   0,   50],
        [1000,  50,    0,  50,    0],
    ]
    data["time_matrix"] = [
        [0, 10, 30, 10, 30],
        [10, 0,  5,  2,  5],
        [30, 5,  0,  5,  0],
        [10, 2,  5,  0,  5],
        [30, 5,  0,  5,  0],
    ]
    data["pickups_deliveries"] = [
        [1, 2],
        [3, 4],
    ]
    data["time_windows"] = [
        (0, 1000),
        (0, 1000),
        (470, 480),
        (0, 1000),
        (470, 480),
    ]
    data["num_vehicles"] = 2
    data["vehicle_capacities"] = [2, 2]
    data["demands"] = [0, 1, 0, 1, 0] # no of people at each node
    data["depot"] = 0
    return data


def create_data_model(depot, df, num_vehicles, vehicle_capacity):
    """Stores the data for the problem."""
    data = {}

    data["num_vehicles"] = num_vehicles
    data["vehicle_capacities"] = [vehicle_capacity] * num_vehicles
    data["depot"] = 0
    data["addresses"] = [urllib.parse.quote(depot.strip())]
    data["time_windows"] = [(0, 86400)]
    data["demands"] = [0]
    data["pickups_deliveries"] = []

    for index, row in df.iterrows():
        # Home
        home_address = urllib.parse.quote(row["Home"].strip())
        data["addresses"].append(home_address)
        data["demands"].append(1)
        data["time_windows"].append((0, 86400))

        # Destination
        dest_address = urllib.parse.quote(row["Destination"].strip())
        data["addresses"].append(dest_address)
        dest_time = row["Drop off"]
        dest_time_sec = dest_time.hour * 3600 + dest_time.minute * 60 + dest_time.second
        data["time_windows"].append((dest_time_sec-300, dest_time_sec))
        data["demands"].append(0)

        data["pickups_deliveries"].append([index*2+1, index*2+2])

    # Build distance matrix
    # data["distance_matrix"] = matrix.build_distance_matrix(data)

    # Build time matrix
    data['API_key'] = api.get_api_key()
    data["time_matrix"] = matrix.create_time_matrix(data, traffic = True)
    # data["time_matrix"] = [[0, 0, 384, 790, 384], [0, 0, 384, 790, 384], [352, 352, 12, 794, 12], [1044, 1044, 908, 9, 908], [352, 352, 12, 794, 12]]
    # print(data['addresses'])
    return data


def time_to_seconds(time_str):
    # Extract the AM/PM part
    period = time_str[-2:].upper()
    # Extract the hour and minute part
    time_part = time_str[:-2].strip()
    
    # Split the time into hours and minutes
    hours, minutes = map(int, time_part.split(':'))
    
    # Convert to 24-hour format if needed
    if period == 'PM' and hours != 12:
        hours += 12
    elif period == 'AM' and hours == 12:
        hours = 0

    # Calculate total seconds
    total_seconds = hours * 3600 + minutes * 60
    
    return total_seconds


def convert_seconds_to_hhmmss(seconds):
    # Calculate hours and minutes
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def print_2d_matrix(matrix):
    # Convert each element to HH:MM:SS format
    formatted_matrix = [[convert_seconds_to_hhmmss(cell) for cell in row] for row in matrix]
    
    # Print the formatted matrix using tabulate
    print(tabulate(formatted_matrix, tablefmt="grid"))


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print('Distance:')
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

    print()

    print('Time:')
    print(f"Objective: {solution.ObjectiveValue()}")
    time_dimension = routing.GetDimensionOrDie("Time")
    total_time = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            plan_output += (
                f"{manager.IndexToNode(index)}"
                # f" Time({solution.Min(time_var)},{solution.Max(time_var)})"
                f" Time({convert_seconds_to_hhmmss(solution.Min(time_var))},{convert_seconds_to_hhmmss(solution.Max(time_var))})"
                " \n-> "
            )
            index = solution.Value(routing.NextVar(index))
        time_var = time_dimension.CumulVar(index)
        plan_output += (
            f"{manager.IndexToNode(index)}"
            f" Time({convert_seconds_to_hhmmss(solution.Min(time_var))},{convert_seconds_to_hhmmss(solution.Max(time_var))})\n"
        )
        plan_output += f"Time of the route: {convert_seconds_to_hhmmss(solution.Min(time_var))}\n"
        print(plan_output)
        total_time += solution.Min(time_var)
    print(f"Total time of all routes: {convert_seconds_to_hhmmss(total_time)}")


def solve(data):
    # Instantiate the data problem.
    #data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["time_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    # Define the cost of travel, to be the time taken to travel the arcs
    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["time_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Define cost of each arc to be the times taken to travel the arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Time Windows constraint.
    time = "Time"
    routing.AddDimension(
        transit_callback_index,
        5,  # allow waiting time
        100000,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time,
    )
    time_dimension = routing.GetDimensionOrDie(time)
    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data["time_windows"]):
        if location_idx == data["depot"]:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
    # Add time window constraints for each vehicle start node.
    depot_idx = data["depot"]
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(
            data["time_windows"][depot_idx][0], data["time_windows"][depot_idx][1]
        )

    # Instantiate route start and end times to produce feasible times.
    for i in range(data["num_vehicles"]):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i))
        )
        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(i)))

    # Add Capacity constraint.
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
            time_dimension.CumulVar(pickup_index)
            <= time_dimension.CumulVar(delivery_index)
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
    else:
        print("No solution")

def main():
    """Entry point of the program."""

    depot = "Boulevard Salmiya" # Driver home location
    # Read the Excel file
    df = pd.read_excel('Test_sheet.xlsx')
    num_vehicles = 2
    vehicle_capacity = 2
    data = create_data_model(depot, df, num_vehicles, vehicle_capacity)
    print(data)
    print_2d_matrix(data['time_matrix'])
    solve(data)

if __name__ == "__main__":
    main()