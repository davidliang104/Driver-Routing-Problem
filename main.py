from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import re
import numpy as np
import pandas as pd
import urllib
import matrix
import api
from tabulate import tabulate
from openpyxl import Workbook

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


def create_data_model(drivers_df, workers_df, vehicle_capacity):
    '''Stores the data for the problem.'''
    data = {}

    data['addresses'] = []
    address_dict = {} # maps index in data['addresses'] to full name of address
    address_rpt = [] # addresses with home and destinations repeated for the solver
    data['num_vehicles'] = drivers_df.shape[0]
    data['vehicle_capacities'] = [vehicle_capacity] * data['num_vehicles']
    data['depot'] = 0
    data['driver_names'] = drivers_df['Name'].tolist()
    data['worker_name'] = ['depot']
    data['node_type'] = ['depot']
    data['time_windows'] = [(0, 86400)]
    data['demands'] = [0]
    data['pickups_deliveries'] = []
    # data['time_matrix'] = []
    index = 1
    for _, row in workers_df.iterrows():
        # Home
        data['worker_name'].append(row['Name'])
        data['node_type'].append('from_1')
        home_address = format_address(row['Home'])
        data['demands'].append(1)
        data['time_windows'].append((0, 86400))
        if home_address not in address_dict:
            data['addresses'].append(home_address)
            address_dict[home_address] = len(data['addresses']) - 1
        address_rpt.append(address_dict[home_address])

        # Work
        data['worker_name'].append(row['Name'])
        data['node_type'].append('to_1')
        dest_address = format_address(row['Destination'])
        dest_time = row['Arrival Time']
        dest_time_sec = dest_time.hour * 3600 + dest_time.minute * 60 # + dest_time.second
        data['time_windows'].append((dest_time_sec-300, dest_time_sec))
        data['demands'].append(-1)
        if dest_address not in address_dict:
            data['addresses'].append(dest_address)
            address_dict[dest_address] = len(data['addresses']) - 1
        address_rpt.append(address_dict[dest_address])

        data["pickups_deliveries"].append([index, index + 1])

        """Repeat, but in reverse for the return trip"""
        # Work
        data['worker_name'].append(row['Name'])
        data['node_type'].append('from_2')
        data['demands'].append(1)
        leave_time = row['Leave Time']
        leave_time_sec = leave_time.hour * 3600 + leave_time.minute * 60 # + leave_time.second
        data['time_windows'].append((leave_time_sec, leave_time_sec + 300))
        address_rpt.append(address_dict[dest_address])

        # Home
        data['worker_name'].append(row['Name'])
        data['node_type'].append('to_2')
        dest_time = row['Arrival Time']
        data['time_windows'].append((0, 86400))
        data['demands'].append(-1)
        address_rpt.append(address_dict[home_address])

        data["pickups_deliveries"].append([index + 2, index + 3])
        index += 4 

    data['address_rpt'] = [-1] + address_rpt
    data['address_dict'] = address_dict

    # Build time matrix from unique addresses
    data['API_key'] = api.get_api_key()
    time_matrix = matrix.create_time_matrix(data, traffic = True)
    # time_matrix = [[   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #        0,    0],
    #    [   0,    0, 1633, 1633,    0, 1552, 1633, 1633, 1552, 1552, 1633,
    #     1633, 1552],
    #    [   0, 1723,   15,   15, 1723,  909,   15,   15,  909,  300,   15,
    #       15,  300],
    #    [   0, 1723,   15,   15, 1723,  909,   15,   15,  909,  300,   15,
    #       15,  300],
    #    [   0,    0, 1633, 1633,    0, 1552, 1633, 1633, 1552, 1552, 1633,
    #     1633, 1552],
    #    [   0, 1773, 1026, 1026, 1773,   10, 1026, 1026,   10, 1152, 1026,
    #     1026, 1152],
    #    [   0, 1723,   15,   15, 1723,  909,   15,   15,  909,  300,   15,
    #       15,  300],
    #    [   0, 1723,   15,   15, 1723,  909,   15,   15,  909,  300,   15,
    #       15,  300],
    #    [   0, 1773, 1026, 1026, 1773,   10, 1026, 1026,   10, 1152, 1026,
    #     1026, 1152],
    #    [   0, 1685,  184,  184, 1685, 1065,  184,  184, 1065,    0,  184,
    #      184,    0],
    #    [   0, 1723,   15,   15, 1723,  909,   15,   15,  909,  300,   15,
    #       15,  300],
    #    [   0, 1723,   15,   15, 1723,  909,   15,   15,  909,  300,   15,
    #       15,  300],
    #    [   0, 1685,  184,  184, 1685, 1065,  184,  184, 1065,    0,  184,
    #      184,    0]]

    array = np.zeros((len(address_rpt) + 1, len(address_rpt) + 1), dtype = int)

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if i!=0 and j!=0:
                array[i, j] = time_matrix[address_rpt[i-1]][address_rpt[j-1]]

    data['time_matrix'] = array
    
    print(address_dict)
    print(address_rpt)

    return data


def format_address(address):
    # Remove surrounding spaces
    address = address.strip()

    # Check if the address is in the format "double, double" or "double,double"
    pattern = r"^\d+\.\d+,\s?\d+\.\d+$"
    
    if re.match(pattern, address):
        # Remove the space if it exists
        address = re.sub(r"\s+", "", address)
    else:
        # Add ", Kuwait" to the end
        address = f"{address}, Kuwait"

    # Percent encode to use in URL
    return urllib.parse.quote(address)


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
    # Determine AM or PM
    # period = 'AM' if hours < 12 else 'PM'
    # Adjust hours to 12-hour format
    # if hours == 0:
    #     hours = 12
    # elif hours > 12:
    #     hours -= 12

    return f"{hours:02}:{minutes:02}:{seconds:02}"


def convert_seconds_to_hhmm(seconds):
    # Calculate hours and minutes
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    # Determine AM or PM
    period = 'AM' if hours < 12 else 'PM'
    # Adjust hours to 12-hour format
    if hours == 0:
        hours = 12
    elif hours > 12:
        hours -= 12

    # Format the time string
    time_str = f"{hours:02}:{minutes:02} {period}"

    return time_str


def print_2d_matrix(matrix):
    # Convert each element to HH:MM:SS format
    formatted_matrix = [[convert_seconds_to_hhmmss(cell) for cell in row] for row in matrix]
    
    # Print the formatted matrix using tabulate
    print(tabulate(formatted_matrix, tablefmt="grid"))
    print()


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
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

    # Add Time constraint.
    time = "Time"
    routing.AddDimension(
        transit_callback_index,
        100000,  # allow waiting time
        10000000000,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time,
    )
    time_dimension = routing.GetDimensionOrDie(time)
    # # set a large coefficient for the global span of the routes (i,e, maximum time of routes)
    # # to minimize time of longest route
    # # time_dimension.SetGlobalSpanCostCoefficient(100)

    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data["time_windows"]):
        if location_idx == data["depot"]:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
    # # Add time window constraints for each vehicle start and end node.
    # depot_idx = data["depot"]
    # for vehicle_id in range(data["num_vehicles"]):
    #     index = routing.Start(vehicle_id)
    #     time_dimension.CumulVar(index).SetRange(
    #         data["time_windows"][depot_idx][0], data["time_windows"][depot_idx][1]
    #     )

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
        create_schedule(data, manager, routing, solution)
    else:
        print("\nNo solution\n")

    return solution


def create_schedule(data, manager, routing, solution):
    """Store schedule into Excel sheet."""
    wb = Workbook()
    ws = wb.active
    row_count = 1
    time_dimension = routing.GetDimensionOrDie("Time")
    for vehicle_id in range(data["num_vehicles"]):
        ws.append([data['driver_names'][vehicle_id]])
        print([data['driver_names'][vehicle_id]])
        ws.merge_cells(f'A{row_count}:E{row_count}')
        row_count += 1
        ws.append(['Name', 'From', 'To',  'Pick-up Time', 'Arrival Time'])
        row_count += 1
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            time_var = time_dimension.CumulVar(index)
            ind = manager.IndexToNode(index)
            row = []
            if data['node_type'][ind] == 'from_1' or data['node_type'][ind] == 'from_2':
                name = data['worker_name'][ind]
                home_address = urllib.parse.unquote(data['addresses'][data['address_rpt'][ind]])#[:-len(', Kuwait')]
                dest_address = urllib.parse.unquote(data['addresses'][data['address_rpt'][ind + 1]])#[:-len(', Kuwait')]
                if data['node_type'][ind] == 'from_1':
                    # Pickup time range:
                    # pickup_time = f'{convert_seconds_to_hhmm(solution.Min(time_var))} to {convert_seconds_to_hhmm(solution.Max(time_var))}'
                    # Midpoint of pickup time range:
                    # pickup_time = convert_seconds_to_hhmm((solution.Min(time_var)+solution.Max(time_var)) // 2)
                    # Max pickup time:
                    pickup_time = convert_seconds_to_hhmm(solution.Max(time_var))
                    arrival_time = convert_seconds_to_hhmm(data['time_windows'][ind + 1][1])
                elif data['node_type'][ind] == 'from_2':
                    pickup_time = convert_seconds_to_hhmm((solution.Min(time_var)+solution.Max(time_var)) // 2)
                    arrival_time = '-'
                row = [ name,
                        home_address, 
                        dest_address, 
                        pickup_time, 
                        arrival_time]
                ws.append(row)
                print(row, '[', data['address_rpt'][ind], 'to', data['address_rpt'][ind + 1], ']')
                row_count += 1
        ws.append([])
        row_count += 1

    wb.save('Schedule.xlsx')

def main():
    """Entry point of the program."""

    # Read the Excel files
    drivers_df = pd.read_excel('Test_Drivers.xlsx')
    # drivers_df = pd.read_excel('Drivers.xlsx')
    workers_df = pd.read_excel('Test_Workers.xlsx')
    # workers_df = pd.read_excel('Workers (edited).xlsx')
    vehicle_capacity = 2
    data = create_data_model(drivers_df, workers_df, vehicle_capacity)
    print(data)
    print_2d_matrix(data['time_matrix'])
    solve(data)

if __name__ == "__main__":
    main()