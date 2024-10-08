# Guide: https://developers.google.com/optimization/routing/vrp

import requests
import json
import urllib
import urllib.request


def create_time_matrix(data, traffic = False, departure_time = 'now'):
  addresses = data["addresses"]
  API_key = data["API_key"]
  # Distance Matrix API only accepts 100 elements per request, so get rows in multiple requests.
  max_elements = 100
  num_addresses = len(addresses)
  # Maximum number of rows that can be computed per request.
  max_rows = max_elements // num_addresses
  # num_addresses = q * max_rows + r.
  q, r = divmod(num_addresses, max_rows)
  dest_addresses = addresses
  time_matrix = []
  # Send q requests, returning max_rows rows per request.
  for i in range(q):
    origin_addresses = addresses[i * max_rows: (i + 1) * max_rows]
    response = send_request(origin_addresses, dest_addresses, traffic, departure_time, API_key)
    # print(response)
    time_matrix += build_time_matrix(response, traffic)

  # Get the remaining remaining r rows, if necessary.
  if r > 0:
    origin_addresses = addresses[q * max_rows: q * max_rows + r]
    response = send_request(origin_addresses, dest_addresses, traffic, departure_time, API_key)
    # print(response)
    time_matrix += build_time_matrix(response, traffic)

  return time_matrix


def create_distance_matrix(data):
  addresses = data["addresses"]
  API_key = data["API_key"]
  # Distance Matrix API only accepts 100 elements per request, so get rows in multiple requests.
  max_elements = 100
  num_addresses = len(addresses)
  # Maximum number of rows that can be computed per request (6 in this example).
  max_rows = max_elements // num_addresses
  # num_addresses = q * max_rows + r (q = 2 and r = 4 in this example).
  q, r = divmod(num_addresses, max_rows)
  dest_addresses = addresses
  distance_matrix = []
  # Send q requests, returning max_rows rows per request.
  for i in range(q):
    origin_addresses = addresses[i * max_rows: (i + 1) * max_rows]
    response = send_request(origin_addresses, dest_addresses, API_key)
    distance_matrix += build_distance_matrix(response)

  # Get the remaining remaining r rows, if necessary.
  if r > 0:
    origin_addresses = addresses[q * max_rows: q * max_rows + r]
    response = send_request(origin_addresses, dest_addresses, API_key)
    distance_matrix += build_distance_matrix(response)
  return distance_matrix


def send_request(origin_addresses, dest_addresses, traffic, departure_time, API_key):
  """ Build and send request for the given origin and destination addresses."""
  def build_address_str(addresses):
    # Build a pipe-separated string of addresses
    address_str = ''
    for i in range(len(addresses) - 1):
      address_str += addresses[i] + '|'
    address_str += addresses[-1]
    return address_str

  request = 'https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial'
  origin_address_str = build_address_str(origin_addresses)
  dest_address_str = build_address_str(dest_addresses)
  request = request + '&origins=' + origin_address_str + '&destinations=' + dest_address_str + '&key=' + API_key
  if traffic:
    request = request + '&departure_time=' + departure_time
  # print('Request:\n'+str(request)+'\n')
  jsonResult = urllib.request.urlopen(request).read()
  response = json.loads(jsonResult)
  # print('Response:\n'+str(response)+'\n')
  return response


def build_time_matrix(response, traffic):
  time_matrix = []
  for row in response['rows']:
    if traffic:
      row_list = [row['elements'][j]['duration_in_traffic']['value'] for j in range(len(row['elements']))]
    else:
      row_list = [row['elements'][j]['duration']['value'] for j in range(len(row['elements']))]
    time_matrix.append(row_list)
  return time_matrix


def build_distance_matrix(response):
  distance_matrix = []
  for row in response['rows']:
    row_list = [row['elements'][j]['distance']['value'] for j in range(len(row['elements']))]
    distance_matrix.append(row_list)
  return distance_matrix

def main():
    """Test code here"""
    
if __name__ == "__main__":
    main()