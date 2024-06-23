# Guide: https://developers.google.com/optimization/routing/vrp

import requests
import json
import urllib


def create_time_matrix(data):
  addresses = data["addresses"]
  API_key = data["API_key"]
  # Time Matrix API only accepts 100 elements per request, so get rows in multiple requests.
  max_elements = 100
  num_addresses = len(addresses)
  # Maximum number of rows that can be computed per request (6 in this example).
  max_rows = max_elements // num_addresses
  # num_addresses = q * max_rows + r (q = 2 and r = 4 in this example).
  q, r = divmod(num_addresses, max_rows)
  dest_addresses = addresses
  time_matrix = []
  # Send q requests, returning max_rows rows per request.
  for i in range(q):
    origin_addresses = addresses[i * max_rows: (i + 1) * max_rows]
    response = send_request(origin_addresses, dest_addresses, API_key)
    time_matrix += build_time_matrix(response)

  # Get the remaining remaining r rows, if necessary.
  if r > 0:
    origin_addresses = addresses[q * max_rows: q * max_rows + r]
    response = send_request(origin_addresses, dest_addresses, API_key)
    time_matrix += build_time_matrix(response)
  return time_matrix


def send_request(origin_addresses, dest_addresses, API_key):
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
  request = request + '&origins=' + origin_address_str + '&destinations=' + \
                       dest_address_str + '&key=' + API_key
  jsonResult = urllib.urlopen(request).read()
  response = json.loads(jsonResult)
  return response


def build_time_matrix(response):
  time_matrix = []
  for row in response['rows']:
    row_list = [row['elements'][j]['duration']['value'] for j in range(len(row['elements']))]
    time_matrix.append(row_list)
  return time_matrix