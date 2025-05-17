import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
import copy
import networkx as nx
import matplotlib.pyplot as plt
import os
from funcs import *

# VRP and ACO parameters
DEPOT_CITY = 'City_61'
TRUCK_AREA_CAPACITY_MULTIPLIER = 1.5
TRUCK_WEIGHT_CAPACITY_MULTIPLIER = 1.5
TRUCK_SPEED_MPS = 60 * 1000 / 3600  # 60 km/h to meters/second
SERVICE_TIME_SECONDS = 30 * 60  # 30 minutes
N_ANTS = 1
N_ITERATIONS = 1
ALPHA = 1.0
BETA = 2.0
RHO = 0.1
Q = 100
INITIAL_PHEROMONE = 1.0


# Function to st.write solution details
def print_solution_details(solution, orders_df):
    st.header("Final Best Overall Solution")
    if not solution or solution['total_distance'] == float('inf'):
        st.write("No feasible solution found.")
        return

    st.subheader(f"Total Distance: {solution['total_distance'] / 1000:.2f} km")
    st.subheader(f"Number of Trucks Used: {len(solution['routes'])}")
    st.subheader(f"Number of Unserviced Orders: {solution['unserviced_orders']}")

    total_orders_serviced_in_best = 0
    all_order_ids_in_solution = set()

    for i, route in enumerate(solution['routes']):
        truck_id = route.get('truck_id', f"Truck_{i+1}")
        st.write(f"\n{truck_id}:")
        st.write(f"  Serviced Orders: {route['orders_serviced']}")
        total_orders_serviced_in_best += len(route['orders_serviced'])
        for order_id in route['orders_serviced']:
            all_order_ids_in_solution.add(order_id)

        st.write(f"  Path: {' -> '.join(route['path_cities'])}")
        st.write(f"  Distance for this truck: {route['route_distance'] / 1000:.2f} km")

        st.write("  Timeline of Events:")
        sorted_events = sorted(route['events'], key=lambda x: x['time'])
        for event in sorted_events:
            event_time_str = event['time'].strftime('%Y-%m-%d %H:%M:%S')
            if event['type'] in ['pickup_start', 'pickup_end', 'delivery_start', 'delivery_end']:
                st.write(f"    {event_time_str}: {event['type']} for Order {event['order_id']} at {event['city']}")
            else:
                st.write(f"    {event_time_str}: {event['type']} at {event['city']}")

    st.write(f"\nTotal orders in dataset: {len(orders_df)}")
    st.write(f"Total orders serviced in best solution: {total_orders_serviced_in_best}")

    if total_orders_serviced_in_best < len(orders_df):
        all_dataset_order_ids = set(o['Order_ID'] for o in orders_df)
        unserviced_orders_set = all_dataset_order_ids - all_order_ids_in_solution
        st.write(f"\nWarning: {len(unserviced_orders_set)} orders were not serviced.")
        if unserviced_orders_set:
            st.write(f"Unserviced Order IDs: {list(unserviced_orders_set)}")

ORDER_FILE_PATH = "order_large.csv"
DISTANCE_FILE_PATH = "distance.csv"

# Create Data object
def run(n_ants, n_iterations, alpha, beta, rho, Q, initial_pheromone, orders_dataset):
    data = Data(distance_file=DISTANCE_FILE_PATH, order_file=orders_dataset)
    data.preview()
    data.visualize_network()  # Visualize network (optional)

    # Check if orders were loaded
    if data.orders_processed.empty:
        st.write("No orders loaded. Exiting.")
    else:
        # Run ACO algorithm
        final_best_solution = run_aco(
            orders=data.orders_processed.to_dict('records'),
            get_distance=data.get_distance,
            all_locations=data.cities,
            location_to_idx=data.city_to_index,
            idx_to_location=data.index_to_city,
            truck_area_cap=data.truck_area_cap,
            truck_weight_cap=data.truck_weight_cap,
            n_ants=n_ants,
            n_iterations=n_iterations,
            alpha=alpha,
            beta=beta,
            rho=rho,
            q=Q,
            initial_pheromone=initial_pheromone,
        )
        # st.write solution details
        print_solution_details(final_best_solution, data.orders_processed.to_dict('records'))

# run_button = st.button("Run")
# if run_button:
#     run()

st.title("Vehicle Routing Problem with Ant Colony Optimization")

# File paths
DISTANCE_FILE_PATH = "distance.csv"
ORDER_FILE_PATH = "order_large.csv"

# Create form for parameters
with st.form(key="run_form"):
    st.subheader("ACO Parameters")
    num_ants = st.number_input("Number of ants", min_value=1, max_value=100, value=30)
    num_iterations = st.number_input("Number of iterations", min_value=1, max_value=500, value=200)
    alpha = st.number_input("Alpha (pheromone influence)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    beta = st.number_input("Beta (heuristic influence)", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
    rho = st.number_input("Rho (evaporation rate)", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
    
    st.subheader("Dataset Selection")
    orders_dataset = st.selectbox("Choose orders dataset", ["order_large.csv", "order_small.csv"])
    
    run_button = st.form_submit_button("Run Optimization")

if run_button:
    # Create Data object
    data = Data(
        distance_file=DISTANCE_FILE_PATH,
        order_file=orders_dataset
    )
    
    # Show data preview
    st.write(f"Depot City: {DEPOT_CITY}")
    data.preview()
    data.visualize_network()
    
    # Run VRP optimization
    route_cities, total_distance = run_vrp(
        data=data,
        num_ants=num_ants,
        num_iterations=num_iterations,
        alpha=alpha,
        beta=beta,
        rho=rho
    )
    
    # Display results
    st.header("Optimization Results")
    st.write(f"Total Distance: {total_distance/1000:.2f} km")
    st.write("Optimized Route:")
    st.write(" → ".join(route_cities))
    
    # Create a DataFrame for the route details
    route_details = []
    for i in range(len(route_cities)-1):
        from_city = route_cities[i]
        to_city = route_cities[i+1]
        distance = data.get_distance(from_city, to_city)
        route_details.append({
            'From': from_city,
            'To': to_city,
            'Distance (km)': distance/1000
        })
    
    st.header("Route Details")
    st.dataframe(pd.DataFrame(route_details))
    
    # Verify depot is first and last
    if route_cities[0] != DEPOT_CITY or route_cities[-1] != DEPOT_CITY:
        st.warning("Warning: Route does not start and end at the depot!")