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
N_ANTS = 50
N_ITERATIONS = 50
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

st.title("Ant Colony Optimization")

form_data = {
    "n_ants": None,
    "n_iterations": None,
    "alpha": None,
    "beta": None,
    "rho": None,
    "Q": None,
    "initial_pheromone": None,
    "orders_dataset": None
}

with st.form(key="run_form"):
    form_data["n_ants"] = st.number_input("Number of ants", 3)
    form_data["n_iterations"] = st.number_input("Number of iterations", 3)
    form_data["alpha"] = st.number_input("Alpha", 1)
    form_data["beta"] = st.number_input("Beta", 2)
    form_data["rho"] = st.number_input("Pheromone evaporation rate", 0.1)
    form_data["Q"] = st.number_input("Q", 100)
    form_data["initial_pheromone"] = st.number_input("Initial pheromone matrix values", 1)
    form_data["orders_dataset"] = st.selectbox("Choose orders dataset", ["order_large", "order_small"])
    run_button = st.form_submit_button("Run")
    
    if run_button:
        run(**form_data)