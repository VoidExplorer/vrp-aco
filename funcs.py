import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
import copy
import networkx as nx
import matplotlib.pyplot as plt
import os

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

class Data:
    def __init__(self, distance_file, order_file, depot_city=DEPOT_CITY,
                 area_multiplier=TRUCK_AREA_CAPACITY_MULTIPLIER,
                 weight_multiplier=TRUCK_WEIGHT_CAPACITY_MULTIPLIER):
        self.depot_city = depot_city
        self.area_multiplier = area_multiplier
        self.weight_multiplier = weight_multiplier
        try:
            self.distance = pd.read_csv(distance_file)
            self.order = pd.read_csv(order_file)
        except FileNotFoundError as e:
            st.write(f"Error: File not found. {e}")
            st.write("Please ensure the file paths are correct.")
            exit()
        self.cities = self._extract_cities()
        self.city_to_index = {city: i for i, city in enumerate(self.cities)}
        self.index_to_city = {i: city for city, i in self.city_to_index.items()}
        self.distance_matrix = self._matrix()
        self.orders_processed = self._process_orders()
        self.truck_area_cap = self.orders_processed['Total_Area'].max() * self.area_multiplier
        self.truck_weight_cap = self.orders_processed['Total_Weight'].max() * self.weight_multiplier
        self.metadata = self._create_metadata()

    def _extract_cities(self):
        cities = pd.unique(self.order[["Source", "Destination"]].values.ravel("K"))
        cities = sorted(set(cities) | {self.depot_city})
        return cities

    def _matrix(self):
        matrix = np.full((len(self.cities), len(self.cities)), np.inf)
        for _, row in self.distance.iterrows():
            src, dst, dist = row["Source"], row["Destination"], row["Distance(M)"]
            if src in self.city_to_index and dst in self.city_to_index:
                i, j = self.city_to_index[src], self.city_to_index[dst]
                matrix[i, j] = dist
                matrix[j, i] = dist
        np.fill_diagonal(matrix, 0)
        return matrix

    def _process_orders(self):
        orders_df = self.order.copy()
        orders_df['Available_Time'] = pd.to_datetime(orders_df['Available_Time'])
        orders_df['Deadline'] = pd.to_datetime(orders_df['Deadline'])
        
        if 'Item_ID' in orders_df.columns:
            orders_processed = orders_df.groupby('Order_ID').agg(
                Material_ID=('Material_ID', 'first'),
                Source=('Source', 'first'),
                Destination=('Destination', 'first'),
                Available_Time=('Available_Time', 'min'),
                Deadline=('Deadline', 'min'),
                Danger_Type=('Danger_Type', 'first'),
                Total_Area=('Area', 'sum'),
                Total_Weight=('Weight', 'sum'),
                Item_Count=('Item_ID', 'nunique')
            ).reset_index()
        else:
            orders_processed = orders_df.copy()
            orders_processed.rename(columns={'Area': 'Total_Area', 'Weight': 'Total_Weight'}, inplace=True)
            orders_processed['Item_Count'] = 1
        return orders_processed

    def _create_metadata(self):
        metadata = pd.DataFrame(index=range(len(self.cities)))
        metadata['Urgency'] = 1.0
        metadata['Weight'] = 0.0
        
        # Calculate urgency based on deadlines and available times
        for _, order in self.orders_processed.iterrows():
            source_idx = self.city_to_index[order['Source']]
            dest_idx = self.city_to_index[order['Destination']]
            
            time_window = (order['Deadline'] - order['Available_Time']).total_seconds()
            urgency = 1.0 / (time_window + 1e-3)  # Add small constant to avoid division by zero
            
            metadata.loc[source_idx, 'Urgency'] = max(metadata.loc[source_idx, 'Urgency'], urgency)
            metadata.loc[dest_idx, 'Urgency'] = max(metadata.loc[dest_idx, 'Urgency'], urgency)
            
            metadata.loc[source_idx, 'Weight'] += order['Total_Weight']
            metadata.loc[dest_idx, 'Weight'] += order['Total_Weight']
        
        return metadata

    def get_distance(self, city1, city2):
        if city1 == city2:
            return 0
        try:
            i, j = self.city_to_index[city1], self.city_to_index[city2]
            dist = self.distance_matrix[i, j]
            return dist if dist != np.inf else float('inf')
        except KeyError:
            st.write(f"Warning: Distance error between {city1} and {city2}. Returning infinity.")
            return float('inf')

    def preview(self):
        st.title("Data Preview")
        st.write(f"Number of cities: {len(self.cities)}")
        st.write(f"Distance matrix shape: {self.distance_matrix.shape}")
        st.write(f"Truck area capacity: {self.truck_area_cap:.2f}")
        st.write(f"Truck weight capacity: {self.truck_weight_cap:.2f}")
        st.header("Processed orders:\n")
        st.write(self.orders_processed)
        st.header("Metadata:\n")
        st.write(self.metadata)
        df = pd.DataFrame(self.distance_matrix, index=self.cities, columns=self.cities)
        st.header("Distance matrix preview:\n")
        st.write(df.round(0))

    def visualize_network(self):
        G = nx.Graph()
        for i, city1 in enumerate(self.cities):
            for j, city2 in enumerate(self.cities):
                if i < j and self.distance_matrix[i, j] != np.inf:
                    G.add_edge(city1, city2, weight=self.distance_matrix[i, j] / 1000)
        plt.figure(figsize=(15, 12))
        pos = nx.spring_layout(G, k=1, iterations=50)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, alpha=0.6)
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(edge_weights)
        edge_widths = [2 * (w/max_weight) for w in edge_weights]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4)
        nx.draw_networkx_labels(G, pos, font_size=8, bbox=dict(facecolor='white', alpha=0.7))
        plt.title("Cities and Distances Network (in Kilometers)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('cities_network.png', dpi=300, bbox_inches='tight')
        st.write("Network saved as 'cities_network.png'")

class AntColonyOptimizer:
    def __init__(self, distance_matrix, metadata=None, num_ants=30, num_iterations=200, alpha=1.0, beta=2.0, rho=0.5):
        self.distances = distance_matrix
        self.pheromones = np.ones_like(distance_matrix)
        self.metadata = metadata
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.num_cities = distance_matrix.shape[0]
        # Find the index of City_61 in the cities list
        self.depot = 61  # This should match the index of City_61 in your cities list

    def _get_penalty(self, city_idx):
        if self.metadata is None or city_idx not in self.metadata.index:
            return 1.0
        urgency = self.metadata.loc[city_idx, 'Urgency']
        weight = self.metadata.loc[city_idx, 'Weight']
        penalty = (1.0 / (urgency + 1e-3)) + (weight / 1e7)
        return penalty

    def _calculate_probabilities(self, current_city, visited):
        # Only allow visiting the depot at the start and end
        if current_city == self.depot:
            allowed = [i for i in range(self.num_cities) if i not in visited]
        else:
            allowed = [i for i in range(self.num_cities) if i not in visited and i != self.depot]
        
        probs = np.zeros(self.num_cities)

        for j in allowed:
            tau = self.pheromones[current_city][j] ** self.alpha
            eta = (1.0 / self.distances[current_city][j]) ** self.beta
            penalty = self._get_penalty(j)
            probs[j] = tau * eta * penalty

        total = np.sum(probs)
        return probs / total if total > 0 else np.ones(self.num_cities) / self.num_cities

    def _construct_solution(self):
        route = [self.depot]  # Start at depot (City_61)
        visited = set(route)
        total_distance = 0
        current = self.depot

        while len(visited) < self.num_cities:
            probs = self._calculate_probabilities(current, visited)
            next_city = np.random.choice(range(self.num_cities), p=probs)

            if next_city in visited:
                continue

            visited.add(next_city)
            route.append(next_city)
            total_distance += self.distances[current][next_city]
            current = next_city

        # Return to depot
        total_distance += self.distances[current][self.depot]
        route.append(self.depot)

        return route, total_distance

    def run(self):
        best_route = None
        best_distance = float('inf')
        progress_bar = st.progress(0)
        status_text = st.empty()

        for iteration in range(self.num_iterations):
            status_text.text(f"Iteration {iteration + 1}/{self.num_iterations}")
            routes = []
            distances = []

            for _ in range(self.num_ants):
                route, dist = self._construct_solution()
                routes.append(route)
                distances.append(dist)

            min_idx = np.argmin(distances)
            if distances[min_idx] < best_distance:
                best_distance = distances[min_idx]
                best_route = routes[min_idx]

            # Evaporate pheromones
            self.pheromones *= (1 - self.rho)

            # Reinforce best path
            for i in range(len(best_route) - 1):
                a, b = best_route[i], best_route[i + 1]
                self.pheromones[a][b] += 1.0 / best_distance

            progress_bar.progress((iteration + 1) / self.num_iterations)
            st.write(f"Iteration {iteration + 1}: Best distance = {best_distance/1000:.2f} km")

        return best_route, best_distance

def run_vrp(data, num_ants=30, num_iterations=200, alpha=1.0, beta=2.0, rho=0.5):
    # Ensure City_61 is the depot
    depot_idx = data.city_to_index[DEPOT_CITY]
    
    optimizer = AntColonyOptimizer(
        distance_matrix=data.distance_matrix,
        metadata=data.metadata,
        num_ants=num_ants,
        num_iterations=num_iterations,
        alpha=alpha,
        beta=beta,
        rho=rho
    )
    
    # Set the depot index
    optimizer.depot = depot_idx
    
    best_route, best_distance = optimizer.run()
    
    # Convert route indices back to city names
    route_cities = [data.index_to_city[idx] for idx in best_route]
    
    st.header("Best Route Found")
    st.write(f"Total Distance: {best_distance/1000:.2f} km")
    st.write("Optimized Route:")
    st.write(" → ".join(route_cities))
    
    return route_cities, best_distance

# Function to calculate travel time
def calculate_travel_time(dist_meters):
    if dist_meters == float('inf') or TRUCK_SPEED_MPS <= 0:
        return datetime.timedelta(seconds=float('inf'))
    return datetime.timedelta(seconds=(dist_meters / TRUCK_SPEED_MPS))

# Function to check order feasibility
def is_order_feasible(truck_current_weight, truck_current_area,
                      truck_current_time_at_loc, truck_current_loc,
                      order, get_distance_func, truck_area_cap, truck_weight_cap):
    # Check capacity
    if (truck_current_weight + order['Total_Weight'] > truck_weight_cap or
        truck_current_area + order['Total_Area'] > truck_area_cap):
        return False, None, None, None

    # Calculate distance and time to source
    dist_to_source = get_distance_func(truck_current_loc, order['Source'])
    if dist_to_source == float('inf'):
        return False, None, None, None
    travel_time_to_source = calculate_travel_time(dist_to_source)
    arrival_at_source = truck_current_time_at_loc + travel_time_to_source

    # Service time at source
    service_start_at_source = max(arrival_at_source, pd.Timestamp(order['Available_Time']))
    service_finish_at_source = service_start_at_source + datetime.timedelta(seconds=SERVICE_TIME_SECONDS / 2)

    # Calculate distance and time to destination
    dist_source_to_dest = get_distance_func(order['Source'], order['Destination'])
    if dist_source_to_dest == float('inf'):
        return False, None, None, None
    travel_time_to_dest = calculate_travel_time(dist_source_to_dest)
    arrival_at_destination = service_finish_at_source + travel_time_to_dest

    # Service time at destination
    service_start_at_destination = arrival_at_destination
    service_finish_at_destination = service_start_at_destination + datetime.timedelta(seconds=SERVICE_TIME_SECONDS / 2)

    # Check deadline
    if service_finish_at_destination > pd.Timestamp(order['Deadline']):
        return False, None, None, None

    return True, service_start_at_source, service_finish_at_source, service_finish_at_destination

# Function to run ACO algorithm
def run_aco(orders, get_distance, all_locations, location_to_idx, idx_to_location,
            truck_area_cap, truck_weight_cap, n_ants, n_iterations, alpha, beta, rho, q, initial_pheromone):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    n_locations = len(all_locations)
    pheromone_matrix = np.full((n_locations, n_locations), initial_pheromone, dtype=float)
    np.fill_diagonal(pheromone_matrix, 0)

    best_overall_solution = {'routes': [], 'total_distance': float('inf'), 'unserviced_orders': len(orders)}
    BASE_SIMULATION_START_TIME = min(pd.Timestamp(o['Available_Time']) for o in orders) if orders else pd.Timestamp.now()

    st.header("Starting ACO Iterations")
    for iteration in range(n_iterations):
        status_text.text(f"Iteration {iteration + 1}/{n_iterations}")
        all_ants_solutions = []
        
        # Add randomization to initial pheromone values
        if iteration == 0:
            pheromone_matrix += np.random.uniform(0, 0.1, pheromone_matrix.shape)
            np.fill_diagonal(pheromone_matrix, 0)

        for ant_idx in range(n_ants):
            ant_routes = []
            ant_total_distance = 0
            remaining_orders_for_ant = sorted(copy.deepcopy(orders), key=lambda o: (o['Deadline'], o['Available_Time']))
            serviced_order_ids_this_ant = set()
            truck_id_counter = 0

            while remaining_orders_for_ant:
                truck_id_counter += 1
                current_truck_route_details = {
                    'truck_id': f"Ant{ant_idx}_Iter{iteration}_Truck{truck_id_counter}",
                    'orders_serviced': [],
                    'path_cities': [DEPOT_CITY],
                    'route_distance': 0,
                    'events': []
                }
                current_truck_weight = 0
                current_truck_area = 0
                truck_available_at_depot_time = BASE_SIMULATION_START_TIME
                truck_current_loc = DEPOT_CITY
                truck_actual_departure_from_depot = None
                orders_added_to_this_truck_run = False

                while True:
                    candidate_orders_with_info = []
                    time_for_feasibility_check = truck_actual_departure_from_depot if truck_actual_departure_from_depot else truck_available_at_depot_time

                    for i, order in enumerate(remaining_orders_for_ant):
                        is_feasible, s_start_src, s_finish_src, s_finish_dest = is_order_feasible(
                            current_truck_weight, current_truck_area,
                            time_for_feasibility_check, truck_current_loc,
                            order, get_distance, truck_area_cap, truck_weight_cap
                        )
                        if is_feasible:
                            candidate_orders_with_info.append((order, i, (s_start_src, s_finish_src, s_finish_dest)))

                    if not candidate_orders_with_info:
                        break

                    probabilities = []
                    total_pheromone_heuristic = 0
                    from_city_idx = location_to_idx[truck_current_loc]

                    for order_info_tuple in candidate_orders_with_info:
                        order_data, _, _ = order_info_tuple
                        to_source_city = order_data['Source']
                        to_city_idx = location_to_idx[to_source_city]
                        pheromone_val = max(pheromone_matrix[from_city_idx, to_city_idx], 1e-6)
                        dist_to_source = get_distance(truck_current_loc, to_source_city)
                        heuristic_dist = 1.0 / (dist_to_source + 1e-5)
                        _, _, _, est_finish_dest_time = is_order_feasible(
                            current_truck_weight, current_truck_area,
                            time_for_feasibility_check, truck_current_loc, order_data,
                            get_distance, truck_area_cap, truck_weight_cap
                        )
                        time_until_deadline_seconds = (pd.Timestamp(order_data['Deadline']) - est_finish_dest_time).total_seconds()
                        heuristic_urgency = 1.0 / (time_until_deadline_seconds + 1.0) if time_until_deadline_seconds > 0 else 1000.0
                        
                        # Modified heuristic calculation with more emphasis on distance
                        heuristic_val = (heuristic_dist * 0.8) + (heuristic_urgency * 0.2)
                        
                        # Add some randomization to prevent getting stuck in local optima
                        random_factor = np.random.uniform(0.9, 1.1)
                        prob_val = (pheromone_val ** alpha) * (heuristic_val ** beta) * random_factor
                        probabilities.append(prob_val)
                        total_pheromone_heuristic += prob_val

                    if total_pheromone_heuristic == 0 or not probabilities:
                        break
                    probabilities = [p / total_pheromone_heuristic for p in probabilities]

                    try:
                        chosen_candidate_idx = np.random.choice(len(candidate_orders_with_info), p=probabilities)
                    except ValueError:
                        chosen_candidate_idx = random.choice(range(len(candidate_orders_with_info)))

                    selected_order_tuple = candidate_orders_with_info[chosen_candidate_idx]
                    selected_order = selected_order_tuple[0]
                    selected_order_original_idx = selected_order_tuple[1]
                    s_start_src, s_finish_src, s_finish_dest = selected_order_tuple[2]

                    if not orders_added_to_this_truck_run:
                        dist_depot_to_first_source = get_distance(DEPOT_CITY, selected_order['Source'])
                        time_depot_to_first_source = calculate_travel_time(dist_depot_to_first_source)
                        arrival_at_first_source_if_now = truck_available_at_depot_time + time_depot_to_first_source
                        actual_service_start_at_first_source = max(arrival_at_first_source_if_now, pd.Timestamp(selected_order['Available_Time']))
                        truck_actual_departure_from_depot = actual_service_start_at_first_source - time_depot_to_first_source
                        current_truck_route_details['events'].append({
                            'type': 'depart_depot', 'city': DEPOT_CITY,
                            'time': truck_actual_departure_from_depot,
                            'order_id': selected_order['Order_ID']
                        })
                        current_truck_route_details['route_distance'] += dist_depot_to_first_source
                        truck_current_time_at_loc = actual_service_start_at_first_source
                        _, s_start_src, s_finish_src, s_finish_dest = is_order_feasible(
                            current_truck_weight, current_truck_area,
                            truck_actual_departure_from_depot,
                            DEPOT_CITY,
                            selected_order, get_distance, truck_area_cap, truck_weight_cap
                        )
                    else:
                        dist_prev_loc_to_source = get_distance(truck_current_loc, selected_order['Source'])
                        current_truck_route_details['route_distance'] += dist_prev_loc_to_source

                    current_truck_route_details['path_cities'].append(selected_order['Source'])
                    current_truck_route_details['events'].append({
                        'type': 'pickup_start', 'order_id': selected_order['Order_ID'], 'city': selected_order['Source'],
                        'time': s_start_src
                    })
                    current_truck_route_details['events'].append({
                        'type': 'pickup_end', 'order_id': selected_order['Order_ID'], 'city': selected_order['Source'],
                        'time': s_finish_src
                    })

                    dist_source_to_dest = get_distance(selected_order['Source'], selected_order['Destination'])
                    current_truck_route_details['route_distance'] += dist_source_to_dest
                    current_truck_route_details['path_cities'].append(selected_order['Destination'])
                    current_truck_route_details['events'].append({
                        'type': 'delivery_start', 'order_id': selected_order['Order_ID'], 'city': selected_order['Destination'],
                        'time': s_finish_dest - datetime.timedelta(seconds=SERVICE_TIME_SECONDS/2)
                    })
                    current_truck_route_details['events'].append({
                        'type': 'delivery_end', 'order_id': selected_order['Order_ID'], 'city': selected_order['Destination'],
                        'time': s_finish_dest
                    })

                    current_truck_route_details['orders_serviced'].append(selected_order['Order_ID'])
                    serviced_order_ids_this_ant.add(selected_order['Order_ID'])
                    current_truck_weight += selected_order['Total_Weight']
                    current_truck_area += selected_order['Total_Area']
                    truck_current_loc = selected_order['Destination']
                    truck_actual_departure_from_depot = s_finish_dest
                    time_for_feasibility_check = truck_actual_departure_from_depot
                    orders_added_to_this_truck_run = True
                    remaining_orders_for_ant.pop(selected_order_original_idx)

                if orders_added_to_this_truck_run:
                    dist_last_loc_to_depot = get_distance(truck_current_loc, DEPOT_CITY)
                    current_truck_route_details['route_distance'] += dist_last_loc_to_depot
                    current_truck_route_details['path_cities'].append(DEPOT_CITY)
                    time_to_return_depot = calculate_travel_time(dist_last_loc_to_depot)
                    arrival_back_at_depot = truck_actual_departure_from_depot + time_to_return_depot
                    current_truck_route_details['events'].append({
                        'type': 'return_depot', 'city': DEPOT_CITY,
                        'time': arrival_back_at_depot
                    })
                    ant_routes.append(current_truck_route_details)
                    ant_total_distance += current_truck_route_details['route_distance']

            num_unserviced = len(orders) - len(serviced_order_ids_this_ant)
            all_ants_solutions.append({'routes': ant_routes, 'total_distance': ant_total_distance, 'unserviced_orders': num_unserviced})

            if ant_total_distance > 0 and ant_total_distance != float('inf'):
                pheromone_deposit_val = q / ant_total_distance
                for route_detail in ant_routes:
                    for i in range(len(route_detail['path_cities']) - 1):
                        idx1 = location_to_idx[route_detail['path_cities'][i]]
                        idx2 = location_to_idx[route_detail['path_cities'][i+1]]
                        pheromone_matrix[idx1, idx2] += pheromone_deposit_val
                        pheromone_matrix[idx2, idx1] += pheromone_deposit_val

        # Update pheromone evaporation
        pheromone_matrix *= (1 - rho)
        
        # Add minimum pheromone threshold to prevent complete evaporation
        min_pheromone = 0.1
        pheromone_matrix = np.maximum(pheromone_matrix, min_pheromone)
        
        # Update pheromones for all ants
        for ant_solution in all_ants_solutions:
            if ant_solution['total_distance'] > 0 and ant_solution['total_distance'] != float('inf'):
                pheromone_deposit_val = q / ant_solution['total_distance']
                for route_detail in ant_solution['routes']:
                    for i in range(len(route_detail['path_cities']) - 1):
                        idx1 = location_to_idx[route_detail['path_cities'][i]]
                        idx2 = location_to_idx[route_detail['path_cities'][i+1]]
                        pheromone_matrix[idx1, idx2] += pheromone_deposit_val
                        pheromone_matrix[idx2, idx1] += pheromone_deposit_val

        # Sort solutions for elite ant update
        iteration_solutions_sorted = sorted(all_ants_solutions, key=lambda s: (s['unserviced_orders'], s['total_distance']))

        # Elite ant update (best solution of iteration)
        if iteration_solutions_sorted and iteration_solutions_sorted[0]['total_distance'] > 0 and iteration_solutions_sorted[0]['total_distance'] != float('inf'):
            best_iter_solution = iteration_solutions_sorted[0]
            pheromone_deposit_val_elite = (q * 2.0) / best_iter_solution['total_distance']  # Increased elite ant influence
            for route_detail in best_iter_solution['routes']:
                for i in range(len(route_detail['path_cities']) - 1):
                    idx1 = location_to_idx[route_detail['path_cities'][i]]
                    idx2 = location_to_idx[route_detail['path_cities'][i+1]]
                    pheromone_matrix[idx1, idx2] += pheromone_deposit_val_elite
                    pheromone_matrix[idx2, idx1] += pheromone_deposit_val_elite

        current_best_in_iteration = iteration_solutions_sorted[0] if iteration_solutions_sorted else None
        if current_best_in_iteration:
            if (current_best_in_iteration['unserviced_orders'] < best_overall_solution['unserviced_orders'] or
                (current_best_in_iteration['unserviced_orders'] == best_overall_solution['unserviced_orders'] and
                 current_best_in_iteration['total_distance'] < best_overall_solution['total_distance'])):
                best_overall_solution = copy.deepcopy(current_best_in_iteration)

        best_dist_iter_str = f"{current_best_in_iteration['total_distance']/1000:.2f} km, Unserviced: {current_best_in_iteration['unserviced_orders']}" if current_best_in_iteration else "N/A"
        best_overall_dist_str = f"{best_overall_solution['total_distance']/1000:.2f} km, Unserviced: {best_overall_solution['unserviced_orders']}" if best_overall_solution['total_distance'] != float('inf') else "N/A"
        st.write(f"Iteration {iteration + 1}/{N_ITERATIONS} - Best in Iteration: {best_dist_iter_str} - Overall Best: {best_overall_dist_str}")
        progress_bar.progress((iteration + 1) / n_iterations)

    return best_overall_solution