# Vehicle Routing Problem (VRP) using Ant Colony Optimization (ACO)

## Project Analysis Report

## 1. Project Overview

This project implements a solution for the Vehicle Routing Problem (VRP) using Ant Colony Optimization (ACO). It's designed to optimize delivery routes for multiple vehicles while considering various real-world constraints including:

- Vehicle capacity (area and weight)
- Time windows (pickup and delivery)
- Multiple depots
- Service times
- Vehicle speed

The implementation is provided in both Python script format and Jupyter Notebook format, with a Streamlit web interface for easy interaction.

## 2. Key Components

### 2.1 Core Parameters

```python
DEPOT_CITY = 'City_61'
TRUCK_SPEED_MPS = 60 * 1000 / 3600  # 60 km/h
SERVICE_TIME_SECONDS = 30 * 60  # 30 minutes
TRUCK_AREA_CAPACITY_MULTIPLIER = 1.5
TRUCK_WEIGHT_CAPACITY_MULTIPLIER = 1.5
```

### 2.2 ACO Parameters

```python
N_ANTS = 50
N_ITERATIONS = 50
ALPHA = 1.0  # Pheromone importance
BETA = 2.0   # Heuristic information importance
RHO = 0.1    # Pheromone evaporation rate
Q = 100      # Pheromone deposit factor
```

## 3. Main Components

### 3.1 Data Management (`Data` class)

- Handles input data from CSV files (order_large.csv, order_small.csv, distance.csv)
- Processes orders and distances
- Creates distance matrices
- Manages city indexing
- Visualizes network using NetworkX

### 3.2 Order Feasibility Checker

Key features:

- Validates capacity constraints
- Checks time windows
- Calculates travel and service times
- Ensures deadline compliance

### 3.3 ACO Implementation

The ACO algorithm includes:

1. Initialization of pheromone trails
2. Ant solution construction
3. Local pheromone updates
4. Global pheromone updates
5. Solution evaluation and selection

### 3.4 Route Management

Features:

- Tracks vehicle routes
- Manages pickup and delivery events
- Calculates distances and times
- Handles vehicle capacity constraints

## 4. Algorithm Flow

1. **Initialization**
   - Load order and distance data
   - Create distance matrix
   - Initialize pheromone trails

2. **Main Loop (For each iteration)**
   - Create multiple ant solutions
   - Each ant constructs a solution:
     - Select orders based on pheromone and heuristic information
     - Build feasible routes
     - Update local pheromone trails

3. **Solution Construction**
   - Consider vehicle constraints
   - Check time windows
   - Calculate probabilities based on:
     - Pheromone levels (α parameter)
     - Heuristic information (β parameter)
     - Distance
     - Urgency (deadline proximity)

4. **Pheromone Update**
   - Evaporate existing pheromone (ρ parameter)
   - Deposit new pheromone based on solution quality
   - Elite ant reinforcement

## 5. Key Features

1. **Multi-Constraint Handling**
   - Vehicle capacity (area and weight)
   - Time windows
   - Service times
   - Travel times

2. **Dynamic Route Construction**
   - Real-time feasibility checking
   - Adaptive order selection
   - Multiple vehicle management

3. **Solution Quality Metrics**
   - Total distance
   - Number of vehicles used
   - Unserviced orders
   - Detailed timeline of events

4. **Visualization**
   - Network visualization
   - Solution reporting
   - Progress tracking

## 6. User Interface

The project provides two interfaces:

1. **Jupyter Notebook** (`VRP_Final[1].ipynb`)
   - Interactive development
   - Visualization
   - Result analysis

2. **Streamlit Web Interface** (`main.py`)
   - Parameter adjustment
   - Real-time execution
   - Result visualization
   - User-friendly controls

## 7. Input Data Format

### 7.1 Order Data Format

Example from order_small.csv:

```csv
Order_ID,Material_ID,Item_ID,Source,Destination,Available_Time,Deadline,Danger_Type,Area,Weight
A140109,B-6128,P01-79c46a02...,City_61,City_54,2022-04-05 23:59:59,2022-04-11 23:59:59,type_1,38880,30920000
```

Key fields:

- Order_ID: Unique identifier for each order
- Source/Destination: Pickup and delivery locations
- Available_Time/Deadline: Time window constraints
- Area/Weight: Capacity requirements
- Danger_Type: Type classification

### 7.2 Distance Data

The distance.csv file contains:

- City pairs
- Distances in meters
- Bidirectional connectivity information

## 8. Output and Results

The system provides detailed solution information:

- Total distance traveled
- Number of vehicles used
- Unserviced orders
- Detailed route information per vehicle
- Timeline of events (pickup/delivery)
- Service times and locations

Example output format:

```
Total Distance: X km
Number of Trucks Used: Y
Number of Unserviced Orders: Z

Truck_1:
  Serviced Orders: [Order_IDs]
  Path: City_A -> City_B -> City_C
  Distance: X km
  Timeline of Events:
    [Timestamp]: pickup_start for Order_1 at City_A
    [Timestamp]: pickup_end for Order_1 at City_A
    ...
```

## 9. Optimization Strategy

The ACO implementation uses several strategies to improve solution quality:

1. **Exploration vs Exploitation**
   - α parameter controls pheromone importance
   - β parameter controls heuristic information importance
   - Balance between known good paths and exploring new options

2. **Solution Enhancement**
   - Elite ant reinforcement
   - Dynamic heuristic information
   - Multiple ant solutions per iteration

3. **Local Search Prevention**
   - Pheromone evaporation (ρ parameter)
   - Prevents stagnation in local optima
   - Maintains search diversity

4. **Constraint Handling**
   - Capacity constraints
   - Time window feasibility
   - Service time requirements

## 10. Conclusion

This VRP implementation using ACO demonstrates a sophisticated approach to solving complex routing problems. The solution is:

- Scalable to handle large datasets
- Flexible with multiple constraints
- User-friendly with multiple interfaces
- Well-documented and maintainable

The project successfully combines theoretical ACO concepts with practical routing constraints, making it suitable for real-world applications in logistics and delivery optimization.
