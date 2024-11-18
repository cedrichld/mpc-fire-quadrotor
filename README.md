# Fire-Quadrotor

### **Project Description**

We are developing a 3D simulation of a quadcopter drone navigating through a dense forest to extinguish a fire. The simulation combines **path planning algorithms**, **dynamic trajectory optimization**, and **control strategies**. This project aims to explore efficient algorithms for navigating challenging environments while maintaining dynamic constraints and handling real-time adjustments. Eventually, we will extend the simulation to advanced visual frameworks (most probably Gazebo).

---

### **Core Objectives**

1. **Forest Navigation**:
   - Create a randomized 3D forest environment with obstacles represented as cylindrical trees.
   - Implement efficient path-planning algorithms (**A*** and **RRT***).
   - Ensure obstacle avoidance and efficient path computation from the drone's start point to a fire zone.

2. **Fire Extinguishing Scenario**:
   - Define a randomly placed fire zone in the forest, bounded by a spline.
   - Simulate the quadcopter reaching the fire zone and performing extinguishing actions (e.g., hovering in the fire zone).

3. **Quadcopter Dynamics and Trajectory Optimization**:
   - Implement the drone's dynamic model and optimize its trajectory using advanced control algorithms.
   - Use **Model Predictive Control (MPC)** to compute dynamically feasible trajectories through the forest while maintaining constraints (e.g., velocity, proximity to obstacles).

4. **Simulation and Visualization**:
   - Visualize the quadcopter navigating through the forest in 3D, using ROS-based tools (Gazebo).
   - Include dynamic, real-time adjustments for trajectory optimization and path re-planning.

---

### **Current Progress**

1. **Path Planning**:
   - **A*** and **RRT*** algorithms are fully implemented for 3D navigation in the forest zone.
   - Both algorithms can compute paths avoiding a dense array of cylindrical obstacles (trees in a forest) and compare results based on time and cost metrics.
   - Path smoothing is integrated to improve trajectory feasibility.

2. **Obstacle Representation**:
   - Cylindrical obstacles with collision detection have been successfully implemented.
   - Visualization of the workspace (forest and fire zone), obstacles, and paths is complete.

3. **Algorithm Comparison**:
   - Performance comparison between **A*** and **RRT*** includes path efficiency, cost, and computation time. RRT* performs much better in denser environments.
  
4. **Forest Visualization with Fire Zone**:
   - Represent fores with trees well dispersed throughout the map.
   - Create a fire zone represented as a small bounded area in the center of the forest, defined by a random spline.

---

### **Next Steps**

1. **Algorithm to extinguish the fire**:
   - Adapt the A*/RRT* algorithms to compute paths once inside the fire zone to extinguish the fire zone.
     
2. **Integration with Path Planning**:
   - Adapt the A*/RRT* algorithms for the cylindrical obstacle configuration and compute paths to the fire zone.

3. **Quadcopter Model**:
   - Develop the drone’s dynamic model, beginning with a simple PID or LQR controller for stabilization.
   - Transition to MPC for optimal trajectory tracking.

4. **Advanced Visualization**:
   - Gazebo integration.

---

### **End Goal**

By the end of the project, we aim to have a fully functional 3D simulation where a quadcopter autonomously navigates a pre-mapped forest, avoids trees in real-time, and reaches a fire zone to perform extinguishing actions. The solution will combine robust path planning, dynamic optimization, and visually immersive simulation, paving the way for more complex real-world scenarios.