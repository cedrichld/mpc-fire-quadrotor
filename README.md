# Fire-Quadrotor
## .gif visualization:
<p align="center">
  <img src="img/quadrotor_trajectory_TPV_forest3.gif" alt=".gif visualization" height="400">
</p>

### **Project Description**

We are developing a 3D simulation of a quadcopter drone navigating through a **dense forest** to extinguish a fire. The simulation combines **path planning** with **RRT\***, **dynamic trajectory optimization**, and **advanced control strategies** (e.g., **LPV-MPC**). The goal is to explore efficient algorithms for navigating cluttered environments while maintaining dynamic constraints, all while allowing real-time adjustments. Ultimately, the project is planned for integration within advanced visualization frameworks such as Gazebo.

For additional details, including more images and videos of our results, please consult the accompanying **paper**.

---

### **Core Objectives**

1. **Forest Navigation**:
   - **Randomized 3D Forest Generation**: Systematically place cylindrical obstacles (trees) by randomly determining location, radius, and height.
   - **Efficient Path Planning**: Implement and compare **RRT\***-based algorithms to navigate from the drone’s start point to a designated fire zone, ensuring safe obstacle avoidance.

2. **Fire Extinguishing Scenario**:
   - Define a fire zone in the forest as a bounded region (e.g., spline-based).
   - Simulate the quadcopter’s arrival at the fire zone and basic extinguishing actions (hovering, spraying, etc.).

3. **Quadcopter Dynamics and Trajectory Optimization**:
   - Model the full nonlinear dynamics of the quadcopter.  
   - Employ **Linear Parameter-Varying Model Predictive Control (LPV-MPC)** to compute and track feasible trajectories while respecting system constraints on velocity, angular rates, and proximity to obstacles.

4. **Simulation and Visualization**:
   - Visualize the quadcopter’s 3D flight through the generated forest and over the fire zone.
   - Incorporate **dynamic re-planning**: handle control updates at each timestep for precise trajectory tracking.

---

### **Current Progress**

1. **Path Planning with RRT\***:
   - RRT\* quickly finds collision-free 3D paths in dense forests, outperforming simpler algorithms in heavily cluttered environments.
   - Implementation includes path smoothing and obstacle clearance checks.

2. **Dense Forest Representation**:
   - Random generation of cylindrical trees (Fig. 1), each placed with a specified radius and height for collision detection.  
   - A spline-based fire zone is generated at a chosen location in the forest as the final objective.

3. **LPV-MPC Integration**:
   - Developed a **Linear Parameter-Varying** model around the quadrotor’s current state to handle nonlinear dynamics.  
   - Solved a Quadratic Program (QP) at each timestep to minimize tracking error (Fig. 5) while meeting control constraints.
   - Demonstrated different integral steps in the controller (30 vs. 80) to show the effect on path tracking accuracy (Figs. 7 and 8).

4. **Preliminary Results**:
   - The quadrotor successfully follows an RRT\*-generated path (Fig. 2).  
   - Some small path deviations appear, attributed to discretization and tuning of the LPV-MPC.  
   - Higher integral steps in the MPC yield more precise tracking but increase computational effort.

---

### **Figures & Media**

- **Fig. 1** – *Randomly Generated 3D Forest map*  
  ![Fig. 1 – Randomly Generated 3D Forest map](img/3D_Forest.png)

- **Fig. 2** – *Fast and optimal RRT\* pathfinding in obstacle dense forest*  
  ![Fig. 2 – RRT\* Path in dense forest](img/Smoothed_RRT_star_forrest.png)

- **Fig. 4** – *Quadrotor dynamics force diagram*  
  ![Fig. 4 – Quadrotor Dynamics Force Diagram]([url](https://github.com/cedrichld/mpc-fire-quadrotor/blob/main/img/Screenshot%202024-12-16%20at%208.04.23%20PM.png)

- **Fig. 5** – *Final Costs used in the LPV-MPC*  
  ![Fig. 5 – LPV-MPC Cost Function]([url](https://github.com/cedrichld/mpc-fire-quadrotor/blob/main/img/Screenshot%202024-12-16%20at%208.51.35%20PM.png))

- **Fig. 7** – *LPV-MPC with 30 Integral steps (Low Precision)*  
  ![Fig. 7 – MPC with 30 Integral Steps](img/3d_Quad_trajectory_30_integral_steps.png)

- **Fig. 8** – *LPV-MPC with 80 Integral steps (High Precision)*  
  ![Fig. 8 – MPC with 80 Integral Steps](img/3d_Quad_trajectory_80_integral_steps.png)

**Demo Video**  
A short video of the drone navigating the randomized 3D forest can be viewed here:  
[Demo Video Link](img/quadrotor_trajectory_TPV_forest3.mp4)

For further details and additional videos/figures, please consult the accompanying **paper**.

---

### **Next Steps**

1. **Refining Fire-Zone Interaction**:
   - Adapt the RRT\* planner for local maneuvers inside the fire zone to simulate extinguishing behaviors more accurately.

2. **Full Integration & Advanced Visualization**:
   - Incorporate Gazebo for a more realistic 3D environment, including potential sensor modeling (e.g., thermal imaging).

3. **Mass Variation & Future Enhancements**:
   - Introduce a variable mass (e.g., water payload decreasing over time) into the LPV model to simulate dynamics changes.
   - Explore real-world forest data or geospatial imagery for more realistic layouts.

4. **Controller Tuning & Stability**:
   - Further tune the MPC horizon, integral steps, and cost function to reduce path deviation in increasingly cluttered environments.
   - Evaluate alternative control methods or different linearization schemes for enhanced robustness.

---

### **End Goal**

By project completion, we aim to:
- Generate a **collision-free** path from the drone’s start location to the fire zone using **RRT\***.  
- Employ an **LPV-MPC** that accurately follows the 3D path, adapting in real time to changes in the drone’s state.  
- Demonstrate the system in a **Gazebo-based** simulation, showing the quadrotor successfully navigating dense forests and extinguishing a simulated fire. This integrated solution will provide a foundation for more advanced aerial firefighting applications in real-world scenarios.
