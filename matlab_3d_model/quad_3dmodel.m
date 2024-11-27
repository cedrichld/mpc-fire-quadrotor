% Simulation times, in seconds.
start_time = 0;
end_time = 10;
dt = 0.005;
times = start_time:dt:end_time;

% Number of points in the simulation.
N = numel(times);

% Physical constants
m = 1.0; % Mass (kg)
g = 9.81; % Gravity (m/s^2)
k = 3.13e-5; % Thrust coefficient
b = 7.5e-7; % Drag coefficient
L = 0.25; % Length of quadcopter arm (m)
kd = 0.1; % Linear drag coefficient
I = diag([0.02, 0.02, 0.04]); % Moment of inertia matrix (kg·m^2)

% Initial simulation state.
x = [0; 0; 10]; % Initial position (m)
xdot = zeros(3, 1); % Initial linear velocity (m/s)
theta = zeros(3, 1); % Initial angles (rad)
thetadot = deg2rad(2 * 100 * rand(3, 1) - 100); % Initial angular velocity (rad/s)

% Desired angles for stabilization
desired_theta = [0; 0; 0]; % Roll, pitch, yaw

% Controller gains
Kp = diag([0.02, 0.02, 0.02]); % Proportional gains
Kd = diag([1, 1, 1]); % Derivative gains

% Pre-allocate storage for trajectory
trajectory = zeros(3, N);
angles = zeros(3, N);
angular_velocity = zeros(3, N);
motor_inputs = zeros(4, N);



% Simulation loop
for t_idx = 1:N
    t = times(t_idx); % Current time step

    % Compute current angular velocity
    omega = thetadot2omega(thetadot, theta);
    
    % Compute thrust and torques
    a = acceleration([0; 0; 0; 0], theta, xdot, m, g, k, kd);
    omegadot = angular_acceleration([0; 0; 0; 0], omega, I, L, b, k);

    % Controller input
    inputs = controller([x; xdot], theta, thetadot, desired_theta, Kp, Kd, m, g, k, L, b);

    % Update angular velocity and angles
    omega = omega + dt * omegadot;
    thetadot = omega2thetadot(omega, theta);
    theta = theta + dt * thetadot;

    % Update linear velocity and position
    xdot = xdot + dt * a;
    x = x + dt * xdot;
    
    % Store results for visualization
    trajectory(:, t_idx) = x; % Store position
    angles(:, t_idx) = theta; % Store angles
    motor_inputs(:, t_idx) = inputs; % Store motor inputs
    angular_velocity(:, t_idx) = omega; % Store angular velocity
end



figure;
plot3(x(1, :), x(2, :), x(3, :), 'LineWidth', 1.5);
grid on;
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title('Quadcopter Trajectory');


figure;
plot(times, angles(1, :), 'r', 'LineWidth', 1.5); hold on;
plot(times, angles(2, :), 'g', 'LineWidth', 1.5);
plot(times, angles(3, :), 'b', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Angles (rad)');
legend('Roll (\phi)', 'Pitch (\theta)', 'Yaw (\psi)');
title('Roll, Pitch, and Yaw Angles');


figure;
plot(times, angular_velocity(1, :), 'r', 'LineWidth', 1.5); hold on;
plot(times, angular_velocity(2, :), 'g', 'LineWidth', 1.5);
plot(times, angular_velocity(3, :), 'b', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Angular Velocity (rad/s)');
legend('\omega_x', '\omega_y', '\omega_z');
title('Angular Velocities');


figure;
plot(times, sqrt(motor_inputs(1, :)), 'r', 'LineWidth', 1.5); hold on;
plot(times, sqrt(motor_inputs(2, :)), 'g', 'LineWidth', 1.5);
plot(times, sqrt(motor_inputs(3, :)), 'b', 'LineWidth', 1.5);
plot(times, sqrt(motor_inputs(4, :)), 'k', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Motor Speeds (rad/s)');
legend('\gamma_1', '\gamma_2', '\gamma_3', '\gamma_4');
title('Motor Speeds');


figure;
hold on;
grid on;
xlim([-5, 5]);
ylim([-5, 5]);
zlim([0, 15]);
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title('Quadcopter Animation');

% Initialize the plot
quad_plot = plot3(0, 0, 0, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');

for t_idx = 1:N
    % Update the quadcopter position
    set(quad_plot, 'XData', trajectory(1, t_idx), ...
                   'YData', trajectory(2, t_idx), ...
                   'ZData', trajectory(3, t_idx));
    drawnow;
    pause(dt); % Pause for the simulation time step
end






function R = rotation(angles)
    phi = angles(1);
    theta = angles(2);
    psi = angles(3);
    R = [
        cos(theta)*cos(psi), cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi), cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi);
        cos(theta)*sin(psi), sin(psi)*sin(theta)*sin(phi) + cos(psi)*cos(phi), sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi);
        -sin(theta), cos(theta)*sin(phi), cos(theta)*cos(phi)
    ];
end

function omega = thetadot2omega(thetadot, theta)
    phi = theta(1);
    theta = theta(2);
    T = [
        1, 0, -sin(theta);
        0, cos(phi), cos(theta)*sin(phi);
        0, -sin(phi), cos(theta)*cos(phi)
    ];
    omega = T * thetadot;
end

function thetadot = omega2thetadot(omega, theta)
    phi = theta(1);
    theta = theta(2);
    T_inv = [
        1, sin(phi)*tan(theta), cos(phi)*tan(theta);
        0, cos(phi), -sin(phi);
        0, sin(phi)/cos(theta), cos(phi)/cos(theta)
    ];
    thetadot = T_inv * omega;
end

function T = thrust(inputs, k)
    T = [0; 0; k * sum(inputs)];
end

function tau = torques(inputs, L, b, k)
    tau = [
        L * k * (inputs(1) - inputs(3)); % Roll
        L * k * (inputs(2) - inputs(4)); % Pitch
        b * (inputs(1) - inputs(2) + inputs(3) - inputs(4)) % Yaw
    ];
end

function a = acceleration(inputs, angles, xdot, m, g, k, kd)
    gravity = [0; 0; -g];
    R = rotation(angles);
    T = R * thrust(inputs, k);
    Fd = -kd * xdot;
    a = gravity + (1 / m) * T + Fd;
end

function omegadot = angular_acceleration(inputs, omega, I, L, b, k)
    tau = torques(inputs, L, b, k);
    omegadot = inv(I) * (tau - cross(omega, I * omega));
end

function inputs = controller(state, theta, thetadot, desired_theta, Kp, Kd, m, g, k, L, b)
    % PD control
    error = desired_theta - theta;
    error_dot = -thetadot;
    tau = Kp * error + Kd * error_dot;

    % Compute motor inputs using solve_motor_inputs
    inputs = solve_motor_inputs(tau, m, g, k, theta, L, b);
end


function gamma = solve_motor_inputs(tau, m, g, k, theta, L, b)
    % Total thrust needed to keep the quadcopter aloft
    T = m * g / (cos(theta(1)) * cos(theta(2)));

    % Matrix to map motor inputs to thrust and torques
    A = [
        1, 1, 1, 1;           % Total thrust
        L, 0, -L, 0;          % Roll torque
        0, L, 0, -L;          % Pitch torque
        b, -b, b, -b          % Yaw torque
    ];

    % Desired thrust and torques
    b_vec = [T; tau];

    % Solve for motor inputs (squared angular velocities)
    gamma = A \ b_vec;
end
