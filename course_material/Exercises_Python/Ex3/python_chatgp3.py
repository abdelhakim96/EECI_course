% Define the number of control intervals and prediction horizon
N = 10;
MPCIterations = 15;

% Define the optimization variables
z = opti.variable(n,N+1); % state trajectory
v = opti.variable(m,N); % input trajectory

% Dynamic constraint
for k = 1:N
    opti.subject_to(z(:,k+1) == Ad*z(:,k) + Bd*v(:,k));
end

% Input & state constraints
for k = 1:N
    opti.subject_to(Hu*v(:,k) <= ku);
    opti.subject_to(Hx*z(:,k+1) <= kx);
end

% Initial constraint
xt = opti.parameter(n,1); % Declare the initial state as an optimization parameter
opti.set_value(xt,x0); % Assign the value of initial state
opti.subject_to(-S.A*z(:,1) <= S.b - S.A*xt); % Initial state constraint

% Terminal constraint
opti.subject_to(Hf*z(:,N+1) <= Kf);

% Define the cost function
J = 0;
% Stage cost
for k = 1:N
    J = J + delta * z(:,k)'*Q*z(:,k) + delta * v(:,k)'*R*v(:,k);
end
% Terminal cost
J = J + z(:,N+1)'*P*z(:,N+1);
opti.minimize(J); % Minimize cost function

%% Simulation of the closed loop
% Memory allocation
x_MPC = zeros(n,MPCIterations); 
x_MPC(:,1) = x0;
u_MPC = zeros(m,MPCIterations-1);
z_MPC = zeros(n,MPCIterations);
v_MPC = zeros(m,MPCIterations-1);
for ii = 1:MPCIterations-1 
    
    % solve the optimization problem
    sol = opti.solve();
    
    % Extract the optimal nominal trajectories z*, v*
    z_OL = sol.value(z);
    v_OL = sol.value(v);
    z_MPC(:,ii) = z_OL(:,1);
    v_MPC(:,ii) = v_OL(:,1);
  
    % Calculate the control input using the tube-based MPC control law
    u_MPC(:,ii) = v_MPC(:,ii) + K*(x_MPC(:,ii) - z_MPC(:,ii));
    
    % Add random disturbance to the system at this iteration
    w_min = -0.05;
    w_max = 0.05;
    w(:,ii) = w_min + (w_max - w_min) * rand(n,1);
    
    % Update the closed-loop system
    x_MPC(:,ii+1) = Ad*x_MPC(:,ii) + Bd*u_MPC(:,ii) + w(:,ii);
    
    % Prepare warmstart solution for next time step
    z_init = [z_OL(:,2:end),A_K*z_OL(:,end)];
    v_init = [v_OL(:,2:end),K*z_OL(:,end)];
    opti.set_initial(z,z_init);
    opti.set_initial(v,v_init);

    % Update the initial constraint parameter for the next iteration
    opti.set_value(xt,x_MPC(:,ii+1));
end 

