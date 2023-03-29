%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is Exercise 3 of the EECI course "Nonlinear and Data-driven Model Predictive Control" taught by 
% Prof. Matthias A. Müller and Prof. Frank Allgöwer.
%
% You need the Multi Parametric Toolbox 3 (MPT3) for this MATLAB script!
% https://www.mpt3.org/Main/Installation
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%%
close all
clear all
clc
addpath('/home/hakim/Downloads/casadi-linux-matlabR2014b-v3.5.5')

import casadi.*                 % Import the casadi toolbox
opti = casadi.Opti();           % Create a new optimization problem
opti.solver('ipopt');           % Choose a solver

%% System dynmics

% Parameters
Rm = 2.7;
L = 0.1;
ku = 0.01;
km = 0.05;
J = 0.01;
d = 0.1;

% Sampling time
delta = 0.01;

% continuous-time system
Ac = [-Rm/L, -ku/L;
      km/J,  -d/J];
Bc = [1/L; 0];

% Initial condition
x0 = [-2;4]; 

% Weighting matrices for cost function
Q = [100, 0; 0, 1000]; 
R = 1;

% Prediction horizon
T = 0.05;
N = T/delta;

%% Problem 1: Exact discretization
n = size(Ac,2); % state dimension
m = size(Bc,2); % input dimension

% Exact discretization to obtain discrete-time system
% Ad = ???
% Bd = ???
sysc = ss(Ac,Bc,eye(n),zeros(n,m));

sysd = c2d(sysc,delta,'zoh'); % exact discretization
Ad = sysd.A;
Bd = sysd.B;


%% Problem 2
% Derive K as the LQR controller and the weighting matrix P for the terminal cost function

% Discrete-time Algebraic Riccati Equation 
% P = ???
P = dare(Ad,Bd,Q,R);

% Discrete-time LQR controller
% HINT: We use u = Kx, whereas the Matlab function 'dlqr' uses u=-Kx
% K = ???
K = -dlqr(Ad,Bd,Q,R);

% LQR closed-loop dynamics 
% A_K = ???
A_K = Ad+Bd*K;

% Implement the disturbance and constraint sets by using the 'Polyherdron' function
% State constraint set 
% X = ???
X = Polyhedron([-1 0; 1 0; 0 -1; 0 1],[3; 3;10;10]);

% Input constraint set 
% U = ???
U = Polyhedron([1;-1],[10;10]);

% Disturbance set
% W = ???
W = Polyhedron([1 0; -1 0; 0 1; 0 -1],0.05*ones(4,1));

% compute an invariant outer appoximation of the minimal robust positively
% invariant Set.
% Choose alpha and kappa
% alpha = ??? in [0,1)
% kappa = ???
alpha = 0.25;
kappa = 0;
S = InvariantApprox_mRPIset_lec_solution(A_K,W,alpha,kappa);

% epsilon = 0.1;
% S = InvariantApprox_mRPIset_lit(A_K,W,epsilon);

% minimal representation of S (deleting redundant inequalities)


% S.minHRep();

%% Problem 3
% Implement tightened state and input constraint sets
% Z = X (-) S and V = U (-) KS
% and extract the matrices Hx, kx, Hu, ku
Z = X-S;
V = U-K*S;

Hx = Z.A;
kx = Z.b;
Hu = V.A;
ku = V.b;

% Constraints to determine the terminal set {x|HXf*x<=KXf}
HXf = [Hx; Hu*K];
KXf = [kx; ku];
% Terminal set Zf
% [Zf, Hf, Kf] = MaxInvSet(?,?,?)
[Zf, Hf, Kf] = MaxInvSet(A_K, HXf, KXf);

figure(1)
hold on
plot(X,'color','red')
plot(Zf,'color','cyan')
plot(S,'color','yellow')
plot(W,'color','green')
legend('X','Zf','S','W')
figure(2)
hold on
plot(U,'color','red')
plot(K*S,'color','cyan')
legend('U','K*S')

%% Problem 4
% Implementation of the MPC optimal control problem

% Set number of MPC iterations
MPCIterations = 15;

% Define the optimization variables
% z = ???
% v = ???
z = opti.variable(n,N+1);
v = opti.variable(m,N); 

% Dynamic constraint
for k = 1: N
    opti.subject_to(z(:,k+1) == Ad*z(:,k)+Bd*v(:,k));
end

% Input & state constraints
for k = 1:N
    opti.subject_to(Hu*v(:,k) <= ku);
    opti.subject_to(Hx*z(:,k+1) <= kx);
end

% Initial constraint
% Declare the initial state as an optimization parameter
xt = opti.parameter(n,1);
% Assign the value of initial state
opti.set_value(xt,x0);
% Initial state constraint
opti.subject_to(-S.A*z(:,1) <= S.b - S.A*xt);

% Terminal constraint
opti.subject_to(Hf*z(:,N+1) <= Kf);

% cost function
J = 0;
% stage cost
for k = 1:N
    J = J + delta * z(:,k)'*Q*z(:,k) + delta * v(:,k)'*R*v(:,k);
end
% terminal cost
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
    % z_OL = ???
    % v_OL = ???
    % z_MPC = ???
    % v_MPC = ???
    z_OL = sol.value(z);
    v_OL = sol.value(v);
    z_MPC(:,ii) = z_OL(:,1);
    v_MPC(:,ii) = v_OL(:,1);
  
    % Application of the tube-based MPC control law
    u_MPC(:,ii) = v_MPC(:,ii) + K*(x_MPC(:,ii) - z_MPC(:,ii));
    
    % Disturbance at this iteration (random point in W)
    w_min = -0.05;
    w_max = 0.05;
    w(:,ii) = w_min + (w_max - w_min) * rand(n,1);
    
    % Update the closed-loop system
    x_MPC(:,ii+1) = Ad*x_MPC(:,ii) + Bd*u_MPC(:,ii) + w(:,ii);
    
    % Prepare warmstart solution for next time step (take the endpiece of the optimal solution 
    % and add a last piece) 
    % z_init = ???
    % v_init = ???
    z_init = [z_OL(:,2:end),A_K*z_OL(:,end)];
    v_init = [v_OL(:,2:end),K*z_OL(:,end)];
    opti.set_initial(z,z_init);
    opti.set_initial(v,v_init);

    % udate initial constraint parameter
    opti.set_value(xt,x_MPC(:,ii+1));
end
%% Plots
% Plot the states of the real (disturbed) system and the sequence of the
% initial states of the nominal system
% Plot terminal Set
figure(3)
%S.plot
hold on
plot(Zf+S,'color','cyan'), plot(Zf)
for ii = 1:(MPCIterations-1)
    plot(S+z_MPC(:,ii),'color','yellow')
end
clear('ii')
plot(x_MPC(1,:),x_MPC(2,:))
plot(z_MPC(1,:),z_MPC(2,:),'g-.')
plot(0,0,'x','color','black')
title('state space')
xlabel('x_1')
ylabel('x_2')

% Plot the inputs
figure(4), stairs(u_MPC), hold on, stairs(v_MPC,'g')
title('input')
xlabel('iteration')
ylabel('u')
legend('u_{MPC}','v_{MPC}','Location','northwest')
