% This is Exercise 1 of the EECI course "Nonlinear and Data-driven Model Predictive Control" taught by 
% Prof. Matthias A. M�ller and Prof. Frank Allg�wer.
% The goal of this exersice is to implement an MPC alogrithm for a nonlinear
% system with terminal equality constraint.

%% Setup
% clear workspace, close open figures, clear command window
clear all
close all
clc

addpath('/home/aha@eiva.local/Downloads/casadi-linux-matlabR2014b-v3.5.5')
import casadi.*                                
opti = casadi.Opti();       % Create a new optimization problem
opti.solver('ipopt')        % Choose solver

%% System parameters

% continuous system
% Define a function that computes the system dynamics f(x,u):
% x_dot = dynamics( x(t), u(t) )
mu = 0.5; 
dynamics = @(x,u) [x(2)+ u*(mu +(1-mu)*x(1)); x(1) + u*(mu -4*(1-mu)*x(2))]; 

n = 2;	% State dimension
m = 1;	% Input dimension                 

% Parameters
delta = 0.1;                        % Sampling time
T = 5;                             % continuous-time prediction horizon
N = T/delta;                        % discrete-time prediction horizon
tmeasure = 0.0;                     % initial time
x0 = [0.4;-0.5];                   % initial state
SimTime = 5;                        % Simulation time
mpciterations = SimTime / delta;    % discrete simulation time

% cost function
Q = 0.5 * eye(n);
R = 1;

%% Problem 1
% Define optimization variables
x = opti.variable(n,N+1);	% Declares the state sequence as an optimization variable
u = opti.variable(m,N);     % Declares the input sequence as an optimization variable

% Constraints
% dynamic constraint
for k=1:N  
   % Runge-Kutta 4 integration
   k1 = dynamics(x(:,k),         u(:,k));
   k2 = dynamics(x(:,k)+delta/2*k1, u(:,k));
   k3 = dynamics(x(:,k)+delta/2*k2, u(:,k));
   k4 = dynamics(x(:,k)+delta*k3,   u(:,k));
   x_next = x(:,k) + delta/6*(k1+2*k2+2*k3+k4);

   opti.subject_to(x(:,k+1)== x_next);           
end

% Initial constraint
xt = opti.parameter(n,1);       % Set x(t) as a parameter
opti.set_value(xt,x0);          % Set x(t) to be the initial state
opti.subject_to(x(:,1) == xt);  % initial constraint

% terminal constraint
opti.subject_to( x(:,N+1) == zeros(2,1) );

% input constraint
opti.subject_to( -2 <= u <= 2 );

% cost function
J = 0;
for i=1:N
    J = J + x(:,i)'*Q*x(:,i) + u(i)'*R*u(i); % stage cost
end
opti.minimize(J);

% Initial guess
u_init = zeros(m,N);            % Initial guess for input sequence: all 0
opti.set_initial(u,u_init);
x_init = zeros(n,N+1);
x_init(:,1) = x0;               % Initial guess for state sequence via simulation
for k=1:N                                       
    % Runge-Kutta 4 integration
   k1 = dynamics(x_init(:,k),              0);
   k2 = dynamics(x_init(:,k)+delta/2*k1,   0);
   k3 = dynamics(x_init(:,k)+delta/2*k2,   0);
   k4 = dynamics(x_init(:,k)+delta*k3,     0);
   x_init(:,k+1) = x_init(:,k) + delta/6*(k1+2*k2+2*k3+k4);   
end
opti.set_initial(x,x_init);

% solve the OCP
sol = opti.solve();
x_OL = sol.value(x);
u_OL = sol.value(u);

%% Plots
subplot(3,1,1)
	grid on
	plot(0:delta:T,x_OL(1,:))
	xlabel('t')
	ylabel('x_1(t)')
    xlim([0,T])
subplot(3,1,2)
	plot(0:delta:T,x_OL(2,:))
	xlabel('t')
	ylabel('x_2(t)')
    xlim([0,T])
subplot(3,1,3)
	stairs(0:delta:T-delta,u_OL)
	xlabel('t')
	ylabel('u(t)')
    xlim([0,T])
    
%% Problem 2
% Storage
x_MPC = zeros(n,mpciterations+1); x_MPC(:,1) = x0;
u_MPC = zeros(m,mpciterations);
t = 0;




% closed-loop iterations
for ii = 1:mpciterations
    sol = opti.solve();
    x_OL = sol.value(x);
    u_OL = sol.value(u);
    
    % store closed-loop data
    u_MPC(ii) = u_OL(1);
    x_MPC(:,ii+1) = x_OL(:,2);
    t = t + delta;
    
    % update initial constraint
    opti.set_value(xt,x_MPC(:,ii+1));
    
    % update initial guess
    opti.set_initial(x,[x_OL(:,2:N+1),zeros(n,1)]);
    opti.set_initial(u,[u_OL(2:N),0]);

    %Plot state sequences (open-loop and closed-loop) in state space plot (x_1;x_2)
    f2 = figure(2);
    plot(x_MPC(1,1:ii+1),x_MPC(2,1:ii+1),'b')
    grid on, hold on,
    plot(x_OL(1,:),x_OL(2,:),'g')
    plot(x_MPC(1,1:ii+1),x_MPC(2,1:ii+1),'ob')
    xlabel('x(1)')
    ylabel('x(2)')
    title('state space')
    drawnow
    
    %Plot input sequences (open-loop and closed-loop) over time
    f3 = figure(3);
    stairs(t+delta*(0:1:N-1),u_OL), grid on, hold on,
    plot(t,u_MPC(ii),'bo')
    xlabel('t')
    ylabel('u_{OL}')
    drawnow
end
