% This is Exercise 2 of the EECI course "Nonlinear and Data-driven Model Predictive Control" taught by 
% Prof. Matthias A. Müller and Prof. Frank Allgöwer.
% In this exercise you are supposed to implement a Quasi-Infinite Horizon MPC algorithm.
% 
% 
% You will need the following two addons for MATLAB:
% - ellipsoids - Ellipsoidal Toolbox 1.1.3 lite:
%   http://code.google.com/p/ellipsoids/downloads/list
% - casadi:
%   https://web.casadi.org/

% clear workspace, close open figures
clear all
close all
clc

import casadi.*                               % Import the casadi toolbox
opti = casadi.Opti();                         % Create a new optimization problem
opti.solver('ipopt');                         % Choose a solver

%% System parameters

% continuous system
% Define a function that computes th system dynamics f(x,u):
% x_dot = dynamics( x(t), u(t) )
mu = 0.5; 
dynamics = @(x,u) [x(2)+ u*(mu +(1-mu)*x(1)); x(1) + u*(mu -4*(1-mu)*x(2))]; 

n = 2;	% State dimension
m = 1;	% Input dimension                 

% Parameters
delta = 0.1;                        % Sampling time
T = 1.5;                            % continuous-time prediction horizon
N = T/delta;                        % discrete-time prediction horizon
tmeasure = 0.0;                     % initial time
x0 = [-0.4;-0.5];                   % initial state
SimTime = 2;                        % Simulation time
mpciterations = SimTime / delta;    % discrete simulation time

% cost function
Q = 0.5 * eye(n);
R = 1;

%% Problem 1
% Define optimization variables
x = opti.variable(n,N+1);           % Declares the state sequence as an optimization variable
u = opti.variable(m,N);             % Declares the input sequence as an optimization variable

% Constraints
% dynamic constraints
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

% input constraint
opti.subject_to( -2 <= u <= 2 );

% cost function
J = 0;
for i=1:N
    J = J + delta*x(:,i)'*Q*x(:,i) + delta*u(i)'*R*u(i); % stage cost
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

% Storage
x_MPC = zeros(n,mpciterations+1); x_MPC(:,1) = x0;
u_MPC = zeros(m,mpciterations);
t = 0;

% closed-loop iterations
f2 = figure(2); 
hold on
plot(0,0,'rx')
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
    plot(x_MPC(1,1:ii+1),x_MPC(2,1:ii+1),'b'), 
    grid on, hold on, axis equal,
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
%% Problem 2
close all
% Terminal set and cost
kappa = 0.95;
[K_loc,P,alpha] = computeAlpha_solution(kappa);
% K_loc = [-2.118, -2.118];
% P = [16.5926, 11.5926; 11.5926, 16.5926];
% alpha = 0.2495;
% alpha = 0.732; % obtained via the alternative way

% Plot terminal region
% ellipsoids toolbox needed (Matlab central)
%E = ellipsoid(alpha*inv(P));
%figure(2)
%hold on
%plot(E,'r')

%% Problem 3
% Terminal region constraint
x_eq = zeros(n,1);
opti.subject_to((x(:,N+1)-x_eq)'*P*(x(:,N+1)-x_eq) <= alpha);

% cost function
J = 0;
for i=1:N
    J = J + delta*x(:,i)'*Q*x(:,i) + delta*u(i)'*R*u(i); % stage cost
end
% terminal cost
J = J + x(:,N+1)'*P*x(:,N+1);
opti.minimize(J);

% reset initial constraint from simulation in Problem 1
opti.set_value(xt,x0);

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
    opti.set_initial(x,[x_OL(:,2:N+1),x_OL(:,end)+delta*dynamics(x_OL(:,end),K_loc*x_OL(:,end))]); % Euler approximation for initial guess
    opti.set_initial(u,[u_OL(2:N),K_loc*x_OL(:,end)]);

    %Plot state sequences (open-loop and closed-loop) in state space plot (x_1;x_2)
    f2 = figure(2);
    plot(x_MPC(1,1:ii+1),x_MPC(2,1:ii+1),'b'), 
    grid on, hold on, axis equal,
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
    ylabel('u_OL')
    drawnow
end