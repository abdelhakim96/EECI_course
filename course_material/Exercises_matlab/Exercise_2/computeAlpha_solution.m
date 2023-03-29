% This is Exercise 2 of the EECI course "Nonlinear and Data-driven Model Predictive Control" taught by 
% Prof. Matthias A. M�ller and Prof. Frank Allg�wer.
% In this exercise, a suitable terminal set for a
% Quasi-Infinite-Horizon MPC scheme shall be computed.
% 
% Here, we use the 4-step procedure described in the paper [Quasi-Infinite Horizon
% Nonlinear Model Predictive Control Scheme with Guaranteed Stability] by
% Chen & Allg�wer. 
% 
% You will need the following addon for MATLAB:
% - ellipsoids - Ellipsoidal Toolbox 1.1.3 lite:
%   http://code.google.com/p/ellipsoids/downloads/list

function [K,P,alpha] = computeAlpha_solution(kappa)
% nonlinear system
mu = 0.5; 
dynamics = @(x,u) [x(2)+ u*(mu +(1-mu)*x(1)); x(1) + u*(mu -4*(1-mu)*x(2))]; 

% Cost function
Q = 0.5*eye(2);
R = 1;

% dimensions (state and input)
n = 2;
m = 1;

% input constraints (box constraints)
u_min = -2;
u_max =  2;

%% Step 1: Compute K such that Jacobi linearization is stabilized
disp('Step 1')
% Linearized system
% A = ???
% B = ???
A = [0 1; 1 0];
B = [0.5; 0.5];

% LQR controller
% K = ???
K = -lqr(A,B,Q,R);

% closed-loop system dynamics
% Attention! The lqr command produces A-B*K, whereas in the paper A+B*K is used!
AK = A+B*K;

disp(['K = [',num2str(K),']'])
disp('eig(AK)='),disp(eig(AK))
disp('______________________________')

%% Step 2: Choose kappa satisfying (6) and solve (5) to optain P
disp('Step 2')
disp(['kappa = ',num2str(kappa)])
% check condition (6)
if kappa >= -max(real(eig(AK)))
    error('kappa >= -max(real(eig(AK)))')
end

% solve Lyapunov equation (5)
% P = ???
% Hint: Use Matlab's lyap() command
P=lyap(AK+kappa*eye(n),Q+K'*R*K);


disp('P = '); disp(P)
disp('______________________________')

%% Step 3: Find largest possible alpha_1 s.t. Kx in U for all x in X^f_alpha_1
disp('Step 3')
% We determine the largest alpha_1, such that Kx\in U \forall x\in X_f
% This is done by looking at each constraint individually and obtaining the
% smallest x'*P*x, such that the inequality constraint is fullfilled with equality.
% Then, we take the smallest of value for alpha to be alpha_1

options = optimset('Display','off');

x_opt_1 = quadprog(P,zeros(2,1),[],[],K,u_max,[],[],[],options);
x_opt_2 = quadprog(P,zeros(2,1),[],[],-K,-u_min,[],[],[],options);
alpha_1 = min(x_opt_1'*P*x_opt_1,x_opt_2'*P*x_opt_2)

disp('______________________________')

%% Step 4: Find largest possible alpha in (0,alpha_1] such that (3) holds
disp('Step 4')
% In this step, we search for a alpha<=alpha_l, such that
% L_Phi<=L_Phi_max.
% This is done using bisection.

% Definition of phi in (1)
phi = @(x) dynamics(x,K*x) - AK*x;

% upper bound for L_Phi
L_Phi_max = ( kappa * min(real(eig(P))) ) / (norm(P,2));

% initial conditions for optimization
alpha_ub = alpha_1;                 % upper bound for alpha
alpha_lb = 0;                       % lower bound for alpha
L_Phi = FcnL_phi(AK,K,P,alpha_1);   % Compute L_phi for alpha_1
exitflag = 1;
nn = 1;
n_max = 100;                        % maximum number of iterations

% Check upper bound for alpha_1
if L_Phi <= L_Phi_max
    alpha_lb = alpha_1;
    exitflag = 0;                   % End bisection if alpha_1 is small enough
end
alpha = 0.5*alpha_1;                % Next guess for alpha
L_Phi = FcnL_phi(AK,K,P,alpha);     % Compute L_phi for next guess


% bisection
while exitflag == 1 && nn <= n_max
    
    alpha_old = alpha;
    
    if L_Phi > L_Phi_max                    % alpha is too big   
        alpha_ub = alpha;                   % new upper bound
    elseif L_Phi <= L_Phi_max && L_Phi ~= 0 % alpha too small
        alpha_lb = alpha;                   % new lower bound
    else
        error('error')
    end
    
    alpha = 0.5*(alpha_ub + alpha_lb);      % New guess by bisection
    L_Phi = FcnL_phi(AK,K,P,alpha);         % Compute L_phi for new guess
    
    % exit conditions
    if abs(alpha - alpha_old)/abs(alpha_old) <= 10^-12 && L_Phi <= L_Phi_max && L_Phi ~= 0
        exitflag = 0;
    end
    nn = nn + 1;    
end

alpha = alpha_lb;                           % alpha = lower bound that satisfies < L_phi_max
disp(['alpha = ',num2str(alpha)])

%% Step 4 (alternative)
alternative_procedure = true;
if alternative_procedure

disp('Step 4 (alternative)')
%  Set lower and upper bounds for alpha for bisection
alpha = alpha_1;
alpha_ub = alpha_1;
alpha_lb = 0;
max_iter = 10;                      % Max iterations for bisection
opt = optimset('MaxFunEvals',10000,'MaxIter',10000,'Display','off'); % Options for fmincon
x2_init = [1 1 -1 -1; 1 -1 1 -1];   % Initialization for fmincon

% start bisection
for i= 1:max_iter
    fval = inf;
    % inner for loop to check multiple initializations for fmincon
    for j = 1:4

        [x2,val] = fmincon(@(x)-(x'*P*phi(x) - kappa*x'*P*x)/(x'*P*x),x2_init(:,j),[],[],[],[],[],[],@(x)nonlcon(x,P,alpha),opt);
        
        % Take minimum value
        fval = min(fval,val);
    end
    % Check condition: Is optimal value of (7) nonpositive  and update upper and lower bounds for alpha  
    fval = -fval; % Correct sign due to maximization
    if fval>=0 % condition not satisfied
        alpha_ub = alpha;
    else % condiction satisfied
        alpha_lb = alpha;
    end
    
    alpha = (alpha_lb + alpha_ub)/2; % bisection
end
alpha = alpha_lb; % Take best known feasible value for alpha

disp(['alpha = ',num2str(alpha)])
end
end


function [c, ceq] = nonlcon(x,P,alpha)
% This function contains the nonlinear equality constraints ceq and
    % the nonlinear inequality constraints c, i.e., the constraints are
    % ceq = 0 and c <= 0
    
    % Inequality constraint x'Px <= alpha
    % c = ???
    c = x'*P*x-alpha;
    
    % No equality constraints
    ceq = [];
end