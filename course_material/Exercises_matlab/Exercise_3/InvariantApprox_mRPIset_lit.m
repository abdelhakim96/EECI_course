% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Computaion of an invariant outer-epsilon approximation of the
% minimal Robust Positivly Invariant Set (mRPI),
% see Rakovic et al., Invariant Approximations of the Minimal Robust invariant Set. 
% IEEE Transactions on Automatic Control 50, 3 (2005), 406–410.
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [F_alpha_s, alpha, s] = InvariantApprox_mRPIset_lit(A, W, epsilon)
% system dimension
n = W.Dim;
% initialization
alpha = 0;
logicalVar = 1;
s = 0;
while logicalVar == 1  
    s = s + 1;
    % alpha_0(s)
    % inequality representation of the set W: f_i'*w <= g_i , i=1,...,I_max
    f_i = (W.A)';
    g_i = W.b;
    I_max = length(W.b);
    % call of the support function h_W
    h_W = zeros(I_max,1);
    for k = 1:I_max
        
        a = (A^s)' * f_i(:,k);
        
        h_W(k) = fkt_h_W(a, W);
        
    end
    clear('k')
    
    % output
    alpha_opt_s = max( h_W ./ g_i );  
    alpha = alpha_opt_s;
    
    %  M(s)
    ej = eye(n);
    sum_vec_A = zeros(n,1);
    sum_vec_B = zeros(n,1);
    updt_A = zeros(n,1);
    updt_B = zeros(n,1);
    
    for k = 1:s
        for j = 1:n
            a = (A^(k-1))' * ej(:,j);
            updt_A(j) = fkt_h_W(a, W);
            updt_B(j) = fkt_h_W(-a, W);
        end
        sum_vec_A = sum_vec_A + updt_A;
        sum_vec_B = sum_vec_B + updt_B;
    end
    clear('k')
    Ms = max(max(sum_vec_A, sum_vec_B));
    % Interrupt criterion
    if alpha <= epsilon/(epsilon + Ms)
        logicalVar = 0;
    end
end
% Fs
Fs = Polyhedron('A', [], 'b', [], 'Ae', eye(n), 'be', zeros(n,1));
for k = 1:s
    Fs = Fs + (A^(k-1)) * W;
end
% F_Inf approx
F_alpha_s = 1/(1 - alpha) * Fs;
%% support function h_W
function [h_W, diagnostics] = fkt_h_W(a, W)

% dimension of w
nn = W.Dim;

% optimization variable
w = sdpvar(nn,1);

% cost function
Objective = -a' * w;

% constraints
Constraints = [ W.A * w <= W.b ];

% optimization
Options = sdpsettings('solver','quadprog','verbose',0);
% Options = sdpsettings('solver','intlinprog','verbose',0);
diagnostics = optimize(Constraints,Objective,Options);

% output
w_opt = value(w);
h_W = a' * w_opt;

end
end


