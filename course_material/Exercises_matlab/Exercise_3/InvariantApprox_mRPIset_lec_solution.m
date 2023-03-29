%~*************************************************************************
% RPI_SET to determinate the RPI set for robust tube-based MPC 
% by fixing alpha and changing kappa. The algorithm has been explained in the lecture.
% S = RPI_Set(A_K,W,alpha,kappa)
% A_K: closed-loop A matrix of the system. it must be Schur stable
% W: disturbance set 
% alpha: must be in the interval [0,1)
%~*************************************************************************
%%
function S = InvariantApprox_mRPIset_lec_solution(A_K, W, alpha, kappa)
% Check alpha and kappa
count=1;
if (alpha <= 0) || (alpha >= 1)
   error('alpha value must be in the inteval (0,1)');
elseif kappa < 0
   error('kappa must be greater than or equal to zero');
end

% Compute S_kappa
S_kappa = Polyhedron();
for i = 0:kappa-1
    S_kappa = S_kappa+A_K^i*W;
    dis('inside for')
end
% Check (8) -> A^kappa *W is a subset of alpha*W
while ~(A_K^kappa*W <= alpha*W)
    % Update kappa, S_k
    count=count+1;
    disp(count)
    kappa = kappa+1;
    S_kappa = S_kappa+A_K^(kappa-1)*W;
end    
disp('kappa')
disp(kappa)
% Compute S
S = (1-alpha)^(-1)*S_kappa;

end