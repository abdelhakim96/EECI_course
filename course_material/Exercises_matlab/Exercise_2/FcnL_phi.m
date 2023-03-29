function L_Phi = FcnL_phi(AK,K,P,alpha)
    % nonlinear system
    mu = 0.5; 
    dynamics = @(x,u) [x(2)+ u*(mu +(1-mu)*x(1)); x(1) + u*(mu -4*(1-mu)*x(2))]; 
    
    % Phi
    phi = @(x) dynamics(x,K*x) - AK*x;
    %phi = @(x)  - AK*x;
  
    % L_Phi
    opt = optimset('MaxFunEvals',10000,'MaxIter',10000,'Display','off');
    
    [x,L_Phi_tilde] = fmincon(@(x) -sqrt(phi(x)' * phi(x))/sqrt(x'*x) ,...
        [10.0;10.0],[],[],[],[],[],[],@(x)nonlinConsAlpha(x,P,alpha),opt);
    
    L_Phi = -L_Phi_tilde;
    

end

function [c, ceq] = nonlinConsAlpha(x, P, alpha)
% All states inside ellipse X_{\alpha}^f = {x \neq 0 | x'*P*x \leq alpha}

    c = x'*P*x - alpha;
    ceq = [];

end