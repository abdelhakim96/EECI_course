%%
% computes the maximal output admissible set for a discrete time linear
% system based on [Gilbert, Tan: Linear Systems with State and Control
% Constraints: The Theory and Application of Maximal Output Admissible Sets,
% IEEE Transactions on Automatic Control, vol.36, No. 9, 1991


% autonomous system: x+ = Ax
% constraints:       H*x-h <= 0

% MOAS O_inf defined by G*x <= g



function [O_Inf,G,g]=MaxInvSet(A,H,h)


options = optimset('Display','off');

m=length(h(:,1));


notFinished=1;
fmax=-inf;


h_new=h;
H_new=H;

while(notFinished)
      
 for i=1:m 
    [~,fval]=linprog(-H_new(end-m+i,:)*A,H_new,h_new,[],[],[],[],[],options);
    fmax=max(-fval-h(i),fmax);
 end
 
 if(fmax<=0)
     notFinished=0;
 else
     fmax=-inf;
    
     H_new=[H_new; H_new(end-m+1:end,:)*A];
     h_new=[h_new;h_new(end-m+1:end)];   
 end
end

G=H_new;
g=h_new;
O_Inf = Polyhedron(G,g);
O_Inf.minHRep();
O_Inf.minVRep();

end
