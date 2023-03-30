import casadi as ca
mu=0.5
def dynamics( x, u  ):
    return ca.vertcat(  
        x[1]+ u*(mu +(1-mu)*x[0]), 
        x[0] + u*(mu -4 * (1 - mu) *x[1]),
    )
