import torch
from typing import Any
Operator = Any

def RK23(u0: torch.tensor,L: Operator, dt:float,T:float)->torch.tensor:
    '''
    Solve PDE: u_t = L(u, t), t \in [0, T], time step dt.
    Input: 
        u0.  torch.tensor.  []
    Output:
        t, u_path. 
    '''
    u = u0
    steps = int(T/dt)
    u_path = []
    t = 0
    for i in range(steps):
        u_path.append(u)

        u1 = u + dt*L(u,t)
        u2 = 3/4*u + 1/4*u1 + 1/4*dt*L(u1,t)
        u = 1/3*u + 2/3*u2 + 2/3*dt*L(u2,t)
        # k1 = L(u, t)
        # k2 = L(u + 2/3*dt*k1, t+ 2/3*dt)
        # u = u + dt*(1/4*k1 + 3/4*k2)

        t = t + dt

    t = torch.linspace(0, T, steps)
    return t,  torch.stack(u_path,axis=0)