import torch, numpy as np, matplotlib.pyplot as plt
from torch.fft import fft, ifft, fftfreq, fftshift
from ode import RK23
pi = torch.pi


#  burgers equation
#  u_t = - alpha * (u**2)_x - beta * u_xx, x \in [0, L]
#  u(x,0) = u0, periodic BC.

# settings
alpha, beta = 1, -0.1
L = 2*pi
N = 256
dx = L/N
k = fftfreq(N, dx)
x = torch.tensor(np.linspace(0,L,N,endpoint=False))

# initial condition
M = 1
u0 = M*(M+1)/(torch.cosh((x-L/2))**2) #+ np.random.rand(u0.size)*0.1

def Op(u,t):
    return ifft(-alpha*(2*pi*1j*k) * fft(u**2) - beta * (2*pi*1j*k)**2 * fft(u)).real


t,u = RK23(u0, Op, dt=1e-3, T=0.5)
plt.plot(x, u[0], label=f't={t[0]}')
plt.plot(x, u[-1], label=f't={t[-1]}')
plt.legend()
plt.show()