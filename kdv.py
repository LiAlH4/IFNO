import torch, numpy as np, matplotlib.pyplot as plt
from torch.fft import fft, ifft, fftfreq, fftshift
from ode import RK23
pi = torch.pi


#  kdv equation
#  u_t = - alpha * (u**2)_x - beta * u_xxx, x \in [0, L]
#  u(x,0) = u0, periodic BC.

# settings
alpha, beta = 3,1
L = 20
N = 256
dx = L/N
k = fftfreq(N, dx)
x = torch.tensor(np.linspace(0,L,N,endpoint=False))

# initial condition
M = 3
u0 = M*(M+1)/(torch.cosh((x-L/2))**2) #+ np.random.rand(u0.size)*0.1

def Op(u,t):
    return ifft(-alpha*(2*pi*1j*k) * fft(u**2) - beta * (2*pi*1j*k)**3 * fft(u)).real

def g(t):
    return torch.exp( -beta * 1j*(2*pi*k)**3*t)

def L_kdv_stiff(u_tilde,t):
    gt = g(t)
    z = -alpha* 1j*2*pi*k * gt * fft(ifft(1/gt * u_tilde)**2)
    return z

u_tilde0 = fft(u0)
t,up = RK23(u_tilde0, L_kdv_stiff, dt=1e-4, T=3)
up = ifft(up / torch.exp(-1j*(2*pi*k[None,:])**3*t[:,None]), axis=-1).real
# t,u = RK23(u0, Op, dt=1e-5, T=0.1) 很难求解

plt.plot(x, up[0], label=f't={t[0]}')
plt.plot(x, up[-1], label=f't={t[-1]}')
plt.legend()
plt.show()