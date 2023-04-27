#import ".typst_pyimage/pyimage.typ": pyimage

Consider the Lotka--Volterra (predator-prey)
equations:

#pyimage("
import diffrax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def func(t, y, args):
  rabbits, cats = y
  d_rabbits = rabbits - rabbits*cats
  d_cats = -cats + rabbits*cats
  return d_rabbits, d_cats

term = diffrax.ODETerm(func)
solver = diffrax.Tsit5()
y0 = (2, 1)
t0 = 0
t1 = 20
dt0 = 0.01
ts = jnp.linspace(t0, t1, 100)
saveat = diffrax.SaveAt(ts=ts)
sol = diffrax.diffeqsolve(term, solver, t0,
                          t1, dt0, y0,
                          saveat=saveat)

plt.plot(ts, sol.ys[0], label='Rabbits')
plt.plot(ts, sol.ys[1], label='Cats')
plt.xlim(0, 20)
plt.ylim(0, 2.5)
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
", width: 70%)