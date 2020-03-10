# Elastic wave equation implementation on a staggered grid

This second elastic tutorial extends the previous constant parameter implementation to varying parameters (Lame parameters) and takes advantage of the Tensorial capabilities of Devito to write the elastic wave equation following its mathematical definition. The staggering is automated via the TensorFunction API.

## Explosive source

We will first attempt to replicate the explosive source test case described in [1], Figure 4. We start by defining the source signature $g(t)$, the derivative of a Gaussian pulse, given by Eq 4:

$$g(t) = -2 \alpha(t - t_0)e^{-\alpha(t-t_0)^2}$$


```python
from devito import *
from seismic.source import RickerSource, Receiver, TimeAxis
from seismic import plot_image, demo_model
import numpy as np

import matplotlib.pyplot as plt

from sympy import init_printing, latex
init_printing(use_latex='mathjax')

# Some ploting setup
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20)
```


```python
#NBVAL_IGNORE_OUTPUT
# Initial grid: 3km x 3km, with spacing 10m
nlayers = 5
model = demo_model(preset='layers-elastic', nlayers=nlayers, shape=(301, 301), spacing=(10., 10.))
```

    Operator `initdamp` run in 0.01 s
    Operator `padfunc` run in 0.01 s
    Operator `padfunc` run in 0.01 s
    Operator `padfunc` run in 0.01 s



```python
#NBVAL_SKIP
aspect_ratio = model.shape[0]/model.shape[1]

plt_options_model = {'cmap': 'jet', 'extent': [model.origin[0], model.origin[0] + model.domain_size[0],
                                               model.origin[1] + model.domain_size[1], model.origin[1]]}
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15, 15))

slices = [slice(model.nbl, -model.nbl), slice(model.nbl, -model.nbl)]

img1 = ax[0].imshow(np.transpose(model.lam.data[slices]), vmin=1.5**2, vmax=4.0**2, **plt_options_model)
fig.colorbar(img1, ax=ax[0])
ax[0].set_title(r"First Lam\'e parameter $\lambda$", fontsize=20)
ax[0].set_xlabel('X (m)', fontsize=20)
ax[0].set_ylabel('Depth (m)', fontsize=20)
ax[0].set_aspect('auto')


img2 = ax[1].imshow(np.transpose(model.mu.data[slices]), vmin=0, vmax=15, **plt_options_model)
fig.colorbar(img2, ax=ax[1])
ax[1].set_title(r"Shear modulus $\mu$", fontsize=20)
ax[1].set_xlabel('X (m)', fontsize=20)
ax[1].set_ylabel('Depth (m)', fontsize=20)
ax[1].set_aspect('auto')


img3 = ax[2].imshow(1/np.transpose(model.irho.data[slices]), vmin=1.0, vmax=3.0, **plt_options_model)
fig.colorbar(img3, ax=ax[2])
ax[2].set_title(r"Density $\rho$", fontsize=20)
ax[2].set_xlabel('X (m)', fontsize=20)
ax[2].set_ylabel('Depth (m)', fontsize=20)
ax[2].set_aspect('auto')


plt.tight_layout()
```


![png](05_elastic_varying_parameters_files/05_elastic_varying_parameters_3_0.png)



```python
# Timestep size from Eq. 7 with V_p=6000. and dx=100
t0, tn = 0., 2000.
dt = model.critical_dt
time_range = TimeAxis(start=t0, stop=tn, step=dt)

src = RickerSource(name='src', grid=model.grid, f0=0.015, time_range=time_range)
src.coordinates.data[:] = [1500., 10.]
```


```python
#NBVAL_SKIP

src.show()
```


![png](05_elastic_varying_parameters_files/05_elastic_varying_parameters_5_0.png)


# Vectorial form

While conventional litterature writes the elastic wave-equation as a set of scalar PDEs, the higher level representation comes from Hooke's law and the equation of motion and writes as:

\begin{cases}
&\frac{dv}{dt} = \nabla . \tau \\
&\frac{d \tau}{dt} = \lambda tr(\nabla v) \mathbf{I}  + \mu (\nabla v + (\nabla v)^T)
\end{cases}

and as $tr(\nabla v)$ is the divergence of $v$ we can reqrite it as

\begin{cases}
&\frac{dv}{dt} = \nabla . \tau \\
&\frac{d \tau}{dt} = \lambda \text{diag}(\nabla . v) + \mu (\nabla v + (\nabla v)^T)
\end{cases}

where $v$ is a vector valued function:

$v(t, x, y) = (v_x(t, x, y), v_y(t, x, y)$

and the stress $\tau$ is a symmetric tensor valued function:


$\tau(t, x, y) = \begin{bmatrix}\tau_{xx}(t, x, y) & \tau_{xy}(t, x, y)\\\tau_{xy}t, x, y) & \tau_{yy}(t, x, y)\end{bmatrix}$


We show in the following how to setup the elastic wave-equation form Devito's high-level tensorial types.


```python
# Now we create the velocity and pressure fields
so = 8

x, z = model.grid.dimensions
t = model.grid.stepping_dim
time = model.grid.time_dim
s = time.spacing

v = VectorTimeFunction(name='v', grid=model.grid, space_order=so, time_order=1)
tau = TensorTimeFunction(name='t', grid=model.grid, space_order=so, time_order=1)
```


```python
# The source injection term
src_xx = src.inject(field=tau.forward[0, 0], expr=s*src)
src_zz = src.inject(field=tau.forward[1, 1], expr=s*src)

# The receiver
nrec = 301
rec = Receiver(name="rec", grid=model.grid, npoint=nrec, time_range=time_range)
rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
rec.coordinates.data[:, -1] = 5.

rec2 = Receiver(name="rec2", grid=model.grid, npoint=nrec, time_range=time_range)
rec2.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
rec2.coordinates.data[:, -1] = 3000.0/nlayers

rec3 = Receiver(name="rec3", grid=model.grid, npoint=nrec, time_range=time_range)
rec3.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
rec3.coordinates.data[:, -1] = 3000.0/nlayers

rec_term = rec.interpolate(expr=tau[0, 0] + tau[1, 1])
rec_term += rec2.interpolate(expr=v[1])
rec_term += rec3.interpolate(expr=v[0])
```


```python
#NBVAL_SKIP
from seismic import plot_velocity
plot_velocity(model, source=src.coordinates.data,
              receiver=rec.coordinates.data[::10, :])
plot_velocity(model, source=src.coordinates.data,
              receiver=rec2.coordinates.data[::10, :])
```


![png](05_elastic_varying_parameters_files/05_elastic_varying_parameters_9_0.png)



![png](05_elastic_varying_parameters_files/05_elastic_varying_parameters_9_1.png)



```python
# Now let's try and create the staggered updates
# Lame parameters
l, mu, ro = model.lam, model.mu, model.irho

# fdelmodc reference implementation
u_v = Eq(v.forward, model.damp * (v + s*ro*div(tau)))
u_t = Eq(tau.forward,  model.damp *  (tau + s * (l * diag(div(v.forward)) +
                                                mu * (grad(v.forward) + grad(v.forward).T))))

op = Operator([u_v] + [u_t] + src_xx + src_zz + rec_term)
```


```python
v._time_order
```




$\displaystyle 1$




```python
ro._eval_at(v[0]).evaluate
```




$\displaystyle 0.5 \operatorname{irho}{\left(x,y \right)} + 0.5 \operatorname{irho}{\left(x + h_{x},y \right)}$



We can now see that both the particle velocities and stress equations are vectorial and tensorial equations. Devito takes care of the discretization and staggered grids automatically for these types of object.


```python
u_v
```




$\displaystyle \left[\begin{matrix}\operatorname{v_{x}}{\left(t + dt,x + \frac{h_{x}}{2},y \right)}\\\operatorname{v_{y}}{\left(t + dt,x,y + \frac{h_{y}}{2} \right)}\end{matrix}\right] = \left[\begin{matrix}\left(dt \left(\frac{\partial}{\partial x} \operatorname{t_{xx}}{\left(t,x,y \right)} + \frac{\partial}{\partial y} \operatorname{t_{xy}}{\left(t,x + \frac{h_{x}}{2},y + \frac{h_{y}}{2} \right)}\right) \operatorname{irho}{\left(x,y \right)} + \operatorname{v_{x}}{\left(t,x + \frac{h_{x}}{2},y \right)}\right) \operatorname{damp}{\left(x,y \right)}\\\left(dt \left(\frac{\partial}{\partial x} \operatorname{t_{xy}}{\left(t,x + \frac{h_{x}}{2},y + \frac{h_{y}}{2} \right)} + \frac{\partial}{\partial y} \operatorname{t_{yy}}{\left(t,x,y \right)}\right) \operatorname{irho}{\left(x,y \right)} + \operatorname{v_{y}}{\left(t,x,y + \frac{h_{y}}{2} \right)}\right) \operatorname{damp}{\left(x,y \right)}\end{matrix}\right]$




```python
u_t
```




$\displaystyle \left[\begin{matrix}\operatorname{t_{xx}}{\left(t + dt,x,y \right)} & \operatorname{t_{xy}}{\left(t + dt,x + \frac{h_{x}}{2},y + \frac{h_{y}}{2} \right)}\\\operatorname{t_{xy}}{\left(t + dt,x + \frac{h_{x}}{2},y + \frac{h_{y}}{2} \right)} & \operatorname{t_{yy}}{\left(t + dt,x,y \right)}\end{matrix}\right] = \left[\begin{matrix}\left(dt \left(\left(\frac{\partial}{\partial x} \operatorname{v_{x}}{\left(t + dt,x + \frac{h_{x}}{2},y \right)} + \frac{\partial}{\partial y} \operatorname{v_{y}}{\left(t + dt,x,y + \frac{h_{y}}{2} \right)}\right) \operatorname{lam}{\left(x,y \right)} + 2 \mu{\left(x,y \right)} \frac{\partial}{\partial x} \operatorname{v_{x}}{\left(t + dt,x + \frac{h_{x}}{2},y \right)}\right) + \operatorname{t_{xx}}{\left(t,x,y \right)}\right) \operatorname{damp}{\left(x,y \right)} & \left(dt \left(\frac{\partial}{\partial y} \operatorname{v_{x}}{\left(t + dt,x + \frac{h_{x}}{2},y \right)} + \frac{\partial}{\partial x} \operatorname{v_{y}}{\left(t + dt,x,y + \frac{h_{y}}{2} \right)}\right) \mu{\left(x,y \right)} + \operatorname{t_{xy}}{\left(t,x + \frac{h_{x}}{2},y + \frac{h_{y}}{2} \right)}\right) \operatorname{damp}{\left(x,y \right)}\\\left(dt \left(\frac{\partial}{\partial y} \operatorname{v_{x}}{\left(t + dt,x + \frac{h_{x}}{2},y \right)} + \frac{\partial}{\partial x} \operatorname{v_{y}}{\left(t + dt,x,y + \frac{h_{y}}{2} \right)}\right) \mu{\left(x,y \right)} + \operatorname{t_{xy}}{\left(t,x + \frac{h_{x}}{2},y + \frac{h_{y}}{2} \right)}\right) \operatorname{damp}{\left(x,y \right)} & \left(dt \left(\left(\frac{\partial}{\partial x} \operatorname{v_{x}}{\left(t + dt,x + \frac{h_{x}}{2},y \right)} + \frac{\partial}{\partial y} \operatorname{v_{y}}{\left(t + dt,x,y + \frac{h_{y}}{2} \right)}\right) \operatorname{lam}{\left(x,y \right)} + 2 \mu{\left(x,y \right)} \frac{\partial}{\partial y} \operatorname{v_{y}}{\left(t + dt,x,y + \frac{h_{y}}{2} \right)}\right) + \operatorname{t_{yy}}{\left(t,x,y \right)}\right) \operatorname{damp}{\left(x,y \right)}\end{matrix}\right]$




```python
#NBVAL_IGNORE_OUTPUT
# Partial ru for 1.2sec to plot the wavefield
op(dt=model.critical_dt, time_M=int(1000/model.critical_dt))
```

    Operator `Kernel` run in 0.14 s



```python
#NBVAL_SKIP
scale = .5*1e-3

plt_options_model = {'extent': [model.origin[0] , model.origin[0] + model.domain_size[0],
                                model.origin[1] + model.domain_size[1], model.origin[1]]}


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

ax[0, 0].imshow(np.transpose(v[0].data[0][slices]), vmin=-scale, vmax=scale, cmap="RdGy", **plt_options_model)
ax[0, 0].imshow(np.transpose(model.lam.data[slices]), vmin=2.5, vmax=15.0, cmap="jet", alpha=.5, **plt_options_model)
ax[0, 0].set_aspect('auto')
ax[0, 0].set_xlabel('X (m)', fontsize=20)
ax[0, 0].set_ylabel('Depth (m)', fontsize=20)
ax[0, 0].set_title(r"$v_{x}$", fontsize=20)

ax[0, 1].imshow(np.transpose(v[1].data[0][slices]), vmin=-scale, vmax=scale, cmap="RdGy", **plt_options_model)
ax[0, 1].imshow(np.transpose(model.lam.data[slices]), vmin=2.5, vmax=15.0, cmap="jet", alpha=.5, **plt_options_model)
ax[0, 1].set_aspect('auto')
ax[0, 1].set_title(r"$v_{z}$", fontsize=20)
ax[0, 1].set_xlabel('X (m)', fontsize=20)
ax[0, 1].set_ylabel('Depth (m)', fontsize=20)

ax[1, 0].imshow(np.transpose(tau[0,0].data[0][slices]+tau[1,1].data[0][slices]),
             vmin=-10*scale, vmax=10*scale, cmap="RdGy", **plt_options_model)
ax[1, 0].imshow(np.transpose(model.lam.data[slices]), vmin=2.5, vmax=15.0, cmap="jet",
               alpha=.5, **plt_options_model)
ax[1, 0].set_aspect('auto')
ax[1, 0].set_title(r"$\tau_{xx} + \tau_{zz}$", fontsize=20)
ax[1, 0].set_xlabel('X (m)', fontsize=20)
ax[1, 0].set_ylabel('Depth (m)', fontsize=20)


ax[1, 1].imshow(np.transpose(tau[0,1].data[0][slices]), vmin=-scale, vmax=scale, cmap="RdGy", **plt_options_model)
ax[1, 1].imshow(np.transpose(model.lam.data[slices]), vmin=2.5, vmax=15.0, cmap="jet", alpha=.5, **plt_options_model)
ax[1, 1].set_aspect('auto')
ax[1, 1].set_title(r"$\tau_{xy}$", fontsize=20)
ax[1, 1].set_xlabel('X (m)', fontsize=20)
ax[1, 1].set_ylabel('Depth (m)', fontsize=20)


plt.tight_layout()
```


![png](05_elastic_varying_parameters_files/05_elastic_varying_parameters_17_0.png)



```python
#NBVAL_IGNORE_OUTPUT
# Full run for the data
op(dt=model.critical_dt, time_m=int(1000/model.critical_dt))
```

    Operator `Kernel` run in 0.14 s



```python
# Data on a standard 2ms tim axis
rec_plot = rec.resample(num=1001)
rec2_plot = rec2.resample(num=1001)
rec3_plot = rec3.resample(num=1001)
```


```python
scale_for_plot = np.diag(np.linspace(1.0, 2.5, 1001)**2.0)
```


```python
#NBVAL_SKIP
# Pressure (txx + tzz) data at sea surface
extent = [rec_plot.coordinates.data[0, 0], rec_plot.coordinates.data[-1, 0], 1e-3*tn, t0]
aspect = rec_plot.coordinates.data[-1, 0]/(1e-3*tn)/.5

plt.figure(figsize=(15, 15))
plt.imshow(np.dot(scale_for_plot, rec_plot.data), vmin=-.01, vmax=.01, cmap="seismic",
           interpolation='lanczos', extent=extent, aspect=aspect)
plt.ylabel("Time (s)", fontsize=20)
plt.xlabel("Receiver position (m)", fontsize=20)

```




    Text(0.5, 0, 'Receiver position (m)')




![png](05_elastic_varying_parameters_files/05_elastic_varying_parameters_21_1.png)



```python
#NBVAL_SKIP
# OBC data of vx/vz
plt.figure(figsize=(15, 15))
plt.subplot(121)
plt.imshow(rec2_plot.data, vmin=-1e-3, vmax=1e-3, cmap="seismic",
           interpolation='lanczos', extent=extent, aspect=aspect)
plt.ylabel("Time (s)", fontsize=20)
plt.xlabel("Receiver position (m)", fontsize=20)
plt.subplot(122)
plt.imshow(rec3_plot.data, vmin=-1e-3, vmax=1e-3, cmap="seismic",
           interpolation='lanczos', extent=extent, aspect=aspect)
plt.ylabel("Time (s)", fontsize=20)
plt.xlabel("Receiver position (m)", fontsize=20)
```




    Text(0.5, 0, 'Receiver position (m)')




![png](05_elastic_varying_parameters_files/05_elastic_varying_parameters_22_1.png)



```python
# Now that looks pretty! But let's do it again with a 2nd order in time
so = 8
v2 = VectorTimeFunction(name='v2', grid=model.grid, space_order=so, time_order=2)
tau0 = TensorFunction(name='t0', grid=model.grid, space_order=so)
# The source injection term
src_xx = src.inject(field=tau0[0, 0], expr=src.dt)
src_zz = src.inject(field=tau0[1, 1], expr=src.dt)

s = model.grid.time_dim.spacing
# fdelmodc reference implementation
u_v = Eq(v2.forward, model.damp * (2 * v2 - model.damp * v2.backward + s**2*ro*div(tau0)))
u_t = Eq(tau0, model.damp * (l * diag(div(v2.forward)) + mu * (grad(v2.forward) + grad(v2.forward).T)))


# rec_term = rec.interpolate(expr=tau[0, 0] + tau[1, 1])
rec_term = rec2.interpolate(expr=v2[0])
rec_term += rec3.interpolate(expr=v2[1])
op = Operator([u_v] + [u_t] + src_xx + src_zz + rec_term)

```


```python
#NBVAL_IGNORE_OUTPUT
# Partial ru for 1.2sec to plot the wavefield
op(dt=model.critical_dt, time_M=int(1000/model.critical_dt))
```

    Operator `Kernel` run in 0.11 s



```python
#NBVAL_SKIP
scale = 1e-4

plt_options_model = {'extent': [model.origin[0] , model.origin[0] + model.domain_size[0],
                                model.origin[1] + model.domain_size[1], model.origin[1]]}


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

ax[0, 0].imshow(np.transpose(v2[0].data[0][slices]), vmin=-scale, vmax=scale, cmap="RdGy", **plt_options_model)
ax[0, 0].imshow(np.transpose(model.lam.data[slices]), vmin=2.5, vmax=15.0, cmap="jet", alpha=.5, **plt_options_model)
ax[0, 0].set_aspect('auto')
ax[0, 0].set_xlabel('X (m)', fontsize=20)
ax[0, 0].set_ylabel('Depth (m)', fontsize=20)
ax[0, 0].set_title(r"$v_{x}$", fontsize=20)

ax[0, 1].imshow(np.transpose(v2[1].data[0][slices]), vmin=-scale, vmax=scale, cmap="RdGy", **plt_options_model)
ax[0, 1].imshow(np.transpose(model.lam.data[slices]), vmin=2.5, vmax=15.0, cmap="jet", alpha=.5, **plt_options_model)
ax[0, 1].set_aspect('auto')
ax[0, 1].set_title(r"$v_{z}$", fontsize=20)
ax[0, 1].set_xlabel('X (m)', fontsize=20)
ax[0, 1].set_ylabel('Depth (m)', fontsize=20)

ax[1, 0].imshow(np.transpose(tau0[0,0].data[slices]+tau0[1,1].data[slices]),
             vmin=-10*scale, vmax=10*scale, cmap="RdGy", **plt_options_model)
ax[1, 0].imshow(np.transpose(model.lam.data[slices]), vmin=2.5, vmax=15.0, cmap="jet",
               alpha=.5, **plt_options_model)
ax[1, 0].set_aspect('auto')
ax[1, 0].set_title(r"$\tau_{xx} + \tau_{zz}$", fontsize=20)
ax[1, 0].set_xlabel('X (m)', fontsize=20)
ax[1, 0].set_ylabel('Depth (m)', fontsize=20)


ax[1, 1].imshow(np.transpose(tau0[0,1].data[slices]), vmin=-scale, vmax=scale, cmap="RdGy", **plt_options_model)
ax[1, 1].imshow(np.transpose(model.lam.data[slices]), vmin=2.5, vmax=15.0, cmap="jet", alpha=.5, **plt_options_model)
ax[1, 1].set_aspect('auto')
ax[1, 1].set_title(r"$\tau_{xy}$", fontsize=20)
ax[1, 1].set_xlabel('X (m)', fontsize=20)
ax[1, 1].set_ylabel('Depth (m)', fontsize=20)


plt.tight_layout()
```


![png](05_elastic_varying_parameters_files/05_elastic_varying_parameters_25_0.png)



```python
#NBVAL_IGNORE_OUTPUT
op(dt=model.critical_dt, time_m=int(1000/model.critical_dt))
```

    Operator `Kernel` run in 0.13 s



```python
rec2_plot2 = rec2.resample(num=1001)
rec3_plot2 = rec3.resample(num=1001)
```


```python
#NBVAL_SKIP
# OBC data of vx/vz
plt.figure(figsize=(15, 15))
plt.subplot(121)
plt.imshow(rec2_plot2.data, vmin=-1e-3, vmax=1e-3, cmap="seismic",
           interpolation='lanczos', extent=extent, aspect=aspect)
plt.ylabel("Time (s)", fontsize=20)
plt.xlabel("Receiver position (m)", fontsize=20)
plt.subplot(122)
plt.imshow(rec3_plot2.data, vmin=-1e-3, vmax=1e-3, cmap="seismic",
           interpolation='lanczos', extent=extent, aspect=aspect)
plt.ylabel("Time (s)", fontsize=20)
plt.xlabel("Receiver position (m)", fontsize=20)
```




    Text(0.5, 0, 'Receiver position (m)')




![png](05_elastic_varying_parameters_files/05_elastic_varying_parameters_28_1.png)

