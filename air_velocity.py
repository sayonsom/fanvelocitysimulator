from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 2.0            # final time
num_steps = 50     # number of time steps
dt = T / num_steps # time step size
mu = 0.001         # dynamic viscosity
rho = 1            # density

# Domain: A rectangular room with a circular fan
room = Rectangle(Point(0, 0), Point(5, 5))
fan = Circle(Point(2.5, 2.5), 0.5)
domain = room - fan
mesh = generate_mesh(domain, 64)

# Function spaces (P2-P1)
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define fan rotation
fan_center = Constant((2.5, 2.5))
fan_strength = Constant(10.0)

def fan_velocity(x):
    r = x - fan_center
    theta = atan2(r[1], r[0])
    v_theta = fan_strength * exp(-4*sqrt(dot(r, r)))
    return as_vector((-r[1]*v_theta, r[0]*v_theta))

# Boundary conditions
walls = 'on_boundary && !(near(x[0], 0) || near(x[0], 5))'
inflow = 'near(x[0], 0)'
outflow = 'near(x[0], 5)'

bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
bcu_inflow = DirichletBC(V, Constant((0.1, 0)), inflow)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)

bcu = [bcu_walls, bcu_inflow]
bcp = [bcp_outflow]

# Define trial and test functions
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

# Define functions for velocity and pressure
u_n = Function(V)
u_ = Function(V)
p_n = Function(Q)
p_ = Function(Q)

# Define strain-rate tensor
def epsilon(u):
    return sym(nabla_grad(u))

def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# Variational forms
F1 = (rho*dot((u - u_n) / dt, v)*dx
      + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx
      + inner(sigma(u_, p_n), epsilon(v))*dx
      - dot(fan_velocity(x), v)*dx)  # Add fan force
F2 = dot(nabla_grad(p - p_n), q)*dx + dot(div(u_), q)*dx

# Time-stepping
t = 0
for n in range(num_steps):
    t += dt
    
    # Solve the variational problem for velocity
    solve(lhs(F1) == rhs(F1), u_, bcu)
    
    # Solve the variational problem for pressure
    solve(lhs(F2) == rhs(F2), p_, bcp)
    
    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)
    
    # Output results (assuming vtkfile_u and vtkfile_p are defined)
    # vtkfile_u << u_
    # vtkfile_p << p_

# Plotting (make sure to have matplotlib installed)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plot(u_)
plt.title("Velocity field at t = {}".format(t))
plt.colorbar()
plt.savefig("velocity_field.png")
plt.close()

plt.figure(figsize=(10, 5))
plot(p_)
plt.title("Pressure field at t = {}".format(t))
plt.colorbar()
plt.savefig("pressure_field.png")
plt.close()

print("Simulation completed. Check velocity_field.png and pressure_field.png for results.")