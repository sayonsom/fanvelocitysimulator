from dolfin import *
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm
import io

def simulate_airflow(fan_cmm, fan_rpm, fan_diameter, fan_position, ac_position, room_dimensions, time_duration):
    # Unpack parameters
    fan_x, fan_y, fan_z = fan_position
    ac_x, ac_y, ac_z = ac_position
    L, W, H = room_dimensions

    # Create a 3D mesh with reduced resolution to avoid memory issues
    mesh = BoxMesh(Point(0, 0, 0), Point(L, W, H), 30, 15, 15)

    # Define function spaces for velocity and pressure
    V = VectorFunctionSpace(mesh, 'P', 2)  # Velocity space in 3D
    Q = FunctionSpace(mesh, 'P', 1)        # Pressure space

    # Define test and trial functions
    u = Function(V)    # Velocity field
    u0 = Function(V)   # Previous velocity field (initialized to zero)
    v = TestFunction(V)  # Test function for velocity

    # Define boundary conditions in 3D
    class InletAC(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], ac_x) and (ac_y - 0.5 <= x[1] <= ac_y + 0.5) and (ac_z - 1 <= x[2] <= ac_z + 1)

    class Walls(SubDomain):
        def inside(self, x, on_boundary):
            return (near(x[0], L) or near(x[1], W) or near(x[2], H) or near(x[2], 0))

    # Apply boundary conditions
    ac_bc = DirichletBC(V, Expression(("1.0", "0.1*sin(5.0*x[1])", "0.0"), degree=2), InletAC())
    noslip_bc = DirichletBC(V, Constant((0, 0, 0)), Walls())
    boundary_conditions = [ac_bc, noslip_bc]

    # Fan modeled as a radial airflow source term
    fan_area = np.pi * (fan_diameter / 2) ** 2
    fan_velocity = fan_cmm / (fan_area * 60)

    class FanSource(UserExpression):
        def eval(self, values, x):
            r = np.sqrt((x[0] - fan_x)**2 + (x[1] - fan_y)**2 + (x[2] - fan_z)**2)
            if r > 0:
                values[0] = fan_velocity * (x[0] - fan_x) / r
                values[1] = fan_velocity * (x[1] - fan_y) / r
                values[2] = fan_velocity * (x[2] - fan_z) / r
            else:
                values[0] = values[1] = values[2] = 0.0

        def value_shape(self):
            return (3,)

    fan_force = FanSource(degree=2)

    # Define the Navier-Stokes equations
    rho = 1.0  # Density of air
    mu = 0.001  # Dynamic viscosity of air

    # Time-stepping parameters
    dt = 0.01
    t = 0.0

    # Define velocity residual form
    F_velocity = (
        rho * dot((u - u0) / dt, v) * dx +
        rho * dot(dot(u0, nabla_grad(u0)), v) * dx +
        mu * inner(grad(u), grad(v)) * dx -
        dot(fan_force, v) * dx
    )

    # Compute Jacobian for velocity
    J_velocity = derivative(F_velocity, u)

    # Create a nonlinear variational problem and solver for velocity
    velocity_problem = NonlinearVariationalProblem(F_velocity, u, boundary_conditions, J_velocity)
    velocity_solver = NonlinearVariationalSolver(velocity_problem)

    # Switch to PETSc iterative solver
    solver_parameters = {
        'nonlinear_solver': 'newton',
        'newton_solver': {
            'linear_solver': 'gmres',
            'preconditioner': 'ilu',
            'absolute_tolerance': 1E-8,
            'relative_tolerance': 1E-7,
            'maximum_iterations': 1000
        }
    }
    velocity_solver.parameters.update(solver_parameters)

    # Time-stepping
    n_steps = int(time_duration / dt)
    # stdout.write(f"Starting airflow simulation...\n")
    
    with tqdm(total=n_steps, desc="Simulating airflow", unit="step") as pbar:
        while t < time_duration:
            t += dt
            velocity_solver.solve()
            u0.assign(u)
            pbar.update(1)
            # stdout.write(f"Time: {t:.2f}s\n")

    # Extract mesh coordinates and velocity values
    coords = mesh.coordinates()
    u_values = u.compute_vertex_values(mesh)

    # Extract x, y, and z velocity components
    u_x = u_values[0::3]
    u_y = u_values[1::3]
    u_z = u_values[2::3]

    # Create the 3D grid for visualization
    x_grid = np.linspace(0, L, 30)
    y_grid = np.linspace(0, W, 15)
    z_grid = np.linspace(0, H, 15)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid)

    # Interpolate velocity data to the grid for visualization
    U = griddata(coords, u_x, (X, Y, Z), method='linear')
    V = griddata(coords, u_y, (X, Y, Z), method='linear')
    W = griddata(coords, u_z, (X, Y, Z), method='linear')

    # stdout.write("Simulation complete!\n")

    return X, Y, Z, U, V, W

# Example usage:
if __name__ == "__main__":
    # Set default parameters
    fan_cmm = 100.0
    fan_rpm = 3700
    fan_diameter = 1.2
    fan_position = (5.0, 2.5, 1.5)
    ac_position = (0.0, 1.5, 1.0)
    room_dimensions = (10.0, 5.0, 3.0)
    time_duration = 0.03

    # Run the simulation
    X, Y, Z, U, V, W = simulate_airflow(
        fan_cmm=fan_cmm,
        fan_rpm=fan_rpm,
        fan_diameter=fan_diameter,
        fan_position=fan_position,
        ac_position=ac_position,
        room_dimensions=room_dimensions,
        time_duration=time_duration
    )

    # Visualization code 
    if 1 == 0:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(X, Y, Z, U, V, W, length=0.1, normalize=True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

