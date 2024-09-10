import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants for unit conversion
FEET_TO_METERS = 0.3048
MM_TO_METERS = 0.001
CMM_TO_M3S = 1 / 60000  # Convert CMM (cubic meters per minute) to m³/s

def simulate_fan_3d(sweep_mm, rpm, air_delivery_cmm, room_size_feet, fan_position_feet, grid_size=30, time_steps=50):
    # Convert units
    room_size = tuple(dim * FEET_TO_METERS for dim in room_size_feet)
    fan_position = tuple(pos * FEET_TO_METERS for pos in fan_position_feet)
    diameter = sweep_mm * MM_TO_METERS
    air_delivery = air_delivery_cmm * CMM_TO_M3S

    # Constants
    air_viscosity = 1.81e-5  # kg/(m·s)
    air_density = 1.225  # kg/m³
    
    # Calculate angular velocity
    omega = rpm * 2 * np.pi / 60  # rad/s
    
    # Create 3D grid based on room size
    x = np.linspace(0, room_size[0], grid_size)
    y = np.linspace(0, room_size[1], grid_size)
    z = np.linspace(0, room_size[2], grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Calculate distance from fan center
    X_fan = X - fan_position[0]
    Y_fan = Y - fan_position[1]
    Z_fan = Z - fan_position[2]
    R = np.sqrt(X_fan**2 + Y_fan**2)
    
    # Initialize velocity components
    U = np.zeros((grid_size, grid_size, grid_size))
    V = np.zeros((grid_size, grid_size, grid_size))
    W = np.zeros((grid_size, grid_size, grid_size))
    
    # Time array
    t = np.linspace(0, 5, time_steps)
    
    # Simplified Navier-Stokes solver
    def simplified_navier_stokes(y, t):
        u = y[:grid_size**3].reshape((grid_size, grid_size, grid_size))
        v = y[grid_size**3:2*grid_size**3].reshape((grid_size, grid_size, grid_size))
        w = y[2*grid_size**3:].reshape((grid_size, grid_size, grid_size))
        
        # Fan force (simplified 3D model)
        fan_thickness = 0.1 * diameter
        fan_mask = (R <= diameter/2) & (np.abs(Z_fan) <= fan_thickness/2)
        fan_force_u = np.where(fan_mask, -omega * Y_fan, 0)
        fan_force_v = np.where(fan_mask, omega * X_fan, 0)
        fan_force_w = np.where(fan_mask, air_delivery / (np.pi * (diameter/2)**2), 0)  # Axial force based on air delivery
        
        # Simplified Navier-Stokes equations (only considering fan force and diffusion)
        du_dt = air_viscosity/air_density * (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) + 
                                             np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) + 
                                             np.roll(u, 1, axis=2) + np.roll(u, -1, axis=2) - 6*u) + fan_force_u
        dv_dt = air_viscosity/air_density * (np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) + 
                                             np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) + 
                                             np.roll(v, 1, axis=2) + np.roll(v, -1, axis=2) - 6*v) + fan_force_v
        dw_dt = air_viscosity/air_density * (np.roll(w, 1, axis=0) + np.roll(w, -1, axis=0) + 
                                             np.roll(w, 1, axis=1) + np.roll(w, -1, axis=1) + 
                                             np.roll(w, 1, axis=2) + np.roll(w, -1, axis=2) - 6*w) + fan_force_w
        
        # Apply boundary conditions (no-slip at walls)
        du_dt[0, :, :] = du_dt[-1, :, :] = du_dt[:, 0, :] = du_dt[:, -1, :] = du_dt[:, :, 0] = du_dt[:, :, -1] = 0
        dv_dt[0, :, :] = dv_dt[-1, :, :] = dv_dt[:, 0, :] = dv_dt[:, -1, :] = dv_dt[:, :, 0] = dv_dt[:, :, -1] = 0
        dw_dt[0, :, :] = dw_dt[-1, :, :] = dw_dt[:, 0, :] = dw_dt[:, -1, :] = dw_dt[:, :, 0] = dw_dt[:, :, -1] = 0
        
        return np.concatenate([du_dt.flatten(), dv_dt.flatten(), dw_dt.flatten()])
    
    # Initial conditions
    y0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Solve ODE
    solution = odeint(simplified_navier_stokes, y0, t)
    
    # Extract final velocity field
    U = solution[-1, :grid_size**3].reshape((grid_size, grid_size, grid_size))
    V = solution[-1, grid_size**3:2*grid_size**3].reshape((grid_size, grid_size, grid_size))
    W = solution[-1, 2*grid_size**3:].reshape((grid_size, grid_size, grid_size))
    
    # Calculate total velocity
    V_total = np.sqrt(U**2 + V**2 + W**2)
    
    return X, Y, Z, U, V, W, V_total

st.set_page_config(layout="wide")

st.title("3D Ceiling Fan CFD Simulation")

st.sidebar.header("Room Parameters")
room_length = st.sidebar.slider("Room Length (feet)", 10, 50, 20)
room_width = st.sidebar.slider("Room Width (feet)", 10, 50, 15)
room_height = st.sidebar.slider("Room Height (feet)", 8, 20, 10)

st.sidebar.header("Fan Parameters")
fan_models = {
    "Quasar": {"sweep": 1200, "air_delivery": 270, "rpm": 405},
    "Speedo 24": {"sweep": 900, "air_delivery": 270, "rpm": 600},
    "Speedo Deco": {"sweep": 900, "air_delivery": 270, "rpm": 600},
    "Neutron 24": {"sweep": 900, "air_delivery": 270, "rpm": 600},
    "Neo Deco 24": {"sweep": 900, "air_delivery": 270, "rpm": 600},
    "Neo Gold 24": {"sweep": 900, "air_delivery": 270, "rpm": 600}
}

selected_fan = st.sidebar.selectbox("Select Fan Model", list(fan_models.keys()))
sweep = fan_models[selected_fan]["sweep"]
air_delivery = fan_models[selected_fan]["air_delivery"]
rpm = fan_models[selected_fan]["rpm"]

st.sidebar.write(f"Sweep: {sweep} mm")
st.sidebar.write(f"Air Delivery: {air_delivery} CMM")
st.sidebar.write(f"RPM: {rpm}")

st.sidebar.header("Fan Position")
fan_x = st.sidebar.slider("Fan X Position (feet)", 0.0, float(room_length), float(room_length)/2, 0.1)
fan_y = st.sidebar.slider("Fan Y Position (feet)", 0.0, float(room_width), float(room_width)/2, 0.1)
fan_z = st.sidebar.slider("Fan Z Position (feet)", 0.0, float(room_height), float(room_height) - 1.0, 0.1)



grid_size = st.sidebar.slider("Grid Size", 10, 50, 20, 5)
time_steps = st.sidebar.slider("Time Steps", 10, 100, 50, 10)

if st.sidebar.button("Run Simulation"):
    st.write("Running simulation...")
    progress_bar = st.progress(0)

    # Run the simulation
    room_size_feet = (room_length, room_width, room_height)
    fan_position_feet = (fan_x, fan_y, fan_z)
    X, Y, Z, U, V, W, V_total = simulate_fan_3d(sweep, rpm, air_delivery, room_size_feet, fan_position_feet, grid_size, time_steps)

    progress_bar.progress(100)
    st.write("Simulation complete!")

    # Create plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Velocity magnitude slice (XY plane)
    slice_z = grid_size // 2
    c1 = ax1.imshow(V_total[:, :, slice_z].T, origin='lower', extent=[0, room_length, 0, room_width], 
                    aspect='auto', cmap='viridis')
    plt.colorbar(c1, ax=ax1, label='Velocity (ft/s)')
    ax1.set_title(f'Velocity Magnitude (XY plane at Z={room_height/2:.1f}ft)\n{selected_fan}')
    ax1.set_xlabel('X (feet)')
    ax1.set_ylabel('Y (feet)')
    ax1.plot(fan_x, fan_y, 'ro', markersize=10)  # Mark fan position

    # Velocity vectors (XY plane)
    step = max(1, grid_size // 20)  # Adjust arrow density
    ax2.quiver(X[::step, ::step, slice_z] / FEET_TO_METERS, Y[::step, ::step, slice_z] / FEET_TO_METERS,
               U[::step, ::step, slice_z], V[::step, ::step, slice_z], 
               scale=50, width=0.002)
    ax2.set_title(f'Velocity Vectors (XY plane at Z={room_height/2:.1f}ft)\n{selected_fan}')
    ax2.set_xlabel('X (feet)')
    ax2.set_ylabel('Y (feet)')
    ax2.set_xlim(0, room_length)
    ax2.set_ylim(0, room_width)
    ax2.plot(fan_x, fan_y, 'ro', markersize=10)  # Mark fan position

    # Streamlines (XY plane)
    x = np.linspace(0, room_length, grid_size)
    y = np.linspace(0, room_width, grid_size)
    X2, Y2 = np.meshgrid(x, y)
    U2 = U[:, :, slice_z].T
    V2 = V[:, :, slice_z].T
    speed = np.sqrt(U2**2 + V2**2)
    lw = 2 * speed / speed.max()
    strm = ax3.streamplot(X2, Y2, U2, V2, density=1, color=speed, cmap='viridis', linewidth=lw)
    plt.colorbar(strm.lines, ax=ax3, label='Velocity (ft/s)')
    ax3.set_title(f'Streamlines (XY plane at Z={room_height/2:.1f}ft)\n{selected_fan}')
    ax3.set_xlabel('X (feet)')
    ax3.set_ylabel('Y (feet)')
    ax3.set_xlim(0, room_length)
    ax3.set_ylim(0, room_width)
    ax3.plot(fan_x, fan_y, 'ro', markersize=10)  # Mark fan position

    plt.tight_layout()
    st.pyplot(fig)

    # Display some statistics
    st.subheader("Simulation Statistics")
    max_velocity_fts = np.max(V_total) / FEET_TO_METERS
    avg_velocity_fts = np.mean(V_total) / FEET_TO_METERS
    min_velocity_fts = np.min(V_total) / FEET_TO_METERS
    st.write(f"Maximum velocity: {max_velocity_fts:.2f} ft/s")
    st.write(f"Average velocity: {avg_velocity_fts:.2f} ft/s")
    st.write(f"Minimum velocity: {min_velocity_fts:.2f} ft/s")

st.sidebar.markdown("---")
st.sidebar.write("Note: Larger grid sizes and more time steps will result in more accurate simulations but will take longer to compute.")