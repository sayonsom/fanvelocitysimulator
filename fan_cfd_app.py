import streamlit as st
import plotly.graph_objects as go
from cfd_simulation import simulate_airflow
import numpy as np

# Add a title to the app
st.title("AC-Fan Digital Twin Simulator")

# Sidebar inputs for user settings
st.sidebar.title("Fan and AC Settings")

# Fan settings with better defaults
fan_cmm = st.sidebar.number_input("Fan CMM (Cubic Meters per Minute)", min_value=10.0, value=150.0, step=10.0)
fan_rpm = st.sidebar.number_input("Fan RPM (Rotations per Minute)", min_value=100, value=1000, step=10)
fan_diameter = st.sidebar.number_input("Fan Diameter (meters)", min_value=0.1, value=0.7, step=0.1)
fan_x = st.sidebar.number_input("Fan X Position (meters)", min_value=0.0, value=4.0, step=0.1)
fan_y = st.sidebar.number_input("Fan Y Position (meters)", min_value=0.0, value=2.0, step=0.1)
fan_z = st.sidebar.number_input("Fan Z Position (meters)", min_value=0.0, value=1.5, step=0.1)

# AC settings with better defaults
ac_x = st.sidebar.number_input("AC X Position (meters)", min_value=0.0, value=0.0, step=0.1)
ac_y = st.sidebar.number_input("AC Y Position (meters)", min_value=0.0, value=1.5, step=0.1)
ac_z = st.sidebar.number_input("AC Z Position (meters)", min_value=0.0, value=1.0, step=0.1)

# Room dimensions
room_length = st.sidebar.number_input("Room Length (meters)", min_value=1.0, value=8.0, step=1.0)
room_width = st.sidebar.number_input("Room Width (meters)", min_value=1.0, value=4.0, step=1.0)
room_height = st.sidebar.number_input("Room Height (meters)", min_value=1.0, value=3.0, step=1.0)

# Simulation time
time_duration = st.sidebar.number_input("Simulation Time (seconds)", min_value=0.01, value=0.03, step=0.01)

# Button to run simulation and plot results
if st.sidebar.button("Run Simulation and Show Plot"):
    # Show spinner while the simulation is running
    with st.spinner('Simulating airflow... Please wait.'):
        # Run the CFD simulation
        X, Y, Z, U, V, W = simulate_airflow(
            fan_cmm=fan_cmm,
            fan_rpm=fan_rpm,
            fan_diameter=fan_diameter,
            fan_position=(fan_x, fan_y, fan_z),
            ac_position=(ac_x, ac_y, ac_z),
            room_dimensions=(room_length, room_width, room_height),
            time_duration=time_duration
        )

    # Calculate the velocity magnitude at each point
    velocity_magnitude = np.sqrt(U**2 + V**2 + W**2)

    # Create 3D interactive plot
    fig = go.Figure()

    # Add volume rendering of velocity magnitude
    fig.add_trace(go.Volume(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
        value=velocity_magnitude.ravel(),
        isomin=velocity_magnitude.min(),
        isomax=velocity_magnitude.max(),
        opacity=0.1,
        surface_count=20,
        colorscale='Viridis',
        colorbar=dict(title="Velocity (m/s)")
    ))

    # Add scatter points for AC and Fan positions
    fig.add_trace(go.Scatter3d(
        x=[ac_x], y=[ac_y], z=[ac_z],  # AC position
        mode='markers',
        marker=dict(size=5, color='red'),
        name="AC Inlet"
    ))

    fig.add_trace(go.Scatter3d(
        x=[fan_x], y=[fan_y], z=[fan_z],  # Fan position
        mode='markers',
        marker=dict(size=5, color='green'),
        name="Fan"
    ))

    # Update layout to increase plot size to full screen
    fig.update_layout(
        autosize=True,
        height=700,
        width=1000,
        scene=dict(
            xaxis=dict(title="X (meters)"),
            yaxis=dict(title="Y (meters)"),
            zaxis=dict(title="Z (meters)")
        ),
        title="Interactive 3D Airflow Heatmap with AC and Fan",
        showlegend=True
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
