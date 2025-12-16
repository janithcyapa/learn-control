import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from QAttitude.rep.quaternion import Quaternion


def plot_attitude(
    q: Quaternion,
    axis_length=1.0,
    title="Single Attitude Visualization",
    show=True
):
    """
    Plot a single attitude using body-frame axes rotated by quaternion.

    World frame: X (red), Y (green), Z (blue)
    Body frame:  x_b, y_b, z_b (dashed)
    """

    q = q.normalized()

    # World axes
    world_axes = np.eye(3) * axis_length

    # Rotate body axes
    body_axes = np.array([
        q.rotate([axis_length, 0, 0]),
        q.rotate([0, axis_length, 0]),
        q.rotate([0, 0, axis_length])
    ])

    fig = go.Figure()

    # ---- World frame ----
    colors = ["red", "green", "blue"]
    labels = ["X", "Y", "Z"]

    for i in range(3):
        fig.add_trace(go.Scatter3d(
            x=[0, world_axes[i, 0]],
            y=[0, world_axes[i, 1]],
            z=[0, world_axes[i, 2]],
            mode="lines+text",
            line=dict(color=colors[i], width=3, dash="dash"),
            text=["", labels[i]],
            name=f"World {labels[i]}"
        ))

    # ---- Body frame ----
    body_labels = ["x_b", "y_b", "z_b"]
    for i in range(3):
        fig.add_trace(go.Scatter3d(
            x=[0, body_axes[i, 0]],
            y=[0, body_axes[i, 1]],
            z=[0, body_axes[i, 2]],
            mode="lines+text",
            line=dict(color=colors[i], width=5),
            text=["", body_labels[i]],
            name=f"Body {body_labels[i]}"
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[-axis_length, axis_length]),
            yaxis=dict(range=[-axis_length, axis_length]),
            zaxis=dict(range=[-axis_length, axis_length]),
            aspectmode="cube"
        ),
        legend=dict(x=0.02, y=0.95)
    )

    if show:
        fig.show()

    return fig
