import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Kalman Filter for IMU Data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Helpers
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import plotly.express as px
    return (pd,)


@app.cell
def _(pd, plt):
    def ReadData(file_name):
        df = pd.read_csv('IMU_Data.csv')

    # Load data
    df = pd.read_csv('IMU_Data.csv')

    # Plot Gyroscope Data
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(df['Gyro_x'], label='Gyro_x')
    plt.plot(df['Gyro_y'], label='Gyro_y')
    plt.plot(df['Gyro_z'], label='Gyro_z')
    plt.title('Gyroscope Data')
    plt.legend()

    # Plot Accelerometer Data
    plt.subplot(2, 1, 2)
    plt.plot(df['Acc_x'], label='Acc_x')
    plt.plot(df['Acc_y'], label='Acc_y')
    plt.plot(df['Acc_z'], label='Acc_z')
    plt.title('Accelerometer Data')
    plt.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Basic Theory
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The Kalman Filter is a state estimation algorithm that provides both an estimate of the current state and a prediction of the future state, along with a measure of their uncertainty. Moreover, it is an optimal algorithm that minimizes state estimation uncertainty. That is why the Kalman Filter has become such a widely used and trusted algorithm.
    ![image.png](attachment:9da2823c-b347-485d-8982-982818808463.png)

     The Kalman Filter output is a multivariate random variable. A covariance matrix describes the squared uncertainty of the multivariate random variable.

    The uncertainty variables of the multivariate Kalman Filter are:

    - $P_{n,n}$ - is a covariance matrix that describes the squared uncertainty of an estimate
    - $P_{n+1,n}$ - is a covariance matrix that describes the squared uncertainty of a prediction
    - $R_n$ - is a covariance matrix that describes the squared measurement uncertainty
    - $Q$ - is a covariance matrix that describes the process noise
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### State Extrapolation Equation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using the state extrapolation equation, we can predict the next system state based on the knowledge of the current state. It extrapolates the state vector from the present (time step n) to the future (time step n+1).

    The predicted system state equation is:
    $$\hat{x}_{n+1,n} = F\hat{x}_{n,n} + Gu_n + w_n$$

    Where:
    * $\hat{x}_{n+1,n}$ is a **predicted system state vector** at time step $n + 1$
    * $\hat{x}_{n,n}$ is an **estimated system state vector** at time step $n$
    * $u_n$ is a **control variable** or **input variable** - a $\text{measurable}$ (deterministic) input to the system
    * $w_n$ is a **process noise** or disturbance - an $\text{unmeasurable}$ input that affects the state
    * $F$ is a **state transition matrix**
    * $G$ is a **control matrix** or $\text{input transition matrix}$ (mapping control to state variables)


    #### Linear dynamic systems

    For zero-order hold sampling, assuming the input is piecewise constant, the general solution of the state space equation in the form of:
    $$\dot{x}(t) = Ax(t) + Bu(t)$$
    is given by:
    $$x(t + \Delta t) = \underbrace{e^{A\Delta t}}_{\text{F}} x(t) + \underbrace{\int_{0}^{\Delta t} e^{A\tau} B u(t) d\tau}_{\text{G}}$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Covariance Extrapolation Equation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The general form of the Covariance Extrapolation Equation is given by:

    $$P_{n+1,n} = FP_{n,n}F^T + Q$$

    Where:
    * $P_{n,n}$ is the **squared uncertainty of an estimate** (covariance matrix) of the current state
    * $P_{n+1,n}$ is the **squared uncertainty of a prediction** (covariance matrix) for the next state
    * $F$ is the **state transition matrix** that we derived in the "Modeling linear dynamic systems" section
    * $Q$ is the **process noise matrix**


    $w_n$ is the process noise at the time step n. In the multidimensional case, the process noise is a covariance matrix denoted by Q. Process noise variance has a critical influence on the Kalman Filter performance. Too small q causes a lag error. If the q value is too high, the Kalman Filter follows the measurements and produces noisy estimations.

    The process noise can be independent between different state variables. In this case, the process noise covariance matrix Q is a diagonal matrix.The process noise can also be dependent. For example, the constant velocity model assumes zero acceleration (a=0). However, a random variance in acceleration causes a variance in velocity and position. In this case, the process noise is correlated with the state variables.

    There are two models for the environmental process noise.
    - Discrete noise model
    - Continuous noise model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Measurement Equation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The generalized measurement equation in matrix form is given by:

    $$z_n = Hx_n + v_n$$

    Where:
    * $\mathbf{z}_n$ is a **measurement vector**
    * $\mathbf{x}_n$ is a **true system state** (hidden state)
    * $\mathbf{v}_n$ is a **random noise vector**
    * $\mathbf{H}$ is an **observation matrix**

    In many cases, the measured value is not the desired system state. For example, a digital electric thermometer measures an electric current, while the system state is the temperature. There is a need for a transformation of the system state (input) to the measurement (output). The purpose of the observation matrix H is to convert the system state into outputs using linear transformations. The following chapters include examples of observation matrix usage.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Covariance Equations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Measurement Uncertainty

    The measurement covariance is given by:
    $$R_n = E(\mathbf{v}_n \mathbf{v}_n^T)$$

    Where:
    * $\mathbf{R}_n$ is the **covariance matrix of the measurement**
    * $\mathbf{v}_n$ is the **measurement error**

    #### Process Noise Uncertainty

    The process noise covariance is given by:
    $$Q_n = E(\mathbf{w}_n \mathbf{w}_n^T)$$

    Where:
    * $\mathbf{Q}_n$ is the **covariance matrix of the process noise**
    * $\mathbf{w}_n$ is the **process noise**

    #### Estimation Uncertainty

    The estimation covariance is given by:
    $$P_{n,n} = E(\mathbf{e}_n \mathbf{e}_n^T) = E((\mathbf{x}_n - \hat{\mathbf{x}}_{n,n}) (\mathbf{x}_n - \hat{\mathbf{x}}_{n,n})^T)$$

    Where:
    * $\mathbf{P}_{n,n}$ is the **covariance matrix of the estimation error**
    * $\mathbf{e}_n$ is the **estimation error**
    * $\mathbf{x}_n$ is the **true system state** (hidden state)
    * $\hat{\mathbf{x}}_{n,n}$ is the **estimated system state vector at time step $n$**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### State Update Equation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The State Update Equation in the matrix form is given by,

    $$\hat{x}_{n,n} = \hat{x}_{n,n-1} + K_n (z_n - H\hat{x}_{n,n-1})$$

    Where:
    * $\hat{x}_{n,n}$ is an **estimated system state vector at time step $n$**
    * $\hat{x}_{n,n-1}$ is a **predicted system state vector at time step $n - 1$**
    * $K_n$ is a **Kalman Gain**
    * $z_n$ is a **measurement**
    * $H$ is an **observation matrix**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Covariance Update Equation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The Covariance Update Equation is given by,

    $$P_{n,n} = (I - K_n H) P_{n,n-1} (I - K_n H)^T + K_n R_n K_n^T$$

    where:
    * $P_{n,n}$ is the **covariance matrix of the current state estimation**
    * $P_{n,n-1}$ is the **prior estimate covariance matrix of the current state** (predicted at the previous state)
    * $K_n$ is the **Kalman Gain**
    * $H$ is the **observation matrix**
    * $R_n$ is the **measurement noise covariance matrix**
    * $I$ is an **Identity Matrix** (the $n \times n$ square matrix with ones on the main diagonal and zeros elsewhere)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Kalman Gain Equation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The Kalman Gain in matrix notation is given by,

    $$K_n = P_{n,n-1}H^T (HP_{n,n-1}H^T + R_n)^{-1}$$

    Where:
    * $\mathbf{K}_n$ is the **Kalman Gain**
    * $\mathbf{P}_{n,n-1}$ is the **prior estimate covariance matrix of the current state** (predicted at the previous step)
    * $\mathbf{H}$ is the **observation matrix**
    * $\mathbf{R}_n$ is the **measurement noise covariance matrix**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![image.png](attachment:7e559476-d6f0-4f2c-adf7-affc7c5ceac3.png)

    The following table summarizes notation (including differences found in the literature) and dimensions.

    | Term | Name | Alternative term | Dimensions |
    | :---: | :--- | :---: | :---: |
    | $\mathbf{x}$ | State Vector | | $n_x \times 1$ |
    | $\mathbf{z}$ | Measurements Vector | $\mathbf{y}$ | $n_z \times 1$ |
    | $\mathbf{F}$ | State Transition Matrix | $\mathbf{\Phi}, \mathbf{A}$ | $n_x \times n_x$ |
    | $\mathbf{u}$ | Input Variable | | $n_u \times 1$ |
    | $\mathbf{G}$ | Control Matrix | $\mathbf{B}$ | $n_x \times n_u$ |
    | $\mathbf{P}$ | Estimate Covariance | $\mathbf{\Sigma}$ | $n_x \times n_x$ |
    | $\mathbf{Q}$ | Process Noise Covariance | | $n_x \times n_x$ |
    | $\mathbf{R}$ | Measurement Covariance | | $n_z \times n_z$ |
    | $\mathbf{w}$ | Process Noise Vector | | $n_x \times 1$ |
    | $\mathbf{v}$ | Measurement Noise Vector | | $n_z \times 1$ |
    | $\mathbf{H}$ | Observation Matrix | $\mathbf{C}$ | $n_z \times n_x$ |
    | $\mathbf{K}$ | Kalman Gain | | $n_x \times n_z$ |
    | $\mathbf{n}$ | Discrete-Time Index | $\mathbf{k}$ | |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Linear Euler-Angle Kalman Filter
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this note, I derive a simple **linear Kalman filter** for estimating **Euler angles**,
    - Roll, ($\phi$)
    - Pitch, ($\theta$)
    - Yaw, ($\psi

    Assume:
    - **Gyroscope** gives angular rates $(p, q, r)$ around body axes.
    - **Accelerometer** is used to estimate roll and pitch from gravity.
    - **Yaw is not corrected** (it will drift).

    Use a **discrete-time linear state-space model** and the **standard Kalman filter**,
    - Prediction: use gyro to propagate angles.
    - Update: use accelerometer to correct roll & pitch.

    ---

    ### 1. State, Inputs, and Measurements
    #### 1.1 State Vector
    Define the state vector as the three Euler angles,
    $x_k =
    \begin{bmatrix}
    \phi_k\\
    \theta_k\\
    \psi_k
    \end{bmatrix}
    $
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
