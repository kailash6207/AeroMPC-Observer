# AeroMPC-Observer: Augmented 5-State Drone MPC with EKF Disturbance Observer

An advanced cyber-physical Guidance, Navigation, and Control (GNC) simulation framework featuring a multi-variable **Augmented 5-State Model Predictive Controller (MPC)** tightly coupled with an **Extended Kalman Filter (EKF) Disturbance Observer** to achieve Active Disturbance Rejection Control (ADRC).

---

## 🛰️ Control Architecture & Methodology

* **Augmented 5-State State Space Model:** The plant model dynamically tracks tracking states across an expanded dimensional vector: 
  $$x_{\text{aug}} = \begin{bmatrix} z & \dot{z} & \theta & \dot{\theta} & \int z_{\text{err}} \, dt \end{bmatrix}^T$$
* **Integral Error Elimination:** Integrating an internal error state variable into the optimal control problem optimization loop actively forces tracking offsets to zero under constant environmental loading.
* **Active Disturbance Observer:** The EKF runs a 6th state variable tracking vertical acceleration bias ($d_t$). By observing the mathematical innovations between raw sensor feedback and expected deterministic state trajectories, it isolates severe downward wind forces or model errors in real time.
* **Dynamic Feed-Forward Compensation:** The calculated wind force estimate is instantly fed forward into the receding horizon path optimizer to apply an immediate torque/thrust counter-balance bias before errors propagate into the position vector.

---

## 📂 Repository Architecture

```text
├── app.py              # FastAPI micro-service exposing asynchronous REST endpoints
├── simulator.py        # Augmented 5-state plant dynamics, MPC horizon solver, & EKF
├── index.html          # HTML5 Canvas real-time viewport and telemetry dashboard plots
└── .gitignore          # Strips out bytecode artifacting and local environment caches

🚀 Installation & Launch Execution
1. Environment Configurations
Clone this repository to your local machine and set up your execution parameters:

Bash
pip install fastapi uvicorn numpy scipy cvxpy
2. Boot the Core Control Engines
Fire up the backend micro-engine process using your localized terminal workspace:

Bash
uvicorn app:app --reload
3. Initialize the Control Workstation Panel
Open the index.html dashboard panel directly using your local browser server context (e.g., VS Code Live Server or by executing the file path layout directly at http://127.0.0.1:5500).

Input your baseline operational setpoints (e.g., 2.0m, 3.0m), select Launch Mission, and observe the active disturbance observer reject step changes seamlessly!
