import numpy as np
import scipy.linalg
import cvxpy as cp

class DroneStateSpaceSim:
    def __init__(self):
        # Physical Parameters
        self.m = 1.0       
        self.I_y = 0.02    
        self.L = 0.25      
        self.g = 9.81      
        
        # --- UPGRADE: AUGMENTED 5-STATE MATRIX DEFINITIONS [z, z_dot, theta, theta_dot, integral_z] ---
        self.A_nom = np.array([
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0] # Integrator tracks position error over time
        ])
        
        self.B_nom = np.array([
            [0, 0],
            [1/self.m, 1/self.m],
            [0, 0],
            [-self.L/self.I_y, self.L/self.I_y],
            [0, 0]
        ])
        
        self.C = np.array([
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0]
        ])
        
        self.dt = 0.05 
        self.time = 0.0
        self.is_active = True  

        # Discretization via Zero-Order Hold (ZOH) on the 5-State Model
        augmented_matrix = np.zeros((7, 7))
        augmented_matrix[0:5, 0:5] = self.A_nom
        augmented_matrix[0:5, 5:7] = self.B_nom
        matrix_exponential = scipy.linalg.expm(augmented_matrix * self.dt)
        self.Ad = matrix_exponential[0:5, 0:5]
        self.Bd = matrix_exponential[0:5, 5:7]

        # --- UPGRADE: LQR BALANCED FALLBACK FOR 5-STATES ---
        Q_lqr = np.diag([40.0, 4.0, 80.0, 8.0, 15.0])  
        R_lqr = np.diag([5.0, 5.0])              
        P_lqr = scipy.linalg.solve_continuous_are(self.A_nom, self.B_nom, Q_lqr, R_lqr)
        self.K_fallback = scipy.linalg.inv(R_lqr) @ self.B_nom.T @ P_lqr

        # --- UPGRADE: EXPANDED PREDICTION HORIZON ---
        self.N = 5  # Expanded horizon window for anticipatory path optimization
        self.Q_mpc = np.diag([250.0, 15.0, 100.0, 6.0, 25.0]) # 5th weight penalizes tracking drift over time
        self.R_mpc = np.diag([1.2, 1.2])  
        
        self.u_min = 0.0    
        self.u_max = 22.0   
        self.u_prev = np.array([4.905, 4.905])
        self.max_motor_diff = 2.0
        self.mpc_interval = 1  
        self._step_counter = 0
        self._last_u = self.u_prev.copy() 

        # --- UPGRADE: EKF WITH ACTIVE WIND ACCELERATION BIAS STATE ---
        # Covariance size expands to accommodate the new state [z, z_dot, theta, theta_dot, integral_z, wind_bias]
        self.Q_proc = np.diag([0.01, 0.05, 0.01, 0.05, 0.001, 0.2]) # High variance on wind state lets it track rapid changes
        self.R_sens = np.diag([0.20, 0.10])             
        self.P_est = np.eye(6) * 0.1 
        
        # --- INITIAL CONFIGURATIONS ---
        self.true_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) 
        self.est_state_augmented = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Holds the 6th wind state element
        
        self.has_taken_off = False
        self.red_line_altitude = 0.0
        self.mission_phase = 'launch'
        self.last_target_z = 0.0
        self.return_delay = 10.0
        self.contact_counter = 0
        self.contact_debounce_steps = 3

        # Performance Accumulators
        self.iae_accumulator = 0.0  
        self.tce_accumulator = 0.0  

    def solve_mpc(self, x_current, target_z, estimated_wind_accel):
        x = cp.Variable((5, self.N + 1))
        u = cp.Variable((2, self.N))
        
        cost = 0
        constraints = [x[:, 0] == x_current] 
        target_state = np.array([target_z, 0.0, 0.0, 0.0, 0.0])
        u_hover = (self.m * self.g) / 2.0  

        for k in range(self.N):
            cost += cp.quad_form(x[:, k] - target_state, self.Q_mpc)
            cost += cp.quad_form(u[:, k] - np.array([u_hover, u_hover]), self.R_mpc)
            
            # --- UPGRADE: DISTURBANCE COMPENSATED STATE PREDICTION ---
            # Wind acceleration vector maps directly to vertical velocity calculations
            wind_compensation_vector = np.array([0.0, estimated_wind_accel, 0.0, 0.0, 0.0]) * self.dt
            constraints += [x[:, k+1] == self.Ad @ x[:, k] + self.Bd @ u[:, k] + wind_compensation_vector]
            
            constraints += [u[0, k] >= self.u_min, u[0, k] <= self.u_max]
            constraints += [u[1, k] >= self.u_min, u[1, k] <= self.u_max]
            constraints += [x[0, k+1] >= -0.1]
            constraints += [x[2, k+1] >= -0.35, x[2, k+1] <= 0.35]
            constraints += [u[0, k] - u[1, k] <= self.max_motor_diff, u[1, k] - u[0, k] <= self.max_motor_diff]
            
            if k == 0:
                constraints += [u[:, 0] - self.u_prev >= -6.0, u[:, 0] - self.u_prev <= 6.0]
            else:
                constraints += [u[:, k] - u[:, k-1] >= -6.0, u[:, k] - u[:, k-1] <= 6.0]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.OSQP, warm_start=True)
            if u[:, 0].value is not None:
                u0 = u[:, 0].value
                self.u_prev = u0
                return u0
            raise ValueError("Infeasible")
        except:
            error = x_current - target_state
            u_nominal_hover = (self.m * self.g) / (2.0 * np.cos(x_current[2]) + 1e-5)
            # Subtract feedforward wind balance force directly from fallback
            u_recovery = np.array([u_nominal_hover, u_nominal_hover]) - (self.K_fallback @ error) - (estimated_wind_accel * self.m / 2.0)
            u_recovery = np.clip(u_recovery, self.u_min, self.u_max)
            self.u_prev = u_recovery
            return u_recovery

    def step(self, base_target_z):
        if not self.is_active:
            return self._generate_telemetry_payload(base_target_z, np.array([0.0, 0.0]), np.array([0.0, 0.0]))

        self.time += self.dt
        if self.mission_phase == 'launch':
            target_z = base_target_z + 1.2 * np.tanh(0.4 * self.time)
            self.last_target_z = target_z 
        else:
            target_z = self.last_target_z * 0.96
            if target_z < 0.05:
                target_z = self.red_line_altitude
            self.last_target_z = target_z

        # Separate 5-state current vector and extract 6th wind state from observer
        x_current_5state = self.est_state_augmented[0:5]
        estimated_wind_accel = self.est_state_augmented[5]

        # 2. RUN CONTROLLER SOLVER WITH ACTIVE BIAS FEED-FORWARD
        if self._step_counter % self.mpc_interval == 0:
            u = self.solve_mpc(x_current_5state, target_z, estimated_wind_accel)
            u = np.clip(u, self.u_min, self.u_max)
            self._last_u = u
        else:
            u = self._last_u
        self._step_counter += 1 
        
        # 3. RUN PHYSICS ENVIRONMENT WITH AN INTEGRATED CONTINUOUS STOCHASTIC WIND FORCE
        u_total = u[0] + u[1]
        theta_true = self.true_state[2]
        
        # Add a constant downward force (e.g., severe localized downdraft) to verify rejection profiles
        true_wind_disturbance_acceleration = -1.85 
        
        z_ddx = (u_total * np.cos(theta_true)) / self.m - self.g + true_wind_disturbance_acceleration
        theta_ddx = (self.L * (-u[0] + u[1])) / self.I_y  
        
        # Update true continuous kinematics
        self.true_state[1] += z_ddx * self.dt
        self.true_state[0] += self.true_state[1] * self.dt
        self.true_state[3] += theta_ddx * self.dt
        self.true_state[2] += self.true_state[3] * self.dt
        # Accumulate true mathematical error integration state
        self.true_state[4] += (self.true_state[0] - target_z) * self.dt
        
        contact_tolerance = 1e-3
        if self.true_state[0] <= self.red_line_altitude + contact_tolerance:
            self.contact_counter += 1
        else:
            self.contact_counter = 0

        if not self.has_taken_off and self.true_state[0] <= self.red_line_altitude + contact_tolerance:
            self.true_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.contact_counter = 0

        if self.has_taken_off and self.contact_counter >= self.contact_debounce_steps:
            self.true_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.est_state_augmented = np.zeros(6)
            self.is_active = False
            u = np.array([0.0, 0.0])
            return self._generate_telemetry_payload(target_z, u, np.array([0.0, 0.0]))

        if self.true_state[0] > 0.1:
            if not self.has_taken_off:
                self.has_taken_off = True
                self.liftoff_time = self.time
            if self.mission_phase == 'launch' and self.time - self.liftoff_time >= self.return_delay:
                self.mission_phase = 'return'
            
        if self.is_active:
            self.iae_accumulator += abs(self.true_state[0] - target_z) * self.dt
            self.tce_accumulator += (u[0]**2 + u[1]**2) * self.dt

        # 4. UPGRADE: EXTENDED KALMAN FILTER WITH DISTURBANCE ACCEL COUPLING
        noise = np.array([np.random.normal(0, 0.15), np.random.normal(0, 0.06)])
        measured = self.C @ self.true_state + noise
        
        # State transitions for prediction step
        z_est = self.est_state_augmented[0]
        z_dot_est = self.est_state_augmented[1]
        theta_est = self.est_state_augmented[2]
        theta_dot_est = self.est_state_augmented[3]
        integral_est = self.est_state_augmented[4]
        wind_bias_est = self.est_state_augmented[5]
        
        # Non-linear calculation step using the current wind bias estimate
        z_ddx_est = ((u[0] + u[1]) * np.cos(theta_est)) / self.m - self.g + wind_bias_est
        theta_ddx_est = (self.L * (-u[0] + u[1])) / self.I_y  
        
        x_pred = np.array([
            z_est + z_dot_est * self.dt,
            z_dot_est + z_ddx_est * self.dt,
            theta_est + theta_dot_est * self.dt,
            theta_dot_est + theta_ddx_est * self.dt,
            integral_est + (z_est - target_z) * self.dt,
            wind_bias_est # Wind is modeled as a random walk process
        ])
        
        # Calculate Jacobian Matrix (6x6)
        F_jac = np.zeros((6, 6))
        F_jac[0, 1] = 1.0
        F_jac[2, 3] = 1.0
        F_jac[4, 0] = 1.0
        F_jac[1, 2] = -((u[0] + u[1]) * np.sin(theta_est)) / self.m
        F_jac[1, 5] = 1.0 # Links wind acceleration directly to downward speed variations
        
        # Project Covariance ahead
        A_discrete_jac = np.eye(6) + F_jac * self.dt
        P_pred = A_discrete_jac @ self.P_est @ A_discrete_jac.T + self.Q_proc
        
        # Measurement Mapping (2x6)
        H_meas = np.zeros((2, 6))
        H_meas[0, 0] = 1.0
        H_meas[1, 2] = 1.0
        
        # Compute Kalman Gain matrix
        innovation_cov = H_meas @ P_pred @ H_meas.T + self.R_sens
        K_gain = P_pred @ H_meas.T @ scipy.linalg.inv(innovation_cov)
        
        # Correct and store final state vectors
        innovation = measured - (H_meas @ x_pred)
        self.est_state_augmented = x_pred + K_gain @ innovation
        self.P_est = (np.eye(6) - K_gain @ H_meas) @ P_pred
        
        return self._generate_telemetry_payload(target_z, u, measured)

    def _generate_telemetry_payload(self, target_z, u, measured):
        return {
            "true_z": float(self.true_state[0]),
            "est_z": float(self.est_state_augmented[0]),
            "noisy_z": float(measured[0]),
            "target_z": float(target_z),
            "true_theta": float(self.true_state[2]),
            "u1": float(u[0]),
            "u2": float(u[1]),
            "iae": float(self.iae_accumulator),
            "tce": float(self.tce_accumulator),
            "aborted": bool(not self.is_active)
        }