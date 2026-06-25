from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from simulator import DroneStateSpaceSim

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

drone_sim = DroneStateSpaceSim()

@app.get("/")
async def read_root():
    return {"message": "Augmented 5-State Multi-Variable GNC Matrix Hyper-Engine Live"}

@app.get("/api/step")
async def simulation_step(target_z: float = 2.0):
    telemetry = drone_sim.step(target_z)
    return telemetry

@app.get("/api/reset")
async def reset_simulation():
    global drone_sim
    drone_sim.is_active = False
    
    drone_sim = DroneStateSpaceSim() 
    drone_sim.true_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    drone_sim.est_state_augmented = np.zeros(6)
    return {"status": "system reset complete"}