# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:50:45 2025

@author: viliP
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import os

# ==========================
# Settings for each particle
# ==========================
particles = [
    {
        "name": "Proton",
        "srim_file": "C:\\Users\\viliP\\Desktop\\SRIM 2013\\SRIM Outputs\\Proton_in_Silicon_15MeV_trimmed.txt",
        "initial_energy_keV": 16000,
        "min_energy_keV": 4000,
        "color": "Magenta",
        "label": "Proton",
        "step_keV": 1000,
        "cutg_name": "Proton_pE_dE"
    },
    {
        "name": "Deuteron",
        "srim_file": "C:\\Users\\viliP\\Desktop\\SRIM 2013\\SRIM Outputs\\Deuteron_in_Silicon_23MeV_trimmed.txt",
        "initial_energy_keV": 20000,
        "min_energy_keV": 5000,
        "color": "Green",
        "label": "Deuteron",
        "step_keV": 1000,
        "cutg_name": "Deuteron_pE_dE"
    },
    {
        "name": "Tritium",
        "srim_file": "C:\\Users\\viliP\\Desktop\\SRIM 2013\\SRIM Outputs\\Tritium_in_Silicon_27MeV_trimmed.txt",
        "initial_energy_keV": 30000,
        "min_energy_keV": 6000,
        "color": "Cyan",
        "label": "Tritium",
        "step_keV": 1000,
        "cutg_name": "Tritium_pE_dE"
   },
    {
        "name": "Alpha",
        "srim_file": "C:\\Users\\viliP\\Desktop\\SRIM 2013\\SRIM Outputs\\Helium_in_Silicon_65MeV_trimmed.txt",
        "initial_energy_keV": 62000,
        "min_energy_keV": 16000,
        "color": "Red",
        "label": "Alpha",
        "step_keV": 1000,
        "cutg_name": "Alpha_pE_dE"
    },
    {
        "name": "Lithium7",
        "srim_file": "C:\\Users\\viliP\\Desktop\\SRIM 2013\\SRIM Outputs\\Lithium_in_Silicon_80_MeV_trimmed.txt",
        "initial_energy_keV": 76000,
        "min_energy_keV": 32000,
        "color": "Orange",
        "label": "Lithium",
        "step_keV": 2000,
        "cutg_name": "Lithium7_pE_dE"
    },
]

# Detector geometry
length = 26.5e+3      # mm
height_min = 9e+3     # mm
height_max = 40e+3    # mm
width = 140           # microns
angle_min = np.radians(16)#np.arctan(height_min / length)
angle_max = np.radians(51) #np.arctan(height_max / length)
path_min = width / np.cos(angle_min)
path_max = width / np.cos(angle_max)

# ==========================
# Helper functions
# ==========================

def load_srim_data(file_path):
    energy = []
    stopping_power = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) >= 3:
                energy.append(float(parts[0]))
                stopping_power.append(float(parts[2]))
    for i in range(1, len(energy)):
        if energy[i] < energy[i - 1]:
            for j in range(i, len(energy)):
                energy[j] *= 1000  # MeV to keV
            break
    return np.array(energy), np.array(stopping_power)

def stopping_interp_func(energy, stopping_power):
    return interp1d(energy, stopping_power, kind='cubic', fill_value='extrapolate')

def dE_dx(stopping_interp):
    return lambda x, E: -stopping_interp(E[0])

def compute_energy_loss(energy0, dE_dx_func, path):
    sol = solve_ivp(dE_dx_func, [0, path], [energy0], max_step=1e-1)
    return energy0 - sol.y[0][-1]

# ==========================
# Main loop over particles
# ==========================

plt.figure()

output_path = "cutg_shapes.txt"
cutg_file = open(output_path, "w")

for particle in particles:
    file_path = os.path.join("C:\\Users\\viliP\\Desktop\\SRIM 2013\\SRIM Outputs", particle["srim_file"])
    energy, stopping_power = load_srim_data(file_path)
    stopping_func = stopping_interp_func(energy, stopping_power)
    dEdx = dE_dx(stopping_func)

    E_loss_min = []
    E_loss_max = []
    Energies_min = []
    Energies_max = []

    E = particle["initial_energy_keV"]
    while E >= particle["min_energy_keV"]:
        loss_min = compute_energy_loss(E, dEdx, path_min)
        loss_max = compute_energy_loss(E, dEdx, path_max)

        E_loss_min.append(loss_min)
        E_loss_max.append(loss_max)
        Energies_min.append(E)
        Energies_max.append(E)

        E -= particle["step_keV"]

    plt.plot(Energies_min, E_loss_min, "o--", color=particle["color"], label=f"{particle['label']} min path")
    plt.plot(Energies_max, E_loss_max, "o--", color=particle["color"], label=f"{particle['label']} max path", alpha=0.6)
    
    # Export TCutG format for ROOT to file
    cutg_name = f"{particle['cutg_name']}"
    cutg_pointer = f"cutg_{particle['name']}"
    index = len(Energies_min) + len(Energies_max) + 1
    cutg_file.write(f'TCutG *{cutg_pointer} = new TCutG("{cutg_name}", {index});\n')

    i = 0
    for x, y in zip(Energies_min, E_loss_min):
        cutg_file.write(f'{cutg_pointer}->SetPoint({i}, {x}, {y});\n')
        i += 1

    for x, y in zip(np.flip(Energies_max), np.flip(E_loss_max)):
        cutg_file.write(f'{cutg_pointer}->SetPoint({i}, {x}, {y});\n')
        i += 1

    cutg_file.write(f'{cutg_pointer}->SetPoint({i}, {Energies_min[0]}, {E_loss_min[0]});\n')
    cutg_file.write(f'{cutg_pointer}->SetLineColor(k{particle['color']});\n')
    cutg_file.write(f'{cutg_pointer}->SetLineWidth(3);\n\n')


cutg_file.close()
print(f"TCutG data written to {output_path}")

# ==========================
# Finalize plot
# ==========================

plt.title("Energy loss vs Initial Energy for Multiple Particles")
plt.xlabel("Initial Energy (keV)")
plt.ylabel("Energy loss in 140 µm (keV)")
#plt.yticks(np.arange(0,50000,10000))
plt.grid()
plt.legend(loc="upper left" ,fontsize="8")
plt.tight_layout()
plt.savefig("MultiParticle_EnergyLoss.png")
plt.show()

