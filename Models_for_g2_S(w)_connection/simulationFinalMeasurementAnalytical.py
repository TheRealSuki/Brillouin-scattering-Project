import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.fft import fft, fftshift, fftfreq

# A small constant to prevent division by zero in normalization
eps = 1e-15

def generate_single_mode(p):
    """
    Calculates the evolution of a single mode based on its parameters using an
    analytical solution.
    """
    from numpy.lib import scimath

    k_j = -1j * p['gtilde'] * p['A_p']
    alpha_m = p['alpha_p']
    chi_s = -p['alpha_s']/2 + 1j * p['Delta_s']
    chi_m = -alpha_m/2 - 1j * p['Delta_m']

    A = (chi_s + chi_m) / 2
    D = scimath.sqrt((chi_s - chi_m)**2 + 4 * k_j * np.conj(k_j))
    lambda_p = A + D/2
    lambda_m = A - D/2

    P_denom = chi_s - lambda_p
    Q_denom = chi_s - lambda_m
    
    P = k_j / (P_denom + eps)
    Q = k_j / (Q_denom + eps)
    L = 1 / (Q - P + eps)

    z = np.linspace(0, p['z_max'], p['num_points'])
    term1 = (-P * np.exp(lambda_p * z) + Q * np.exp(lambda_m * z)) * p['a0']
    term2 = (P * Q * (-np.exp(lambda_p * z) + np.exp(lambda_m * z))) * p['b0_dagger']
    a_z = L * (term1 + term2)
    
    return z, a_z


def compute_g1_vs_tau(a1, a2):
    """
    Computes the first-order temporal coherence g^(1)(tau).
    """
    N = len(a1)
    taus = np.arange(-N + 1, N)
    g1 = np.zeros(len(taus), dtype=complex)
    norm = np.sqrt(np.mean(np.abs(a1)**2) * np.mean(np.abs(a2)**2)) + eps
    for i, tau in enumerate(taus):
        if tau >= 0:
            x1 = a1[:N - tau]
            x2 = a2[tau:]
        else:
            x1 = a1[-tau:]
            x2 = a2[:N + tau]
        g1[i] = np.mean(np.conj(x1) * x2) / norm
    return taus, g1



def compute_g2_vs_tau(a1, a2):
    """
    Computes the second-order temporal coherence g^(2)(tau).
    """
    N = len(a1)
    I1 = np.abs(a1)**2
    I2 = np.abs(a2)**2
    taus = np.arange(-N + 1, N)
    g2 = np.zeros(len(taus), dtype=float)
    norm = np.mean(I1) * np.mean(I2) + eps
    for i, tau in enumerate(taus):
        if tau >= 0:
            x1 = I1[:N - tau]
            x2 = I2[tau:]
        else:
            x1 = I1[-tau:]
            x2 = I2[:N + tau]
        g2[i] = np.mean(x1 * x2) / norm
    return taus, g2

def compute_power_spectrum(x, z, zero_pad_factor=8, window=True):
    """
    Computes the power spectrum of a signal x.
    """
    dt = z[1] - z[0]
    N = len(x)
    Npad = int(2**np.ceil(np.log2(N * zero_pad_factor)))
    if window:
        xw = x * np.hanning(N)
    else:
        xw = x
    X = fftshift(fft(xw, n=Npad))
    S = np.abs(X)**2
    freq = fftshift(fftfreq(Npad, d=dt))
    omega = 2 * np.pi * freq
    return omega, S

# ------------------------------
# --- PARAMETER DEFINITIONS ---
# ------------------------------
z_to_ns = 12.5 / np.pi  # Conversion factor from z-units to nanoseconds

# Mode 1
params1 = {
    'alpha_s': 3.1735e+01, 
    'alpha_p': 3.6134e+01, 
    'Delta_s': 1.9320e+01,
    'Delta_m': 1.1121e+01,
    'gtilde': 4.7697e+00,
    'A_p': 4.6790e+00, 
    'a0': 0 + 0j, 
    'b0_dagger': 2.0439e+00 - 2.0229e+00j,
    'z_max': 50.0, 
    'num_points': 8192,
    'scaling_factor': 6.8425e-11,
    'vertical_shift': 1.2472e-09
}
# Mode 2
params2 = {
    'alpha_s': 3.2120e+01,
    'alpha_p': 3.6453e+01,  
    'Delta_s': 2.0507e+01, 
    'Delta_m': 8.4604e+00,
    'gtilde': 4.6937e+00,
    'A_p': 4.6790e+00, 
    'a0': 0 + 0j, 
    'b0_dagger': 4.9645e-01 - 3.8247e-01j,
    'z_max': 50.0, 
    'num_points': 8192,
    'scaling_factor': 6.8426e-11,
    'vertical_shift': 1.1810e-09
}
# Mode 3
params3 = {
    'alpha_s': 31.735, 
    'alpha_p': 36.134, 
    'Delta_s': 2.2122e+01, 
    'Delta_m': 6.6977e+00,
    'gtilde': 4.7697,
    'A_p': 4.6790, 
    'a0': 0 + 0j,
    'b0_dagger': 5.7993e-06 - 4.8773e-07j,
    'z_max': 50.0, 
    'num_points': 8192,
    'scaling_factor': 6.8426e-11,
    'vertical_shift': 1.2472e-09
}

# --- SIMULATION ---
z1, a1 = generate_single_mode(params1)
z2, a2 = generate_single_mode(params2)
z3, a3 = generate_single_mode(params3)

assert np.allclose(z1, z2) and np.allclose(z1, z3)
z = z1
dz = z[1] - z[0]

# --- CALCULATIONS ---
a_total = a1 + a2 + a3

# First-order coherence
taus, g1_11 = compute_g1_vs_tau(a1, a1); _, g1_22 = compute_g1_vs_tau(a2, a2)
_, g1_33 = compute_g1_vs_tau(a3, a3); _, g1_12 = compute_g1_vs_tau(a1, a2)
_, g1_13 = compute_g1_vs_tau(a1, a3); _, g1_23 = compute_g1_vs_tau(a2, a3)
g1_total = g1_11 + g1_22 + g1_33

# Second-order coherence
_, g2_11 = compute_g2_vs_tau(a1, a1); _, g2_22 = compute_g2_vs_tau(a2, a2)
_, g2_33 = compute_g2_vs_tau(a3, a3); _, g2_total = compute_g2_vs_tau(a_total, a_total)

# --- New g2 calculation from normalized g1 ---
# Take the absolute value of g1_total
g1_total_abs = np.abs(g1_total)
# Find the maximum value to normalize
max_g1_total = np.max(g1_total_abs)
# Normalize so the peak is 1
g1_total_normalized = g1_total_abs / (max_g1_total + eps)
# Apply the Siegert relation
g2_siegert = 1 + g1_total_normalized**2

# -- Another new g2 calculation
max_g2_total = np.max(g2_total)
g2_total_normalized = 2*(g2_total / (max_g2_total + eps))


# Power spectra
omega, S_total_arb = compute_power_spectrum(a_total, z, zero_pad_factor=8)
_, S1_arb = compute_power_spectrum(a1, z, zero_pad_factor=8)
_, S2_arb = compute_power_spectrum(a2, z, zero_pad_factor=8)
_, S3_arb = compute_power_spectrum(a3, z, zero_pad_factor=8)

# --- SCALING TO PHYSICAL UNITS ---
S1_W = S1_arb * params1['scaling_factor'] + params1['vertical_shift']
S2_W = S2_arb * params2['scaling_factor'] + params2['vertical_shift']
S3_W = S3_arb * params3['scaling_factor'] + params3['vertical_shift']
S_total_W = S1_arb * params1['scaling_factor'] + S2_arb * params2['scaling_factor'] + S3_arb * params3['scaling_factor'] + params3['vertical_shift']

delay_ns = (taus * dz) * z_to_ns
freq_mhz = (omega / (2 * np.pi)) * (1000 / z_to_ns)

# --- PLOTTING ---
plt.style.use('seaborn-v0_8-darkgrid')

# g1 plot
fig1, axs1 = plt.subplots(3, 2, figsize=(12, 15), sharex=True, sharey=True)
fig1.suptitle('First-Order Coherence |g$^{(1)}$(τ)|', fontsize=16)
axs1[0, 0].plot(delay_ns, np.abs(g1_11)); axs1[0, 0].set_title('Auto: g$^{(1)}$(a1, a1)')
axs1[1, 0].plot(delay_ns, np.abs(g1_22)); axs1[1, 0].set_title('Auto: g$^{(1)}$(a2, a2)')
axs1[2, 0].plot(delay_ns, np.abs(g1_33)); axs1[2, 0].set_title('Auto: g$^{(1)}$(a3, a3)')
axs1[0, 1].plot(delay_ns, np.abs(g1_12)); axs1[0, 1].set_title('Cross: g$^{(1)}$(a1, a2)')
axs1[1, 1].plot(delay_ns, np.abs(g1_13)); axs1[1, 1].set_title('Cross: g$^{(1)}$(a1, a3)')
axs1[2, 1].plot(delay_ns, np.abs(g1_total)); axs1[2, 1].set_title('Sum: |g$^{(1)}_{11}$+g$^{(1)}_{22}$+g$^{(1)}_{33}$|')
for ax in axs1.flat:
    ax.set(xlabel='Delay τ (ns)', ylabel='|g$^{(1)}$(τ)|'); ax.grid(True)
axs1[0,0].set_xlim(-100, 100)

# g2 plot
fig2, axs2 = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
fig2.suptitle('Second-Order Coherence g$^{(2)}$(τ)', fontsize=16)
axs2[0, 0].plot(delay_ns, g2_11); axs2[0, 0].set_title('Auto: g$^{(2)}$(a1, a1)')
axs2[0, 1].plot(delay_ns, g2_22); axs2[0, 1].set_title('Auto: g$^{(2)}$(a2, a2)')
axs2[1, 0].plot(delay_ns, g2_33); axs2[1, 0].set_title('Auto: g$^{(2)}$(a3, a3)')
axs2[1, 1].plot(delay_ns, g2_total); axs2[1, 1].set_title('Auto: g$^{(2)}$(a_total, a_total)')
for ax in axs2.flat:
    ax.set(xlabel='Delay τ (ns)', ylabel='g$^{(2)}$(τ)|'); ax.grid(True)
axs2[0,0].set_xlim(-100, 100)
#Siegert Relation: 1 + |g$^{(1)}_{norm}$|$^2
# New Siegert g2 plot
plt.figure(figsize=(12, 6))
plt.plot(delay_ns, g2_siegert)
plt.title("Siegert Relation: g$^{(2)}$(τ) = 1 + |g$^{(1)}$|$^2$")
plt.xlabel('Delay τ (ns)')
plt.ylabel('g$^{(2)}$(τ)')
plt.xlim(-100, 100)
plt.ylim(0, 2)
plt.grid(True)

# New normalised g2 plot
plt.figure(figsize=(12, 6))
plt.plot(delay_ns, g2_total_normalized)
plt.title("Auto: g$^{(2)}$(a_total, a_total)")
plt.xlabel('Delay τ (ns)')
plt.ylabel('g$^{(2)}$(τ)')
plt.xlim(-100, 100)
plt.ylim(0, 2)
plt.grid(True)

# Power spectrum plot
plt.figure(figsize=(12, 6))
plt.plot(freq_mhz, S_total_W, label='Total Field', linewidth=2)
plt.plot(freq_mhz, S1_W, '--', label="Mode 1")
plt.plot(freq_mhz, S2_W, '--', label="Mode 2")
plt.plot(freq_mhz, S3_W, ':', label="Mode 3")
plt.xlim(0, 800)
plt.ylim(bottom=0)
plt.xlabel('Frequency (MHz)'); plt.ylabel('Power (W)')
plt.title('Calibrated Power Spectra'); plt.legend(); plt.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

