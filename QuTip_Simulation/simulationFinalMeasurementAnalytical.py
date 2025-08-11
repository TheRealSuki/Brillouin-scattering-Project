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
    # Map parameters from the existing structure to the analytical function's needs
    # As requested: k_j = -i * gtilde * A_p
    k_j = -1j * p['gtilde'] * p['A_p']
    alpha_m = p['alpha_p'] # Map phonon loss to alpha_m

    # The analytical solution from the user's provided function
    chi_s = -p['alpha_s']/2 + 1j * p['Delta_s']
    chi_m = -alpha_m/2

    A = (chi_s + chi_m) / 2
    # Use scimath.sqrt to handle potential negative numbers inside the square root
    D = np.lib.scimath.sqrt((chi_s - chi_m)**2 + 4 * k_j * np.conj(k_j))
    lambda_p = A + D/2
    lambda_m = A - D/2

    P_denom = chi_s - lambda_p
    Q_denom = chi_s - lambda_m
    
    # Avoid division by zero
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

def compute_power_spectrum(x, z, zero_pad_factor=4, window=True):
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

# Mode 1 will be at 0 MHz
params1 = {
    'alpha_s': 0.08, 
    'alpha_p': 1, 
    'Delta_s': 0, 
    'gtilde': 0.5,
    'A_p': 0.1, 
    'a0': 0.09 + 0j, 
    'b0_dagger': 0.1 + 0j,
    'z_max': 50.0, 
    'num_points': 8192
}
# Mode 2 will be at +80 MHz
params2 = params1.copy()
params2.update({
    'alpha_s': 0.07,
    'alpha_p': 1,  
    'Delta_s': 2.0, 
    'a0': 0.01 + 0j, 
    'gtilde': 0.5,
    'b0_dagger': 0.1 + 0j
})
# Mode 3 will be at +160 MHz
params3 = params1.copy()
params3.update({
    'alpha_s': 0.07, 
    'alpha_p': 1, 
    'Delta_s': 4.0, 
    'a0': 0.005 + 0j,
    'gtilde': 0.5, 
    'b0_dagger': 0.1 + 0j
})

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

# Second-order coherence
_, g2_11 = compute_g2_vs_tau(a1, a1); _, g2_22 = compute_g2_vs_tau(a2, a2)
_, g2_33 = compute_g2_vs_tau(a3, a3); _, g2_total = compute_g2_vs_tau(a_total, a_total)

# Power spectra
omega, S_total_arb = compute_power_spectrum(a_total, z, zero_pad_factor=8)
_, S1_arb = compute_power_spectrum(a1, z, zero_pad_factor=8)
_, S2_arb = compute_power_spectrum(a2, z, zero_pad_factor=8)
_, S3_arb = compute_power_spectrum(a3, z, zero_pad_factor=8)

# --- SCALING TO PHYSICAL UNITS ---
target_peak_power_watts = 2e-12
current_peak_power_arb = np.max(S_total_arb)
power_scaling_factor = target_peak_power_watts / (current_peak_power_arb + eps)

S_total_W = S_total_arb * power_scaling_factor
S1_W = S1_arb * power_scaling_factor
S2_W = S2_arb * power_scaling_factor
S3_W = S3_arb * power_scaling_factor

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
axs1[2, 1].plot(delay_ns, np.abs(g1_23)); axs1[2, 1].set_title('Cross: g$^{(1)}$(a2, a3)')
for ax in axs1.flat:
    ax.set(xlabel='Delay τ (ns)', ylabel='|g$^{(1)}$(τ)|'); ax.grid(True)

# g2 plot
fig2, axs2 = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
fig2.suptitle('Second-Order Coherence g$^{(2)}$(τ)', fontsize=16)
axs2[0, 0].plot(delay_ns, g2_11); axs2[0, 0].set_title('Auto: g$^{(2)}$(a1, a1)')
axs2[0, 1].plot(delay_ns, g2_22); axs2[0, 1].set_title('Auto: g$^{(2)}$(a2, a2)')
axs2[1, 0].plot(delay_ns, g2_33); axs2[1, 0].set_title('Auto: g$^{(2)}$(a3, a3)')
axs2[1, 1].plot(delay_ns, g2_total); axs2[1, 1].set_title('Auto: g$^{(2)}$(a_total, a_total)')
for ax in axs2.flat:
    ax.set(xlabel='Delay τ (ns)', ylabel='g$^{(2)}$(τ)|'); ax.grid(True)
axs2[0,0].set_xlim(-20, 20)

# Power spectrum plot
plt.figure(figsize=(12, 6))
plt.plot(freq_mhz, S_total_W, label='Total Field', linewidth=2)
plt.plot(freq_mhz, S1_W, '--', label=f"Mode 1 ({params1['Delta_s']*40:.0f} MHz)")
plt.plot(freq_mhz, S2_W, '--', label=f"Mode 2 ({params2['Delta_s']*40:.0f} MHz)")
plt.plot(freq_mhz, S3_W, ':', label=f"Mode 3 ({params3['Delta_s']*40:.0f} MHz)")
plt.xlim(-20, 200)
plt.ylim(0, target_peak_power_watts * 1.1)
plt.xlabel('Frequency (MHz)'); plt.ylabel('Power (W)')
plt.title('Calibrated Power Spectra'); plt.legend(); plt.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()
