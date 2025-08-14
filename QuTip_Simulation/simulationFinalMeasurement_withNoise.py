import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.fft import fft, fftshift, fftfreq


#NOT COMPLETED YET
# A small constant to prevent division by zero in normalization
eps = 1e-15

def generate_mode_with_langevin_noise(params):
    """
    Solves the stochastic differential equation (Langevin equation) for the system
    using the Euler-Maruyama method.
    dX/dz = M*X + N(z)
    """
    # Dissipation rates for the deterministic matrix M
    alpha_s = params['alpha_s']
    alpha_p = params['alpha_p']
    
    # Fluctuation (noise) strengths for the stochastic part
    alpha_s_e = params['alpha_s_e']
    alpha_s_i = params['alpha_s_i']
    
    # Other system parameters
    Delta_s = params['Delta_s']
    gtilde = params['gtilde']
    A_p = params['A_p']
    
    # Initial conditions for the mode amplitudes
    a0 = params['a0']
    b0_dagger = params['b0_dagger']
    
    # Mean values of the input noise fields (typically 0 for vacuum noise)
    a_in_e_mean = params['a_in_e']
    a_in_i_mean = params['a_in_i']
    b_in_mean = params['b_in']

    z_max = params['z_max']
    N = params['num_points']
    z = np.linspace(0, z_max, N)
    dz = z[1] - z[0]

    # The deterministic part of the evolution matrix uses the main alpha_s
    M = np.array([
        [-alpha_s / 2.0 + 1j * Delta_s, -1j * gtilde * A_p],
        [1j * gtilde * A_p, -alpha_p / 2.0]
    ], dtype=complex)

    # Initialize arrays to store the results
    a_z = np.empty(N, dtype=complex)
    bdag_z = np.empty(N, dtype=complex)
    a_z[0] = a0
    bdag_z[0] = b0_dagger

    # Step-by-step integration using Euler-Maruyama method
    for i in range(N - 1):
        # Current state vector
        v = np.array([a_z[i], bdag_z[i]])
        
        # 1. Calculate deterministic part of the step
        deterministic_step = M.dot(v) * dz
        
        # 2. Calculate the stochastic (noise) part of the step
        # Generate complex Gaussian random numbers for each noise channel
        w_e = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
        w_i = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
        w_p = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)

        # Noise terms from the Langevin equations, using separate alpha_s_e and alpha_s_i
        noise_a = (np.sqrt(2 * alpha_s_e) * (a_in_e_mean + w_e) + 
                   np.sqrt(2 * alpha_s_i) * (a_in_i_mean + w_i)) * np.sqrt(dz)
                   
        noise_b_dag = np.sqrt(alpha_p) * (np.conj(b_in_mean) + np.conj(w_p)) * np.sqrt(dz)

        stochastic_step = np.array([noise_a, noise_b_dag])

        # 3. Update the state vector
        v_next = v + deterministic_step + stochastic_step
        a_z[i+1] = v_next[0]
        bdag_z[i+1] = v_next[1]

    return z, a_z, bdag_z

def compute_g1_vs_tau(a1, a2):
    """
    Computes the first-order temporal coherence g^(1)(tau) between two complex fields.
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
    Computes the second-order temporal coherence g^(2)(tau) between two fields.
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
    Computes the power spectrum of a signal x sampled over z.
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
z_to_ns = 12.5 / np.pi

# Mode 1 will be at 0 MHz
params1 = {
    'alpha_s': 0.08,   # Total dissipation rate for the optical mode
    'alpha_s_e': 0.03, # Strength of the external noise channel
    'alpha_s_i': 0.03, # Strength of the internal noise channel
    'alpha_p': 1.0,
    'Delta_s': 0.0, 
    'gtilde': 0.5, 
    'A_p': 0.1,
    'a0': 0.09 + 0j, 
    'b0_dagger': 0.1 + 0j,
    'a_in_e': 0.0 + 0j, 
    'a_in_i': 0.0 + 0j, 
    'b_in': 0.0 + 0j, # Mean of noise fields
    'z_max': 50.0, 
    'num_points': 8192
}
# Mode 2 will be at +80 MHz
params2 = params1.copy()
params2.update({
    'alpha_s': 0.07,
    'alpha_s_e': 0.035,
    'alpha_s_i': 0.035,
    'Delta_s': 2.0, 
    'a0': 0.01 + 0j,
})
# Mode 3 will be at +160 MHz
params3 = params1.copy()
params3.update({
    'alpha_s': 0.07,
    'alpha_s_e': 0.035,
    'alpha_s_i': 0.035,
    'Delta_s': 4.0, 
    'a0': 0.005 + 0j,
})

# --- SIMULATION ---
z1, a1, b1 = generate_mode_with_langevin_noise(params1)
z2, a2, b2 = generate_mode_with_langevin_noise(params2)
z3, a3, b3 = generate_mode_with_langevin_noise(params3)

assert np.allclose(z1, z2) and np.allclose(z1, z3)
z = z1
dz = z[1] - z[0]

# --- CALCULATIONS ---
a_total = a1 + a2 + a3

# First-order coherence
taus, g1_11 = compute_g1_vs_tau(a1, a1)
_, g1_22 = compute_g1_vs_tau(a2, a2)
_, g1_33 = compute_g1_vs_tau(a3, a3)
_, g1_12 = compute_g1_vs_tau(a1, a2)
_, g1_13 = compute_g1_vs_tau(a1, a3)
_, g1_23 = compute_g1_vs_tau(a2, a3)

# Second-order coherence
_, g2_11 = compute_g2_vs_tau(a1, a1)
_, g2_22 = compute_g2_vs_tau(a2, a2)
_, g2_33 = compute_g2_vs_tau(a3, a3)
_, g2_total = compute_g2_vs_tau(a_total, a_total)

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

# Plot |g1(tau)|
fig1, axs1 = plt.subplots(3, 2, figsize=(12, 15), sharex=True, sharey=True)
fig1.suptitle('First-Order Coherence |g$^{(1)}$(τ)|', fontsize=16)
axs1[0, 0].plot(delay_ns, np.abs(g1_11)); axs1[0, 0].set_title('Auto: g$^{(1)}$(a1, a1)')
axs1[1, 0].plot(delay_ns, np.abs(g1_22)); axs1[1, 0].set_title('Auto: g$^{(1)}$(a2, a2)')
axs1[2, 0].plot(delay_ns, np.abs(g1_33)); axs1[2, 0].set_title('Auto: g$^{(1)}$(a3, a3)')
axs1[0, 1].plot(delay_ns, np.abs(g1_12)); axs1[0, 1].set_title('Cross: g$^{(1)}$(a1, a2)')
axs1[1, 1].plot(delay_ns, np.abs(g1_13)); axs1[1, 1].set_title('Cross: g$^{(1)}$(a1, a3)')
axs1[2, 1].plot(delay_ns, np.abs(g1_23)); axs1[2, 1].set_title('Cross: g$^{(1)}$(a2, a3)')
for ax in axs1.flat:
    ax.set(xlabel='Delay τ (ns)', ylabel='|g$^{(1)}$(τ)|')
    ax.grid(True)

# Plot g2(tau)
fig2, axs2 = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
fig2.suptitle('Second-Order Coherence g$^{(2)}$(τ)', fontsize=16)
axs2[0, 0].plot(delay_ns, g2_11); axs2[0, 0].set_title('Auto: g$^{(2)}$(a1, a1)')
axs2[0, 1].plot(delay_ns, g2_22); axs2[0, 1].set_title('Auto: g$^{(2)}$(a2, a2)')
axs2[1, 0].plot(delay_ns, g2_33); axs2[1, 0].set_title('Auto: g$^{(2)}$(a3, a3)')
axs2[1, 1].plot(delay_ns, g2_total); axs2[1, 1].set_title('Auto: g$^{(2)}$(a_total, a_total)')
for ax in axs2.flat:
    ax.set(xlabel='Delay τ (ns)', ylabel='g$^{(2)}$(τ)')
    ax.grid(True)
axs2[0,0].set_xlim(-20, 20)

# Plot spectra
plt.figure(figsize=(12, 6))
plt.plot(freq_mhz, S_total_W, label='Total Field (1+2+3)', linewidth=2)
plt.plot(freq_mhz, S1_W, '--', label='Mode 1 (0 MHz)')
plt.plot(freq_mhz, S2_W, '--', label='Mode 2 (+80 MHz)')
plt.plot(freq_mhz, S3_W, ':', label='Mode 3 (+160 MHz)')
plt.xlim(-20, 200)
plt.ylim(0, target_peak_power_watts * 1.1)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power (W)')
plt.title('Calibrated Power Spectra of Individual and Combined Modes')
plt.legend()
plt.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
