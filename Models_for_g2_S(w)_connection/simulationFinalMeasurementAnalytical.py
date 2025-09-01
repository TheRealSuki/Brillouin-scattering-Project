import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.fft import fft, fftshift, fftfreq
import pandas as pd

# A small constant to prevent division by zero in normalization
eps = 1e-15


def dbm_to_watts(dbm):
    return 10 ** ((dbm - 30) / 10)       

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


def compute_g2_unnormalized(a, method="direct",  target_mean_I=1.0):
    """
    Compute unnormalized second-order correlation function G2(tau) 
    from a complex field array, with rescaling for numerical stability.
    
    Parameters
    ----------
    a : np.ndarray
        Complex field amplitudes (1D array).
    method : str
        Method to compute correlation: 'direct' (default), 'correlate', or 'fft'.
    target_mean_I : float
        Desired mean intensity for rescaling (default = 1.0).
        
    Returns
    -------
    G2 : np.ndarray
        Unnormalized second-order correlation function.
    taus : np.ndarray
        Array of time delays (in sample units).
    scale_factor : float
        Factor applied to rescale input.
    """
    # --- Step 1: rescale field ---
    I = np.abs(a)**2
    orig_mean_I = np.mean(I)
    if orig_mean_I == 0:
        raise ValueError("Input field has zero mean intensity, cannot rescale.")
    scale_factor = np.sqrt(target_mean_I / orig_mean_I)
    a_scaled = a * scale_factor

    I_scaled = np.abs(a_scaled)**2

    # --- Step 2: compute G2 ---
    if method == "direct":
        N = len(I_scaled)
        G2 = np.array([np.mean(I_scaled[:N-t] * I_scaled[t:]) for t in range(N)])
        taus = np.arange(N)

    elif method == "correlate":
        G2 = np.correlate(I_scaled, I_scaled, mode="full")
        mid = len(G2) // 2
        G2 = G2[mid:] / len(I_scaled)
        taus = np.arange(len(G2))

    elif method == "fft":
        F = np.fft.fft(I_scaled, n=2*len(I_scaled))
        G2 = np.fft.ifft(F * np.conj(F)).real[:len(I_scaled)]
        G2 /= len(I_scaled)
        taus = np.arange(len(G2))

    else:
        raise ValueError("Method must be 'direct', 'correlate', or 'fft'.")

    return taus, G2


def compute_g2_vs_tau(a1, a):
    N = len(a)
    taus = np.arange(-N + 1, N)
    g2 = np.zeros(len(taus), dtype=np.complex128)

    for i, tau in enumerate(taus):
        if tau >= 0:
            x1 = a[:N - tau]
            x2 = a[tau:]
        else:
            x1 = a[-tau:]
            x2 = a[:N + tau]

        g2[i] = np.mean(np.conj(x2) * np.conj(x1) * x1 * x2)

    return taus, g2
def compute_g2_with_cross_terms(a1, a2, a3):
    """
    Computes the second-order temporal coherence g^(2)(tau) for the combined field a1 + a2.
    
    This version correctly includes interference effects (cross terms) by first
    calculating the total intensity of the superposed fields.
    """
    # 1. Create the total field by summing the individual fields.
    # This is the crucial step to include interference.
    a_total = a1 + a2 + a3

    # 2. Calculate the total intensity from the combined field.
    I_total = np.abs(a_total)**2
    
    N = len(I_total)
    taus = np.arange(-N + 1, N)
    g2 = np.zeros(len(taus), dtype=float)
    
    # 3. Compute the intensity AUTOCORRELATION of the total intensity.
    # The logic is the same as your original function, but applied to I_total.
    for i, tau in enumerate(taus):
        if tau >= 0:
            # Slices of I_total for positive delay tau
            x1 = I_total[:N - tau]
            x2 = I_total[tau:]
        else:
            # Slices of I_total for negative delay tau
            x1 = I_total[-tau:]
            x2 = I_total[:N + tau]
            
        # Per-slice normalization, which is robust for non-stationary signals.
        norm = np.mean(x1) * np.mean(x2) + eps
        
        # Calculate the normalized correlation for this tau
        g2[i] = np.mean(x1 * x2) / norm
        
    return taus, g2


def compute_g2_vs_tau_3modes(a1, a2, a3):
    N = len(a1)
    taus = np.arange(-N + 1, N)
    g2 = np.zeros(len(taus), dtype=np.complex128)

    for i, tau in enumerate(taus):
        if tau >= 0:
            x1_1, x2_1 = a1[:N - tau], a1[tau:]
            x1_2, x2_2 = a2[:N - tau], a2[tau:]
            x1_3, x2_3 = a3[:N - tau], a3[tau:]
        else:
            x1_1, x2_1 = a1[-tau:], a1[:N + tau]
            x1_2, x2_2 = a2[-tau:], a2[:N + tau]
            x1_3, x2_3 = a3[-tau:], a3[:N + tau]

        g2[i] = np.mean(
            np.conj(x2_3) * np.conj(x2_2) * np.conj(x2_1) *
            np.conj(x1_3) * np.conj(x1_2) * np.conj(x1_1) *
            x1_1 * x1_2 * x1_3 *
            x2_1 * x2_2 * x2_3
        )

    return taus, g2

def compute_g2_vs_tau_3modes_explicit(a1, a2, a3):
    """
    Explicit 3-mode g^(2)(tau).
    Computes all 81 contributions:
    G2(tau) = sum_{i,j,k,l} < a_i^*(t) a_j^*(t+tau) a_k(t+tau) a_l(t) >
    This is mathematically correct but computationally much slower than
    correlating the total intensity directly.
    """
    N = len(a1)
    taus = np.arange(-N + 1, N)
    g2_numerator = np.zeros(len(taus), dtype=np.complex128)

    modes = [a1, a2, a3]

    for ti, tau in enumerate(taus):
        if tau >= 0:
            segs = [m[:N - tau] for m in modes]     # at time t
            segs_tau = [m[tau:] for m in modes] # at time t+tau
        else:
            segs = [m[-tau:] for m in modes]
            segs_tau = [m[:N + tau] for m in modes]

        acc = 0.0
        # loop over i,j,k,l = 0,1,2
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        acc += np.mean(
                            np.conj(segs[i]) * np.conj(segs_tau[j]) *
                            segs_tau[k] * segs[l]
                        )
        g2_numerator[ti] = acc

    return taus, g2_numerator




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
    'alpha_s': 3.1962e+01, 
    'alpha_p': 3.6764e+01, 
    'Delta_s': 2.2578e+01, 
    'Delta_m': 6.7804e+00,
    'gtilde': 4.7743e+00,
    'A_p': 4.6790e+00, 
    'a0': 0 + 0j,
    'b0_dagger': 5.1164e-02 -2.6692e-04j,
    'z_max': 50.0, 
    'num_points': 8192,
    'scaling_factor': 6.8425e-11,
    'vertical_shift': 1.1777e-09
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
_, g2_33 = compute_g2_vs_tau(a3, a3); _, g2_total = compute_g2_with_cross_terms(a1, a2, a3)
g2_total_alt = g2_11 + g2_22 + g2_33

#Alternative g2 calculation using unnormalized method
taus_2, g2_11_unnorm = compute_g2_unnormalized(a1)
taus_2, g2_22_unnorm = compute_g2_unnormalized(a2)
taus_2, g2_33_unnorm = compute_g2_unnormalized(a3)
taus_2, g2_total_unnorm = compute_g2_unnormalized(a_total)

#More alternative g2 calculation using explicit 3-mode method:
#taus_3, g2_total_explicit = compute_g2_vs_tau_3modes_explicit(a1, a2, a3)
#Another alternative g2 calculation using 3-mode method:
#taus_4, g2_total_3mode = compute_g2_vs_tau_3modes(a1, a2, a3)





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

delay_ns_2 = (taus_2 * dz) * z_to_ns
freq_mhz_2 = (omega / (2 * np.pi)) * (1000 / z_to_ns)

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


# g2 plot
fig3, axs3 = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
fig3.suptitle('Second-Order Coherence g$^{(2)}$(τ)', fontsize=16)
axs3[0, 0].plot(delay_ns_2, g2_11_unnorm); axs3[0, 0].set_title('Auto: g$^{(2)}$(a1, a1)')
axs3[0, 1].plot(delay_ns_2, g2_22_unnorm); axs3[0, 1].set_title('Auto: g$^{(2)}$(a2, a2)')
axs3[1, 0].plot(delay_ns_2, g2_33_unnorm); axs3[1, 0].set_title('Auto: g$^{(2)}$(a3, a3)')
axs3[1, 1].plot(delay_ns_2, g2_total_unnorm); axs3[1, 1].set_title('Auto: g$^{(2)}$(a_total, a_total)')
for ax in axs3.flat:
    ax.set(xlabel='Delay τ (ns)', ylabel='g$^{(2)}$(τ)|'); ax.grid(True)
axs3[0,0].set_xlim(-100, 100)
'''

delay_ns_3 = (taus_3 * dz) * z_to_ns
delay_ns_4 = (taus_4 * dz) * z_to_ns

# g2 plot
plt.figure(figsize=(12, 6))
plt.plot(delay_ns_3, g2_total_explicit)
plt.title("New g2 explicit Version 1.")
plt.xlabel('Delay τ (ns)')
plt.ylabel('g$^{(2)}$(τ)')
plt.xlim(-100, 100)
plt.ylim(0, 2)
plt.grid(True)


# g2 plot
plt.figure(figsize=(12, 6))
plt.plot(delay_ns_4, g2_total_3mode)
plt.title("New g2 explicit Version 2.")
plt.xlabel('Delay τ (ns)')
plt.ylabel('g$^{(2)}$(τ)')
plt.xlim(-100, 100)
plt.ylim(0, 2)
plt.grid(True)
'''


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



# New normalised g1 plot
plt.figure(figsize=(12, 6))
plt.plot(delay_ns, g1_total_normalized)
plt.title("Sum: g$^{(1)}_{11}$+g$^{(1)}_{22}$+g$^{(1)}_{33}$")
plt.xlabel('Delay τ (ns)')
plt.ylabel('g$^{(1)}$(τ)')
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

# New normalised g2 plot
plt.figure(figsize=(12, 6))
plt.plot(delay_ns, g2_total_alt)
plt.title("Auto: g$^{(2)}$")
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


#Plotting power spectrum and fit together.

df = pd.read_csv('/home/apolloin/Desktop/Brillouin_Scattering/Brillouin-scattering-Project/Models_for_g2_S(w)_connection/heterodyne_fit_curve.csv')
# The columns are 'fit_power_dBm' and 'frequency_Hz'
power_dbm_plot = df['fit_power_dBm'].values
freq_hz_plot = df['frequency_Hz'].values
freq_mhz_plot = freq_hz_plot / 1e6
power_watt_plot = dbm_to_watts(power_dbm_plot)


# Power spectrum plot
plt.figure(figsize=(12, 6))
plt.plot(freq_mhz_plot, power_watt_plot, 'k.', label='Experimental Data', linewidth=1, color='green')

plt.plot(freq_mhz, S_total_W, label='Total Field', linewidth=2.5, color='blue')
plt.plot(freq_mhz, S1_W, '--', label="Mode 1")
plt.plot(freq_mhz, S2_W, '--', label="Mode 2")
plt.plot(freq_mhz, S3_W, ':', label="Mode 3")
plt.xlim(0, 800)
plt.ylim(bottom=0)
plt.xlabel('Frequency (MHz)'); plt.ylabel('Power (W)')
plt.title('Calibrated Power Spectra'); plt.legend(); plt.grid(True)


plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()
