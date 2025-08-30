'''
This script was just used to make sure the analytical solution is the same as the matrix exponential solution.
It is.
'''





import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.fft import fft, fftshift, fftfreq

# A small constant to prevent division by zero in normalization
eps = 1e-15

def generate_mode_by_matrix(params):
    """
    Solves the differential equation dv/dz = M*v for a coupled-mode system,
    where v = [a_s, b^dagger]^T, using the matrix exponential method.

    Args:
        params (dict): A dictionary containing the simulation parameters:
            'alpha_s': Optical loss for the signal mode.
            'alpha_p': Loss for the phonon mode.
            'Delta_s': Detuning of the signal from its carrier frequency.
            'gtilde': Coupling strength.
            'A_p': Pump amplitude.
            'a0': Initial complex amplitude of the signal mode.
            'b0_dagger': Initial complex amplitude of the phonon creation operator.
            'z_max': The maximum propagation distance.
            'num_points': The number of points to simulate along z.

    Returns:
        tuple: A tuple containing:
            - z (np.ndarray): The array of z positions.
            - a_z (np.ndarray): The complex amplitude of the signal mode vs. z.
            - bdag_z (np.ndarray): The complex amplitude of the phonon creation operator vs. z.
    """
    alpha_s = params['alpha_s']
    alpha_p = params['alpha_p']
    Delta_s = params['Delta_s']
    gtilde = params['gtilde']
    A_p = params['A_p']
    a0 = params['a0']
    b0_dagger = params['b0_dagger']
    z_max = params['z_max']
    N = params['num_points']

    z = np.linspace(0, z_max, N)

    # The coupling matrix M defines the system of linear differential equations.
    M = np.array([
        [-alpha_s / 2.0 + 1j * Delta_s, -1j * gtilde * A_p],
        [1j * gtilde * A_p, -alpha_p / 2.0]
    ], dtype=complex)

    # Initial condition vector
    v0 = np.array([a0, b0_dagger], dtype=complex)
    
    a_z = np.empty(N, dtype=complex)
    bdag_z = np.empty(N, dtype=complex)

    # Solve for v(z) = exp(M*z) * v(0) at each point z.
    for i, zz in enumerate(z):
        T = expm(M * zz)  # Compute the matrix exponential
        v = T.dot(v0)    # Apply the transformation to the initial vector
        a_z[i] = v[0]
        bdag_z[i] = v[1]

    return z, a_z, bdag_z

def compute_g1_vs_tau(a1, a2):
    """
    Computes the first-order temporal coherence g^(1)(tau) between two complex fields.
    g^(1)(tau) = <a1*(t) a2(t+tau)> / sqrt(<|a1|^2><|a2|^2>)

    Args:
        a1 (np.ndarray): The first complex field array.
        a2 (np.ndarray): The second complex field array (must be same length as a1).

    Returns:
        tuple: A tuple containing:
            - taus (np.ndarray): The integer sample delays.
            - g1 (np.ndarray): The complex-valued g^(1) for each delay.
    """
    N = len(a1)
    taus = np.arange(-N + 1, N)
    g1 = np.zeros(len(taus), dtype=complex)
    # Normalize by the geometric mean of the average intensities
    norm = np.sqrt(np.mean(np.abs(a1)**2) * np.mean(np.abs(a2)**2)) + eps

    for i, tau in enumerate(taus):
        if tau >= 0:
            # Positive delay: correlate a1 with a time-advanced a2
            x1 = a1[:N - tau]
            x2 = a2[tau:]
        else:
            # Negative delay: correlate a time-advanced a1 with a2
            x1 = a1[-tau:]
            x2 = a2[:N + tau]
        g1[i] = np.mean(np.conj(x1) * x2) / norm

    return taus, g1

def compute_g2_vs_tau(a1, a2):
    """
    Computes the second-order temporal coherence g^(2)(tau) between two fields.
    g^(2)(tau) = <I1(t) I2(t+tau)> / (<I1><I2>)
    For auto-correlation, call with a1=a2.

    Args:
        a1 (np.ndarray): The first complex field array.
        a2 (np.ndarray): The second complex field array.

    Returns:
        tuple: A tuple containing:
            - taus (np.ndarray): The integer sample delays.
            - g2 (np.ndarray): The real-valued g^(2) for each delay.
    """
    N = len(a1)
    # Intensities are the squared magnitude of the fields
    I1 = np.abs(a1)**2
    I2 = np.abs(a2)**2

    taus = np.arange(-N + 1, N)
    g2 = np.zeros(len(taus), dtype=float)
    # Normalize by the product of the mean intensities
    norm = np.mean(I1) * np.mean(I2) + eps

    for i, tau in enumerate(taus):
        if tau >= 0:
            x1 = I1[:N - tau]
            x2 = I2[tau:]
        else:
            x1 = I1[-tau:]
            x2 = I2[:N + tau]
        # The correlation of real-valued intensities is real
        g2[i] = np.mean(x1 * x2) / norm

    return taus, g2


def compute_power_spectrum(x, z, zero_pad_factor=4, window=True):
    """
    Computes the power spectrum of a signal x sampled over z.

    Args:
        x (np.ndarray): The signal array (complex or real).
        z (np.ndarray): The sampling points (must be equally spaced).
        zero_pad_factor (int, optional): Factor to pad the signal with zeros
                                         to improve frequency resolution. Defaults to 4.
        window (bool, optional): If True, apply a Hanning window to reduce
                                 spectral leakage. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - omega (np.ndarray): The angular frequency axis (in rad / z-unit).
            - S (np.ndarray): The Power Spectral Density (PSD).
    """
    dt = z[1] - z[0]  # Sample spacing in z-units
    N = len(x)
    # Determine padded length, rounding up to the next power of 2 for FFT efficiency
    Npad = int(2**np.ceil(np.log2(N * zero_pad_factor)))
    
    if window:
        w = np.hanning(N)
        xw = x * w
    else:
        xw = x
        
    # Compute the FFT and shift the zero-frequency component to the center
    X = fftshift(fft(xw, n=Npad))
    S = np.abs(X)**2
    
    # Calculate the corresponding frequency axis
    freq = fftshift(fftfreq(Npad, d=dt))  # Frequencies in cycles per z-unit
    omega = 2 * np.pi * freq             # Convert to angular frequency (rad / z-unit)
    
    return omega, S

# ------------------------------
# --- PARAMETER DEFINITIONS ---
# ------------------------------
# We want peaks at 0, 80, and 160 MHz.
# The separation is 80 MHz. Let's set Delta_s separation to 2.0.
# The scaling factor calculation remains the same:
# 80 MHz = (Delta_s_sep / (2*pi*z_to_ns)) * 1000
# 80 = (2.0 / (2*pi*z_to_ns)) * 1000  => z_to_ns = 2000 / (160*pi) = 12.5/pi
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
z1, a1, b1 = generate_mode_by_matrix(params1)
z2, a2, b2 = generate_mode_by_matrix(params2)
z3, a3, b3 = generate_mode_by_matrix(params3)

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

# Power spectra (in arbitrary units first)
omega, S_total_arb = compute_power_spectrum(a_total, z, zero_pad_factor=8)
_, S1_arb = compute_power_spectrum(a1, z, zero_pad_factor=8)
_, S2_arb = compute_power_spectrum(a2, z, zero_pad_factor=8)
_, S3_arb = compute_power_spectrum(a3, z, zero_pad_factor=8)

# --- SCALING TO PHYSICAL UNITS ---
# Scale Power Spectrum to Watts
target_peak_power_watts = 2e-12
current_peak_power_arb = np.max(S_total_arb)
power_scaling_factor = target_peak_power_watts / current_peak_power_arb

S_total_W = S_total_arb * power_scaling_factor
S1_W = S1_arb * power_scaling_factor
S2_W = S2_arb * power_scaling_factor
S3_W = S3_arb * power_scaling_factor

# Scale axes for plotting
delay_ns = (taus * dz) * z_to_ns
freq_mhz = (omega / (2 * np.pi)) * (1000 / z_to_ns) # f = omega/(2pi), t_unit = z_to_ns

# --- PLOTTING ---
plt.style.use('seaborn-v0_8-darkgrid')

# Plot |g1(tau)| in subplots
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

# Plot g2(tau) in subplots
fig2, axs2 = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
fig2.suptitle('Second-Order Coherence g$^{(2)}$(τ)', fontsize=16)

axs2[0, 0].plot(delay_ns, g2_11); axs2[0, 0].set_title('Auto: g$^{(2)}$(a1, a1)')
axs2[0, 1].plot(delay_ns, g2_22); axs2[0, 1].set_title('Auto: g$^{(2)}$(a2, a2)')
axs2[1, 0].plot(delay_ns, g2_33); axs2[1, 0].set_title('Auto: g$^{(2)}$(a3, a3)')
axs2[1, 1].plot(delay_ns, g2_total); axs2[1, 1].set_title('Auto: g$^{(2)}$(a_total, a_total)')

for ax in axs2.flat:
    ax.set(xlabel='Delay τ (ns)', ylabel='g$^{(2)}$(τ)')
    ax.grid(True)
axs2[0,0].set_xlim(-20, 20) # Zoom in near tau=0

# Plot spectra with scaled power axis
plt.figure(figsize=(12, 6))
plt.plot(freq_mhz, S_total_W, label='Total Field (1+2+3)', linewidth=2)
plt.plot(freq_mhz, S1_W, '--', label='Mode 1 (0 MHz)')
plt.plot(freq_mhz, S2_W, '--', label='Mode 2 (+80 MHz)')
plt.plot(freq_mhz, S3_W, ':', label='Mode 3 (+160 MHz)')
plt.xlim(-20, 200) # Adjust plot limits for positive spectrum
plt.ylim(0, target_peak_power_watts * 1.1) # Set y-limit slightly above the peak
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power (W)')
plt.title('Calibrated Power Spectra of Individual and Combined Modes')
plt.legend()
plt.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
