import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.fft import fft, fftshift, fftfreq

# Epsilon to prevent division by zero in calculations
eps = 1e-15

def generate_mode_by_matrix(params):
    """
    Solves dv/dz = M v with v=[a_s, b^dagger]^T using the matrix exponential method.
    """
    alpha_s = params['alpha_s']
    alpha_p = params['alpha_p']
    Delta_s = params['Delta_s']
    gtilde = params['gtilde']
    A_p = params['A_p']
    a0 = params['a0']
    b0 = params['b0_dagger']
    z_max = params['z_max']
    N = params['num_points']

    z = np.linspace(0, z_max, N)

    # Construct the evolution matrix M
    M = np.array([
        [-alpha_s / 2.0 + 1j * Delta_s, -1j * gtilde * A_p],
        [1j * gtilde * A_p, -alpha_p / 2.0]
    ], dtype=complex)

    # Initial condition vector
    v0 = np.array([a0, b0], dtype=complex)
    
    a_z = np.empty(N, dtype=complex)
    bdag_z = np.empty(N, dtype=complex)

    # Compute the evolution for each point z by applying exp(M*z)
    for i, zz in enumerate(z):
        T = expm(M * zz)  # Evolution operator
        v = T.dot(v0)
        a_z[i] = v[0]
        bdag_z[i] = v[1]

    return z, a_z, bdag_z

def compute_g1_multimode(modes, dt):
    """
    Computes g^(1)(tau) for a collection of modes, returning the delay axis in physical units.
    """
    num_modes = len(modes)
    if num_modes == 0:
        return np.array([]), np.array([])
    N = len(modes[0])

    taus_steps = np.arange(-N + 1, N)
    g1_values = np.zeros(len(taus_steps), dtype=complex)

    denominator = 0.0
    for j in range(num_modes):
        denominator += np.mean(np.abs(modes[j])**2)

    if abs(denominator) < eps:
        return taus_steps * dt, g1_values

    for i, tau_step in enumerate(taus_steps):
        numerator = 0 + 0j
        for j in range(num_modes):
            for k in range(num_modes):
                if tau_step >= 0:
                    aj_conj = np.conjugate(modes[j][:N - tau_step])
                    ak_shifted = modes[k][tau_step:]
                else:
                    aj_conj = np.conjugate(modes[j][-tau_step:])
                    ak_shifted = modes[k][:N + tau_step]
                
                if aj_conj.size > 0:
                    numerator += np.mean(aj_conj * ak_shifted)
        
        g1_values[i] = numerator / denominator

    return taus_steps * dt, g1_values

def compute_power_spectrum(x, dt, zero_pad_factor=4, window=True):
    """
    Computes the power spectrum (PSD) of a signal x sampled with time step dt.
    Returns frequency in Hz.
    """
    N = len(x)
    Npad = int(2**np.ceil(np.log2(N * zero_pad_factor)))
    
    if window:
        xw = x * np.hanning(N)
    else:
        xw = x
        
    X = fftshift(fft(xw, n=Npad))
    S = np.abs(X)**2
    freq_hz = fftshift(fftfreq(Npad, d=dt))
    return freq_hz, S

# --------------------------------------------------------------------------
# Main execution block
# --------------------------------------------------------------------------
if __name__ == "__main__":
    NUM_MODES = 5

    # --- PHYSICAL PARAMETERS ---
    c = 299792458
    n_core = 1.468
    v_g = c / n_core
    z_max_meters = 50.0

    # The model uses a normalized detuning `Delta_s`. We set the separation
    # of `Delta_s` to be 2.5 to match the simulation's parameters.
    delta_f_phys = 80e6
    delta_omega_norm = 2.5

    # Derive the physical time per unit of normalized z
    time_per_z_unit = delta_omega_norm / (2 * np.pi * delta_f_phys)
    L_char = v_g * time_per_z_unit
    z_max_norm = z_max_meters / L_char

    # --- SIMULATION PARAMETERS ---
    base_params = {
        'alpha_s': 2.0, 'alpha_p': 2.5, 'A_p': 1.0,
        'b0_dagger': 0.0 + 0j, 'z_max': z_max_norm, 'num_points': 8192
    }

    param_list = []
    for i in range(NUM_MODES):
        params = base_params.copy()
        params['Delta_s'] = 2.5 * (i) # 0, 2.5, 5.0, 7.5, 10.0
        params['gtilde'] = 0.8 + i * 0.05
        # Increase initial amplitude for later modes to make their peaks larger
        params['a0'] = 1.0 + i * 0.2 + 0j 
        param_list.append(params)

    # --- Generate the 5 modes ---
    modes = []
    z_norm = None
    print(f"Generating {NUM_MODES} modes...")
    for i, params in enumerate(param_list):
        print(f"  - Mode {i+1} with Delta_s = {params['Delta_s']:.1f} and a0 = {params['a0']:.2f}")
        z_current, a_current, _ = generate_mode_by_matrix(params)
        if z_norm is None:
            z_norm = z_current
        modes.append(a_current)
    print("...done.")

    # --- CONVERT TO TIME DOMAIN ---
    t = z_norm * time_per_z_unit
    dt = t[1] - t[0]

    # --- Perform Calculations ---
    delay_s, g1 = compute_g1_multimode(modes, dt)
    total_field = np.sum(modes, axis=0)
    freq_hz, S_total_arb = compute_power_spectrum(total_field, dt, zero_pad_factor=8)
    
    individual_spectra_arb = []
    for a_mode in modes:
        _, S_individual = compute_power_spectrum(a_mode, dt, zero_pad_factor=8, window=True)
        individual_spectra_arb.append(S_individual)

    # --- SCALING & PLOTTING ---
    target_peak_power_watts = 2e-12
    current_peak_power_arb = np.max(S_total_arb)
    power_scaling_factor = target_peak_power_watts / (current_peak_power_arb + eps)

    S_total_W = S_total_arb * power_scaling_factor
    individual_spectra_W = [s * power_scaling_factor for s in individual_spectra_arb]

    delay_ns = delay_s * 1e9
    freq_mhz = freq_hz / 1e6

    # --- Plotting Results ---
    plt.style.use('seaborn-v0_8-darkgrid')

    # Plot |g1(tau)|
    plt.figure(figsize=(8, 5))
    plt.plot(delay_ns, np.abs(g1))
    plt.xlabel('Delay τ (ns)')
    plt.ylabel('|g$^{(1)}$(τ)|')
    plt.title(f'First-Order Coherence Magnitude for {NUM_MODES} Modes')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Plot Power Spectra
    plt.figure(figsize=(10, 6))
    plt.plot(freq_mhz, S_total_W, label='Total Field', linewidth=2.5, color='black')
    colors = plt.cm.viridis(np.linspace(0, 1, NUM_MODES))
    for i, S_individual in enumerate(individual_spectra_W):
        freq_label = param_list[i]['Delta_s'] / 2.5 * 80
        plt.plot(freq_mhz, S_individual, '--', label=f'Mode {i+1} ({freq_label:.0f} MHz)', color=colors[i])
    
    plt.xlim(-40, (NUM_MODES) * 80 + 40)
    plt.ylim(0, target_peak_power_watts * 1.1)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (W)')
    plt.title('Calibrated Power Spectra of Total and Individual Fields')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.show()
