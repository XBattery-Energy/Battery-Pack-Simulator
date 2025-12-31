"""
Multi C-Rate Discharge Simulation: Compare 1C, 3C, and 6C discharge curves
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(parent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from sil_bms.pc_simulator.plant.cell_model import LiFePO4Cell
except ImportError:
    from cell_model import LiFePO4Cell


def run_discharge_simulation(
    capacity_ah: float,
    initial_soc: float,
    target_soc: float,
    c_rate: float,
    temperature_c: float,
    dt_ms: float = 10.0
):
    """
    Run discharge simulation for a given C-rate.
    
    Returns:
        Dictionary with time, voltage, soc, ocv data
    """
    # Create cell
    cell = LiFePO4Cell(
        capacity_ah=capacity_ah,
        initial_soc=initial_soc,
        temperature_c=temperature_c,
        ambient_temp_c=temperature_c
    )
    
    # Calculate discharge current (negative for discharge)
    discharge_current_a = -c_rate * capacity_ah
    discharge_current_ma = discharge_current_a * 1000.0
    
    # Data storage
    time_data = []
    voltage_data = []
    soc_data = []
    ocv_data = []
    
    step = 0
    elapsed_time = 0.0
    
    # Run simulation until target SOC is reached
    while cell._soc > target_soc:
        # Update cell
        voltage_mv, soc_pct = cell.update(
            current_ma=discharge_current_ma,
            dt_ms=dt_ms,
            ambient_temp_c=temperature_c
        )
        
        # Store data
        elapsed_time += dt_ms / 1000.0
        time_data.append(elapsed_time)
        voltage_data.append(voltage_mv / 1000.0)  # Convert to V
        soc_data.append(soc_pct)
        ocv_data.append(cell.get_ocv(current_direction=-1))  # Discharge OCV
        
        step += 1
        
        # Safety check (increase limit for low C-rates)
        max_steps = 500000 if c_rate <= 1.0 else 200000
        if step > max_steps:
            print(f"WARNING: Maximum steps reached for {c_rate}C discharge")
            break
    
    return {
        'time': np.array(time_data),
        'voltage': np.array(voltage_data),
        'soc': np.array(soc_data),
        'ocv': np.array(ocv_data),
        'c_rate': c_rate
    }


def plot_multi_crate_comparison():
    """Run simulations at different C-rates and plot comparison."""
    
    # Simulation parameters
    capacity_ah = 100.0
    initial_soc = 1.0  # 100%
    target_soc = 0.05  # 5%
    temperature_c = 25.0
    dt_ms = 10.0
    
    # C-rates to test
    c_rates = [1.0, 3.0, 6.0]
    colors = ['blue', 'green', 'red']
    linestyles = ['-', '--', '-.']
    
    print("=" * 80)
    print("Multi C-Rate Discharge Simulation")
    print("=" * 80)
    print(f"Capacity: {capacity_ah} Ah")
    print(f"Initial SOC: {initial_soc * 100:.0f}%")
    print(f"Target SOC: {target_soc * 100:.0f}%")
    print(f"Temperature: {temperature_c:.1f}°C")
    print(f"C-rates: {c_rates}")
    print("=" * 80)
    
    # Run simulations for each C-rate
    results = {}
    for c_rate in c_rates:
        print(f"\nRunning {c_rate}C discharge simulation...")
        results[c_rate] = run_discharge_simulation(
            capacity_ah=capacity_ah,
            initial_soc=initial_soc,
            target_soc=target_soc,
            c_rate=c_rate,
            temperature_c=temperature_c,
            dt_ms=dt_ms
        )
        print(f"  Completed: {len(results[c_rate]['time'])} steps, "
              f"Duration: {results[c_rate]['time'][-1]:.1f}s, "
              f"Final Voltage: {results[c_rate]['voltage'][-1]:.3f}V")
    
    # Create comparison plot
    print("\nGenerating comparison plot...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    # Plot OCV curve (reference, only once)
    if len(results) > 0:
        first_result = list(results.values())[0]
        ax.plot(first_result['soc'], first_result['ocv'], 
               'k--', linewidth=1.5, alpha=0.6, label='OCV (Reference)', zorder=1)
    
    # Plot terminal voltage for each C-rate
    for i, c_rate in enumerate(c_rates):
        result = results[c_rate]
        ax.plot(result['soc'], result['voltage'], 
               color=colors[i], linestyle=linestyles[i], 
               linewidth=2.5, label=f'{c_rate}C Discharge', 
               marker='o', markersize=2, zorder=2+i)
    
    # Formatting
    ax.set_xlabel('State of Charge (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Voltage (V)', fontsize=13, fontweight='bold')
    ax.set_title('LiFePO₄ Cell: Voltage vs SOC at Different C-Rates\n'
                 f'Discharge from {initial_soc*100:.0f}% to {target_soc*100:.0f}% SOC',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.set_xlim([target_soc * 100 - 2, initial_soc * 100 + 2])
    ax.invert_xaxis()  # SOC decreases during discharge (left to right)
    ax.set_ylim([2.4, 3.7])
    
    # Add annotation for flat plateau region
    ax.axvspan(20, 80, alpha=0.1, color='gray', label='Typical LFP Flat Plateau Region')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save plot
    filename = f'multi_crate_discharge_{initial_soc*100:.0f}to{target_soc*100:.0f}soc.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {filename}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Simulation Summary")
    print("=" * 80)
    for c_rate in c_rates:
        result = results[c_rate]
        print(f"\n{c_rate}C Discharge:")
        print(f"  Duration: {result['time'][-1]:.1f}s ({result['time'][-1]/60:.2f} min)")
        print(f"  Initial Voltage: {result['voltage'][0]:.3f}V")
        print(f"  Final Voltage: {result['voltage'][-1]:.3f}V")
        print(f"  Voltage Drop: {result['voltage'][0] - result['voltage'][-1]:.3f}V")
        # Calculate energy using trapezoidal integration
        energy_wh = np.trapezoid(result['voltage'], result['time']) * abs(c_rate * capacity_ah) / 3600
        print(f"  Energy Discharged: {energy_wh:.2f} Wh")
    print("=" * 80)
    
    # Show plot
    plt.show()
    
    return results


if __name__ == '__main__':
    results = plot_multi_crate_comparison()

