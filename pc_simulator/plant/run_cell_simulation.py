"""Generic script for running pack charge/discharge simulations from command line.
This simulates a 16S battery pack for BMS testing."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from sil_bms.pc_simulator.plant.pack_model import BatteryPack16S

def run_simulation(mode='discharge', current_amp=1.0, duration_sec=None, target_soc_pct=None,
                   initial_soc_pct=100.0, dt_ms=100.0, 
                   temperature_c=25.0, save_plot=False, plot_filename=None, save_csv=True, csv_filename=None):
    """
    Run pack charge/discharge simulation with real-time stepping.
    
    Args:
        mode: 'charge' or 'discharge'
        current_amp: Current in Amperes (positive value)
        duration_sec: Duration in seconds (if None, use target_soc_pct)
        target_soc_pct: Target SOC in percent (if None, use duration_sec)
        initial_soc_pct: Initial SOC in percent
        dt_ms: Time step in milliseconds (default: 100.0ms for real-time simulation)
        temperature_c: Temperature in Celsius
        save_plot: Whether to save the plot (default: False)
        plot_filename: Filename for plot (auto-generated if None)
        save_csv: Whether to save CSV data (default: True)
        csv_filename: Filename for CSV (auto-generated if None)
    """
    capacity_ah = 100.0  # Fixed capacity per cell
    
    # Determine current direction
    if mode.lower() == 'discharge':
        current_ma = -current_amp * 1000.0  # Negative for discharge
        mode_str = "Discharge"
    elif mode.lower() == 'charge':
        current_ma = current_amp * 1000.0  # Positive for charge
        mode_str = "Charge"
    else:
        raise ValueError(f"Mode must be 'charge' or 'discharge', got '{mode}'")
    
    # Determine stopping condition
    if duration_sec is not None:
        stop_condition = f"{duration_sec}s duration"
        # Calculate exact number of steps for real-time simulation
        exact_steps = int((duration_sec * 1000) / dt_ms)
        max_steps = exact_steps + 10  # Small safety margin
    elif target_soc_pct is not None:
        stop_condition = f"{target_soc_pct}% SOC"
        max_steps = 1000000  # Large limit for SOC-based stopping
    else:
        raise ValueError("Either duration_sec or target_soc_pct must be specified")
    
    print("="*80)
    print(f"16S Pack {mode_str} Simulation")
    print("="*80)
    print(f"Mode: {mode_str}")
    print(f"Current: {current_amp}A ({current_amp/capacity_ah:.2f}C)")
    print(f"Cell Capacity: {capacity_ah}Ah (fixed)")
    print(f"Initial SOC: {initial_soc_pct}%")
    print(f"Stop Condition: {stop_condition}")
    print(f"Time Step: {dt_ms}ms (real-time simulation)")
    print(f"Temperature: {temperature_c}°C")
    if duration_sec is not None:
        print(f"Expected data points: {exact_steps} (one per {dt_ms}ms interval)")
    print("="*80)
    
    # Create 16S pack
    pack = BatteryPack16S(
        cell_capacity_ah=capacity_ah,
        initial_soc_pct=initial_soc_pct,
        ambient_temp_c=temperature_c
    )
    
    # Data storage
    time_data = []
    soc_data = []
    pack_voltage_data = []
    pack_current_data = []
    cell_voltages_data = []  # Will store all 16 cell voltages
    cell_temperatures_data = []  # Will store all 16 cell temperatures
    
    step = 0
    elapsed_time = 0.0
    
    print(f"\nStarting simulation (real-time at {dt_ms}ms intervals)...")
    
    initial_pack_state = pack.get_pack_state()
    initial_soc = initial_pack_state['pack_soc_pct']
    initial_voltage = initial_pack_state['pack_voltage_mv'] / 1000.0
    
    # Determine stopping condition
    if duration_sec is not None:
        # Real-time simulation: stop after exact duration
        # For 1 minute at 100ms intervals, we need exactly 600 steps
        target_time = duration_sec
        while elapsed_time < target_time and step < max_steps:
            # Update pack
            pack.update(
                current_ma=current_ma,
                dt_ms=dt_ms,
                ambient_temp_c=temperature_c
            )
            
            # Increment time by exact dt_ms (no fast-forwarding)
            # Use exact calculation to avoid floating point errors
            elapsed_time = round(step * dt_ms / 1000.0, 6)
            step += 1
            
            # Get pack state
            pack_state = pack.get_pack_state()
            
            # Store data (round to 6 decimal places)
            time_data.append(round(elapsed_time, 6))
            soc_data.append(round(pack_state['pack_soc_pct'], 6))
            pack_voltage_data.append(round(pack_state['pack_voltage_mv'] / 1000.0, 6))
            pack_current_data.append(round(current_amp if mode == 'charge' else -current_amp, 6))
            
            # Store all 16 cell voltages (round to 6 decimal places)
            cell_voltages = [round(v / 1000.0, 6) for v in pack_state['cell_voltages_mv']]
            cell_voltages_data.append(cell_voltages)
            
            # Store all 16 cell temperatures (round to 6 decimal places)
            cell_temps = [round(t, 6) for t in pack_state['cell_temperatures_c']]
            cell_temperatures_data.append(cell_temps)
            
            # Progress update every 10 seconds
            if step % int(10.0 / (dt_ms / 1000.0)) == 0:
                print(f"  Time: {elapsed_time:6.1f}s | SOC: {pack_state['pack_soc_pct']:6.2f}% | Pack Voltage: {pack_state['pack_voltage_mv']/1000:.3f}V")
    else:
        # Stop at target SOC
        target_soc = target_soc_pct
        if mode == 'discharge':
            while pack.get_pack_soc() > target_soc and step < max_steps:
                pack.update(
                    current_ma=current_ma,
                    dt_ms=dt_ms,
                    ambient_temp_c=temperature_c
                )
                
                elapsed_time = round(step * dt_ms / 1000.0, 6)
                step += 1
                
                pack_state = pack.get_pack_state()
                
                # Store data (round to 6 decimal places)
                time_data.append(round(elapsed_time, 6))
                soc_data.append(round(pack_state['pack_soc_pct'], 6))
                pack_voltage_data.append(round(pack_state['pack_voltage_mv'] / 1000.0, 6))
                pack_current_data.append(round(-current_amp, 6))
                
                cell_voltages = [round(v / 1000.0, 6) for v in pack_state['cell_voltages_mv']]
                cell_voltages_data.append(cell_voltages)
                
                cell_temps = [round(t, 6) for t in pack_state['cell_temperatures_c']]
                cell_temperatures_data.append(cell_temps)
                
                if step % int(10.0 / (dt_ms / 1000.0)) == 0:
                    print(f"  Time: {elapsed_time:6.1f}s | SOC: {pack_state['pack_soc_pct']:6.2f}% | Pack Voltage: {pack_state['pack_voltage_mv']/1000:.3f}V")
        else:  # charge
            while pack.get_pack_soc() < target_soc and step < max_steps:
                pack.update(
                    current_ma=current_ma,
                    dt_ms=dt_ms,
                    ambient_temp_c=temperature_c
                )
                
                elapsed_time = round(step * dt_ms / 1000.0, 6)
                step += 1
                
                pack_state = pack.get_pack_state()
                
                # Store data (round to 6 decimal places)
                time_data.append(round(elapsed_time, 6))
                soc_data.append(round(pack_state['pack_soc_pct'], 6))
                pack_voltage_data.append(round(pack_state['pack_voltage_mv'] / 1000.0, 6))
                pack_current_data.append(round(current_amp, 6))
                
                cell_voltages = [round(v / 1000.0, 6) for v in pack_state['cell_voltages_mv']]
                cell_voltages_data.append(cell_voltages)
                
                cell_temps = [round(t, 6) for t in pack_state['cell_temperatures_c']]
                cell_temperatures_data.append(cell_temps)
                
                if step % int(10.0 / (dt_ms / 1000.0)) == 0:
                    print(f"  Time: {elapsed_time:6.1f}s | SOC: {pack_state['pack_soc_pct']:6.2f}% | Pack Voltage: {pack_state['pack_voltage_mv']/1000:.3f}V")
    
    final_pack_state = pack.get_pack_state()
    final_soc = final_pack_state['pack_soc_pct']
    final_voltage = pack_voltage_data[-1]
    
    print(f"\nSimulation completed:")
    print(f"  Steps: {step}")
    print(f"  Duration: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
    print(f"  Initial SOC: {initial_soc:.2f}%")
    print(f"  Final SOC: {final_soc:.2f}%")
    print(f"  SOC Change: {abs(initial_soc - final_soc):.4f}%")
    print(f"  Initial Pack Voltage: {initial_voltage:.3f}V")
    print(f"  Final Pack Voltage: {final_voltage:.3f}V")
    print(f"  Voltage Change: {abs(initial_voltage - final_voltage):.3f}V")
    
    # Prepare data dictionary
    data = {
        'time': np.array(time_data),
        'soc': np.array(soc_data),
        'pack_voltage': np.array(pack_voltage_data),
        'pack_current': np.array(pack_current_data),
        'cell_voltages': np.array(cell_voltages_data),  # Shape: (n_steps, 16)
        'cell_temperatures': np.array(cell_temperatures_data)  # Shape: (n_steps, 16)
    }
    
    # Save CSV data
    if save_csv:
        # Create output directory if it doesn't exist
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        
        if csv_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"pack_{mode}_{current_amp}A_{timestamp}.csv"
        
        # Ensure filename doesn't have path separators
        csv_filename = Path(csv_filename).name
        csv_path = output_dir / csv_filename
        
        # Create DataFrame with all cell voltages
        df_dict = {
            'time_s': np.round(data['time'], 6),
            'soc_percent': np.round(data['soc'], 6),
            'pack_voltage_V': np.round(data['pack_voltage'], 6),
            'pack_current_A': np.round(data['pack_current'], 6),
        }
        
        # Add all 16 cell voltages
        for i in range(16):
            df_dict[f'cell_{i+1}_V'] = np.round(data['cell_voltages'][:, i], 6)
        
        # Add all 16 cell temperatures
        for i in range(16):
            df_dict[f'cell_{i+1}_temp_C'] = np.round(data['cell_temperatures'][:, i], 6)
        
        df = pd.DataFrame(df_dict)
        df.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"\nCSV data saved to: {csv_path}")
        print(f"  Total rows: {len(df)}")
        print(f"  Columns: {len(df.columns)} (time, soc, pack_voltage, pack_current, 16 cell voltages, 16 cell temperatures)")
    
    # Generate plot
    if save_plot:
        plot_results(data=data, mode=mode, current_amp=current_amp, plot_filename=plot_filename)
    
    return data

def plot_results(data, mode='discharge', current_amp=1.0, plot_filename=None):
    """Plot simulation results."""
    print("\nGenerating plots...")
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    if plot_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"pack_{mode}_{current_amp}A_{timestamp}.png"
    
    # Ensure filename doesn't have path separators
    plot_filename = Path(plot_filename).name
    plot_path = output_dir / plot_filename
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'16S Pack {mode.capitalize()} Simulation - {current_amp}A', 
                 fontsize=16, fontweight='bold')
    
    # Calculate average cell voltage
    avg_cell_voltage = data['cell_voltages'].mean(axis=1)
    
    # 1. Pack Voltage vs Time
    ax1 = axes[0, 0]
    ax1.plot(data['time'], data['pack_voltage'], 'b-', linewidth=2, label='Pack Voltage', alpha=0.9)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Pack Voltage (V)', fontsize=12)
    ax1.set_title('Pack Voltage vs Time', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='best')
    
    # 2. SOC vs Time
    ax2 = axes[0, 1]
    ax2.plot(data['time'], data['soc'], 'g-', linewidth=2, label='SOC', alpha=0.9)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('State of Charge (%)', fontsize=12)
    ax2.set_title('SOC vs Time', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='best')
    
    # 3. Average Cell Voltage vs SOC
    ax3 = axes[1, 0]
    ax3.plot(data['soc'], avg_cell_voltage, 'b-', linewidth=2, label='Avg Cell Voltage', alpha=0.9)
    ax3.set_xlabel('SOC (%)', fontsize=12)
    ax3.set_ylabel('Average Cell Voltage (V)', fontsize=12)
    ax3.set_title('Average Cell Voltage vs SOC', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11, loc='best')
    if mode == 'discharge':
        ax3.invert_xaxis()
    
    # 4. Cell Voltage Spread
    ax4 = axes[1, 1]
    cell_spread = data['cell_voltages'].max(axis=1) - data['cell_voltages'].min(axis=1)
    ax4.plot(data['time'], cell_spread * 1000, 'r-', linewidth=2, label='Cell Spread', alpha=0.9)
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_ylabel('Cell Voltage Spread (mV)', fontsize=12)
    ax4.set_title('Cell-to-Cell Voltage Variation', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run 16S LiFePO4 pack charge/discharge simulation for BMS testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Discharge at 1A for 60 seconds (real-time at 100ms intervals = 600 data points)
  python run_cell_simulation.py --mode discharge --current 1.0 --duration 60

  # Charge at 2A until 100% SOC
  python run_cell_simulation.py --mode charge --current 2.0 --target-soc 100

  # Discharge at 5A from 100% to 50% SOC
  python run_cell_simulation.py --mode discharge --current 5.0 --initial-soc 100 --target-soc 50

  # Discharge at 0.5A for 2 minutes with plot
  python run_cell_simulation.py --mode discharge --current 0.5 --duration 120 --plot

  # Discharge at 1A for 1 minute, save CSV only (no plot)
  python run_cell_simulation.py --mode discharge --current 1.0 --duration 60
        """
    )
    
    parser.add_argument('--mode', type=str, choices=['charge', 'discharge'], default='discharge',
                       help='Operation mode: charge or discharge (default: discharge)')
    parser.add_argument('--current', type=float, required=True,
                       help='Current in Amperes (positive value)')
    parser.add_argument('--duration', type=float, default=None,
                       help='Duration in seconds (mutually exclusive with --target-soc)')
    parser.add_argument('--target-soc', type=float, default=None,
                       help='Target SOC in percent (mutually exclusive with --duration)')
    parser.add_argument('--initial-soc', type=float, default=100.0,
                       help='Initial SOC in percent (default: 100.0)')
    parser.add_argument('--dt', type=float, default=100.0,
                       help='Time step in milliseconds (default: 100.0ms for real-time simulation)')
    parser.add_argument('--temperature', type=float, default=25.0,
                       help='Temperature in Celsius (default: 25.0)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plot (default: False, CSV is always saved)')
    parser.add_argument('--plot-filename', type=str, default=None,
                       help='Custom filename for plot (default: auto-generated)')
    parser.add_argument('--csv-filename', type=str, default=None,
                       help='Custom filename for CSV (default: auto-generated)')
    parser.add_argument('--no-csv', action='store_true',
                       help='Do not save CSV data')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.duration is None and args.target_soc is None:
        parser.error("Either --duration or --target-soc must be specified")
    if args.duration is not None and args.target_soc is not None:
        parser.error("--duration and --target-soc are mutually exclusive")
    
    # Run simulation
    run_simulation(
        mode=args.mode,
        current_amp=args.current,
        duration_sec=args.duration,
        target_soc_pct=args.target_soc,
        initial_soc_pct=args.initial_soc,
        dt_ms=args.dt,
        temperature_c=args.temperature,
        save_plot=args.plot,
        plot_filename=args.plot_filename,
        save_csv=not args.no_csv,
        csv_filename=args.csv_filename
    )
    
    print("\nSimulation complete!")

if __name__ == '__main__':
    main()
