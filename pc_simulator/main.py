"""
Main Integration Script for SIL BMS Simulator

This script integrates:
- Battery Pack Model (16S)
- AFE Measurement Wrapper
- UART Transmitter

Runs simulation and sends AFE measurement frames to MCU via UART.
"""

import sys
import os

# Add parent directory to path so we can import sil_bms
# File is at: Battery Pack Simulator/pc_simulator/main.py
# We need to add the parent directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))  # Battery Pack Simulator/pc_simulator
parent_dir = os.path.dirname(script_dir)  # Battery Pack Simulator
# Add parent directory to path for imports
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# Also add current directory for relative imports
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import numpy as np
import time
import argparse
from plant.pack_model import BatteryPack16S
from plant.current_profile import CurrentProfile
from afe.wrapper import AFEWrapper, FaultType
from communication.uart_tx import UARTTransmitter
from communication.uart_tx_mcu import MCUCompatibleUARTTransmitter
from communication.uart_tx_xbb import XBBUARTTransmitter


def print_frame_data(frame_data: dict, sequence: int):
    """Print frame data in readable format."""
    print("\n" + "=" * 80)
    print(f"AFE_MEAS_FRAME - Sequence: {sequence}")
    print("=" * 80)
    print(f"Timestamp: {frame_data['timestamp_ms']} ms")
    print(f"\nCell Voltages (mV):")
    vcell = frame_data['vcell_mv']
    for i in range(0, 16, 4):
        cells = [f"Cell {j:2d}: {vcell[j]:6.1f} mV" for j in range(i, min(i+4, 16))]
        print("  " + "  |  ".join(cells))
    
    print(f"\nCell Temperatures (°C):")
    tcell = frame_data['tcell_cc'] / 100.0  # Convert centi-°C to °C
    for i in range(0, 16, 4):
        cells = [f"Cell {j:2d}: {tcell[j]:6.2f} °C" for j in range(i, min(i+4, 16))]
        print("  " + "  |  ".join(cells))
    
    print(f"\nPack Measurements:")
    print(f"  Pack Voltage: {frame_data['pack_voltage_mv'] / 1000.0:.3f} V")
    print(f"  Pack Current: {frame_data['pack_current_ma'] / 1000.0:.3f} A")
    
    print(f"\nStatus Flags: 0x{frame_data['status_flags']:08X}")
    flags = frame_data['status_flags']
    if flags != 0:
        print("  Active Flags:")
        if flags & 0xFFFF:  # Bits 0-15: Open wire
            for i in range(16):
                if flags & (1 << i):
                    print(f"    - Open wire on cell {i}")
        if flags & (0xFFFF << 16):  # Bits 16-31: Other faults
            for i in range(16):
                if flags & (1 << (16 + i)):
                    print(f"    - NTC fault on cell {i}")
        if flags & (1 << 30):
            print(f"    - Current sensor fault")
        if flags & (1 << 31):
            print(f"    - CRC error")
    else:
        print("  No faults")
    
    print("=" * 80)


def main():
    """Main simulation loop."""
    parser = argparse.ArgumentParser(description='SIL BMS Simulator')
    parser.add_argument('--port', type=str, default=None,
                       help='Serial port (e.g., COM3 or /dev/ttyUSB0). If not specified, only prints data.')
    parser.add_argument('--baudrate', type=int, default=921600,
                       help='Baud rate (default: 921600)')
    parser.add_argument('--rate', type=float, default=1.0,
                       help='Frame rate in Hz (default: 1.0 for XBB protocol)')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Simulation duration in seconds (default: 10.0). Use 0 for infinite/continuous transmission.')
    parser.add_argument('--current', type=float, default=50.0,
                       help='Pack current in Amperes (default: 50.0, positive=charge)')
    parser.add_argument('--soc', type=float, default=50.0,
                       help='Initial SOC in percent (default: 50.0)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--print-frames', action='store_true', default=True,
                       help='Print frame data (default: True)')
    parser.add_argument('--no-print', dest='print_frames', action='store_false',
                       help='Disable frame printing')
    parser.add_argument('--protocol', type=str, default='xbb',
                       choices=['xbb', 'mcu', 'legacy'],
                       help='Protocol type: xbb (default), mcu, or legacy')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("SIL BMS Simulator - Main Integration")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Serial Port: {args.port if args.port else 'None (print only)'}")
    print(f"  Baud Rate: {args.baudrate}")
    print(f"  Frame Rate: {args.rate} Hz")
    print(f"  Duration: {args.duration} seconds")
    print(f"  Pack Current: {args.current} A")
    print(f"  Initial SOC: {args.soc}%")
    print("=" * 80 + "\n")
    
    # 1. Create Battery Pack Model
    print("Initializing Battery Pack Model...")
    pack = BatteryPack16S(
        cell_capacity_ah=100.0,
        initial_soc_pct=args.soc,
        ambient_temp_c=25.0,
        seed=42
    )
    print(f"  [OK] Pack initialized: {pack.get_pack_voltage()/1000:.2f}V, SOC: {pack.get_pack_soc():.1f}%")
    
    # 2. Create AFE Wrapper
    print("Initializing AFE Wrapper...")
    afe = AFEWrapper(
        noise_config={
            'voltage_noise_mv': 2.0,
            'temp_noise_c': 0.5,
            'current_noise_ma': 50.0
        },
        seed=42
    )
    afe.start_simulation()  # Initialize simulation time for fault scheduling
    print("  [OK] AFE wrapper initialized")
    
    # 3. Create UART Transmitter (if port specified)
    tx = None
    if args.port:
        protocol_type = args.protocol.upper()
        print(f"Initializing UART Transmitter on {args.port} ({protocol_type} protocol)...")
        try:
            if args.protocol == 'xbb':
                tx = XBBUARTTransmitter(
                    port=args.port,
                    baudrate=args.baudrate,
                    frame_rate_hz=args.rate,
                    verbose=args.verbose,
                    print_frames=args.print_frames
                )
            elif args.protocol == 'mcu':
                tx = MCUCompatibleUARTTransmitter(
                    port=args.port,
                    baudrate=args.baudrate,
                    frame_rate_hz=args.rate,
                    verbose=args.verbose,
                    num_strings=1,
                    num_modules=1,
                    num_cells=16,
                    num_temp_sensors=16
                )
            else:  # legacy
                tx = UARTTransmitter(
                    port=args.port,
                    baudrate=args.baudrate,
                    frame_rate_hz=args.rate,
                    verbose=args.verbose
                )
            if tx.start():
                print(f"  [OK] UART transmitter started ({protocol_type})")
            else:
                print(f"  [ERROR] Failed to start UART transmitter")
                tx = None
        except Exception as e:
            print(f"  [ERROR] Error initializing UART: {e}")
            print(f"  Continuing in print-only mode...")
            tx = None
    else:
        print("  No serial port specified - running in print-only mode")
    
    # 4. Simulation parameters
    dt_ms = 1000.0 / args.rate  # Time step in milliseconds
    continuous_mode = (args.duration <= 0)
    num_steps = int(args.duration * args.rate) if not continuous_mode else 0
    current_ma = args.current * 1000.0  # Convert A to mA
    
    print(f"\nStarting simulation...")
    print(f"  Time step: {dt_ms:.1f} ms")
    if continuous_mode:
        print(f"  Mode: CONTINUOUS (infinite loop - press Ctrl+C to stop)")
    else:
        print(f"  Total steps: {num_steps}")
    print(f"  Current: {args.current} A ({'charge' if args.current > 0 else 'discharge'})")
    print("\n" + "-" * 80)
    
    # 5. Simulation loop
    start_time = time.time()
    frame_count = 0
    
    try:
        step = 0
        while continuous_mode or step < num_steps:
            # Update battery pack
            pack.update(current_ma=current_ma, dt_ms=dt_ms, ambient_temp_c=25.0)
            
            # Get true values from pack
            true_v = pack.get_cell_voltages()      # mV, array[16]
            true_t = pack.get_cell_temperatures()  # °C, array[16]
            true_i = pack.get_pack_current()       # mA
            
            # Apply AFE processing (adds noise, quantization, faults)
            measured_v, measured_t, measured_i, flags = afe.apply_measurement(
                true_v, true_t, true_i
            )
            
            # Prepare frame data based on protocol
            if args.protocol == 'xbb':
                # XBB protocol: convert temperatures from centi-°C to °C
                # measured_t is in centi-°C, convert to °C
                temp_cell_c = np.mean(measured_t / 100.0)  # Average cell temperature in °C
                temp_pcb_c = 25.0  # Use ambient temperature as PCB temperature (or could use average)
                
                frame_data = {
                    'pack_current_ma': int(measured_i),  # Already in milli-Amperes (signed)
                    'pack_voltage_mv': int(pack.get_pack_voltage()),  # Already in milli-Volts
                    'temp_cell_c': float(temp_cell_c),  # Average cell temperature in °C
                    'temp_pcb_c': float(temp_pcb_c),  # PCB temperature in °C
                    'cell_voltages_mv': measured_v.astype(np.int32)  # Cell voltages in milli-Volts
                }
            else:
                # Legacy/MCU protocols
                # Note: measured_t from AFE is already in centi-°C (int16 format)
                frame_data = {
                    'timestamp_ms': int((time.time() - start_time) * 1000),
                    'vcell_mv': measured_v.astype(np.uint16),        # Already in mV
                    'tcell_cc': measured_t.astype(np.int16),         # Already in centi-°C
                    'pack_current_ma': int(measured_i),
                    'pack_voltage_mv': int(pack.get_pack_voltage()),
                    'status_flags': flags
                }
            
            # Print frame data (if enabled and not XBB - XBB prints its own format)
            if args.print_frames and args.protocol != 'xbb':
                print_frame_data(frame_data, frame_count)
            
            # Send frame via UART (if transmitter available)
            if tx is not None:
                success = tx.send_frame(frame_data)
                if not success and args.verbose:
                    print(f"  ⚠ Failed to queue frame {frame_count}")
            
            frame_count += 1
            step += 1
            
            # Rate limiting (sleep to maintain frame rate)
            elapsed = time.time() - start_time
            expected_time = step * dt_ms / 1000.0
            sleep_time = expected_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Print status every 100 frames in continuous mode
            if continuous_mode and frame_count % 100 == 0:
                print(f"[Status] Frames sent: {frame_count}, Elapsed: {elapsed:.1f}s, Pack SOC: {pack.get_pack_soc():.1f}%")
    
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
    
    finally:
        # Stop transmitter
        if tx is not None:
            print("\nStopping UART transmitter...")
            tx.stop()
            
            # Print statistics
            stats = tx.get_statistics()
            print(f"\nTransmission Statistics:")
            print(f"  Frames sent: {stats['sent_count']}")
            print(f"  Errors: {stats['error_count']}")
            print(f"  Last error: {stats['last_error'] if stats['last_error'] else 'None'}")
            print(f"  Final sequence: {stats['sequence']}")
        
        # Print pack state
        print(f"\nFinal Pack State:")
        print(f"  Pack Voltage: {pack.get_pack_voltage()/1000:.3f} V")
        print(f"  Pack SOC: {pack.get_pack_soc():.1f}%")
        print(f"  Pack Current: {pack.get_pack_current()/1000:.3f} A")
        
        imbalance = pack.get_cell_imbalance()
        print(f"\nCell Imbalance:")
        print(f"  Voltage delta: {imbalance['voltage_delta_mv']:.2f} mV")
        print(f"  SOC delta: {imbalance['soc_delta_pct']:.2f}%")
        
        print("\n" + "=" * 80)
        print("Simulation completed!")
        print("=" * 80 + "\n")


if __name__ == '__main__':
    main()

