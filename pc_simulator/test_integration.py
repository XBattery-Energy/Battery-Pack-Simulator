"""
Quick test script to verify integration and print sample data.
"""

import sys
import os

# Add parent directory to path so we can import sil_bms
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import sys
import os

# Add parent directory to path so we can import sil_bms
# File is at: sil_bms/pc_simulator/test_integration.py
# We need: C:\Work\T_appl in path
script_dir = os.path.dirname(os.path.abspath(__file__))  # sil_bms/pc_simulator
parent_dir = os.path.dirname(script_dir)  # sil_bms
project_root = os.path.dirname(parent_dir)  # T_appl
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import time
from sil_bms.pc_simulator.plant.pack_model import BatteryPack16S
from sil_bms.pc_simulator.afe.wrapper import AFEWrapper
from sil_bms.pc_simulator.communication.uart_tx import UARTTransmitter


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
        if flags & 0xFFFF:
            for i in range(16):
                if flags & (1 << i):
                    print(f"    - Open wire on cell {i}")
        if flags & (0xFFFF << 16):
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
    print("\n" + "=" * 80)
    print("SIL BMS Simulator - Integration Test")
    print("=" * 80)
    
    # 1. Create Battery Pack
    print("\n[1/3] Creating Battery Pack Model...")
    pack = BatteryPack16S(
        cell_capacity_ah=100.0,
        initial_soc_pct=50.0,
        ambient_temp_c=25.0,
        seed=42
    )
    print(f"  [OK] Pack: {pack.get_pack_voltage()/1000:.2f}V, SOC: {pack.get_pack_soc():.1f}%")
    
    # 2. Create AFE Wrapper
    print("\n[2/3] Creating AFE Wrapper...")
    afe = AFEWrapper(seed=42)
    afe.start_simulation()
    print("  [OK] AFE wrapper ready")
    
    # 3. Create UART Transmitter (print-only mode)
    print("\n[3/3] Creating UART Transmitter (print-only mode)...")
    print("  [OK] Transmitter ready (not connected to physical port)")
    
    # 4. Simulate a few frames
    print("\n" + "-" * 80)
    print("Simulating 3 frames...")
    print("-" * 80)
    
    current_ma = 50000.0  # 50A charge
    dt_ms = 20.0  # 20ms = 50Hz
    
    for frame_num in range(3):
        # Update pack
        pack.update(current_ma=current_ma, dt_ms=dt_ms, ambient_temp_c=25.0)
        
        # Get true values
        true_v = pack.get_cell_voltages()
        true_t = pack.get_cell_temperatures()
        true_i = pack.get_pack_current()
        
        # Apply AFE processing
        measured_v, measured_t, measured_i, flags = afe.apply_measurement(
            true_v, true_t, true_i
        )
        
        # Prepare frame data
        # Note: measured_t from AFE is already in centi-°C (int16 format)
        frame_data = {
            'timestamp_ms': frame_num * 20,
            'vcell_mv': measured_v.astype(np.uint16),
            'tcell_cc': measured_t.astype(np.int16),  # Already in centi-°C
            'pack_current_ma': int(measured_i),
            'pack_voltage_mv': int(pack.get_pack_voltage()),
            'status_flags': flags
        }
        
        # Print frame
        print_frame_data(frame_data, frame_num)
        
        # Show what would be sent via UART
        from sil_bms.pc_simulator.communication.protocol import AFEMeasFrame
        frame_bytes = AFEMeasFrame.encode(
            frame_data['timestamp_ms'],
            frame_data['vcell_mv'],
            frame_data['tcell_cc'],
            frame_data['pack_current_ma'],
            frame_data['pack_voltage_mv'],
            frame_data['status_flags'],
            frame_num
        )
        print(f"\n[TX] Would send {len(frame_bytes)} bytes via UART")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
    print("\nTo run with actual UART transmission:")
    print("  python main.py --port COM3 --duration 10 --rate 50")
    print("\nTo run in print-only mode:")
    print("  python main.py --duration 10 --rate 50")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()

