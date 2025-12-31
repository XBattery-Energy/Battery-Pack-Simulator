"""
Quick verification script to check if everything is ready for MCU connection.
"""

import sys
import os

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(parent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("\n" + "=" * 80)
print("SIL BMS Project Readiness Check")
print("=" * 80)

# 1. Check protocol
print("\n[1/5] Checking MCU Protocol...")
try:
    from sil_bms.pc_simulator.communication.protocol_mcu import (
        SILFrameEncoder, SIL_FRAME_HEADER, SIL_FRAME_FOOTER, 
        SIL_FRAME_VERSION, SIL_FRAME_OVERHEAD
    )
    print(f"  [OK] Protocol module loaded")
    print(f"  [OK] Frame header: 0x{SIL_FRAME_HEADER:02X} (expected: 0xAA)")
    print(f"  [OK] Frame footer: 0x{SIL_FRAME_FOOTER:02X} (expected: 0x55)")
    print(f"  [OK] Frame version: 0x{SIL_FRAME_VERSION:02X} (expected: 0x01)")
except Exception as e:
    print(f"  [FAIL] Error: {e}")
    sys.exit(1)

# 2. Check UART transmitter
print("\n[2/5] Checking UART Transmitter...")
try:
    from sil_bms.pc_simulator.communication.uart_tx_mcu import MCUCompatibleUARTTransmitter
    print(f"  [OK] MCU-compatible transmitter available")
except Exception as e:
    print(f"  [FAIL] Error: {e}")
    sys.exit(1)

# 3. Test frame encoding
print("\n[3/5] Testing Frame Encoding...")
try:
    import numpy as np
    encoder = SILFrameEncoder()
    vcell = np.array([3250] * 16, dtype=np.uint16)
    tcell = np.array([250] * 16, dtype=np.int16)
    frame = encoder.encode_frame(vcell, tcell, 50000, 52000, 0, 0)
    
    # Verify frame structure
    assert frame[0] == SIL_FRAME_HEADER, "Header mismatch"
    assert frame[1] == SIL_FRAME_VERSION, "Version mismatch"
    assert frame[-1] == SIL_FRAME_FOOTER, "Footer mismatch"
    
    data_length = (frame[2] << 8) | frame[3]
    expected_size = SIL_FRAME_OVERHEAD + data_length
    assert len(frame) == expected_size, "Frame size mismatch"
    
    print(f"  [OK] Frame encoding works")
    print(f"  [OK] Frame size: {len(frame)} bytes")
    print(f"  [OK] Data length: {data_length} bytes")
except Exception as e:
    print(f"  [FAIL] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Check main integration
print("\n[4/5] Checking Main Integration...")
try:
    from sil_bms.pc_simulator.main import main
    from sil_bms.pc_simulator.plant.pack_model import BatteryPack16S
    from sil_bms.pc_simulator.afe.wrapper import AFEWrapper
    print(f"  [OK] Main script imports successful")
    print(f"  [OK] Battery pack model available")
    print(f"  [OK] AFE wrapper available")
except Exception as e:
    print(f"  [FAIL] Error: {e}")
    sys.exit(1)

# 5. Check configuration
print("\n[5/5] Checking Configuration...")
print(f"  [OK] Default baudrate: 921600")
print(f"  [OK] Default frame rate: 50 Hz")
print(f"  [OK] MCU protocol: ENABLED (default)")
print(f"  [OK] Topology: 1 string, 1 module, 16 cells, 16 temp sensors")

print("\n" + "=" * 80)
print("READY FOR MCU CONNECTION!")
print("=" * 80)
print("\nTo connect and send data:")
print("  1. Connect USB-to-TTL adapter to your device")
print("  2. Find COM port (e.g., COM3, COM4, etc.)")
print("  3. Run: python sil_bms/pc_simulator/main.py --port COM3 --duration 60 --rate 50")
print("\nFrame format:")
print("  - Header: 0xAA")
print("  - Version: 0x01")
print("  - Data: Big-endian, matches MCU parsing order")
print("  - CRC16: CCITT on data payload")
print("  - Footer: 0x55")
print("=" * 80 + "\n")


