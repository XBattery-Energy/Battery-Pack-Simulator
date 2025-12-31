"""
Test script for MCU-compatible protocol

Verifies that the MCU protocol encoder produces frames matching the expected format.
"""

import sys
import os

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(parent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from sil_bms.pc_simulator.communication.protocol_mcu import (
    SILFrameEncoder,
    SIL_FRAME_HEADER,
    SIL_FRAME_FOOTER,
    SIL_FRAME_VERSION,
    SIL_FRAME_OVERHEAD,
    crc16_ccitt_be
)


def test_mcu_frame_format():
    """Test that MCU frame format is correct."""
    print("\n" + "=" * 80)
    print("MCU Protocol Frame Format Test")
    print("=" * 80)
    
    # Create encoder
    encoder = SILFrameEncoder()
    
    # Create test data
    vcell_mv = np.array([3250, 3251, 3252, 3253, 3254, 3255, 3256, 3257,
                         3258, 3259, 3260, 3261, 3262, 3263, 3264, 3265], dtype=np.uint16)
    tcell_cc = np.array([250, 251, 252, 253, 254, 255, 256, 257,
                         258, 259, 260, 261, 262, 263, 264, 265], dtype=np.int16)
    pack_current_ma = 50000
    pack_voltage_mv = 52000
    status_flags = 0
    timestamp_ms = 0
    
    # Encode frame
    frame = encoder.encode_frame(
        vcell_mv=vcell_mv,
        tcell_cc=tcell_cc,
        pack_current_ma=pack_current_ma,
        pack_voltage_mv=pack_voltage_mv,
        status_flags=status_flags,
        timestamp_ms=timestamp_ms
    )
    
    # Verify frame structure
    print(f"\nFrame size: {len(frame)} bytes")
    print(f"Expected overhead: {SIL_FRAME_OVERHEAD} bytes")
    
    # Check header
    assert frame[0] == SIL_FRAME_HEADER, f"Header mismatch: got 0x{frame[0]:02X}, expected 0x{SIL_FRAME_HEADER:02X}"
    print(f"[OK] Header: 0x{frame[0]:02X}")
    
    # Check version
    assert frame[1] == SIL_FRAME_VERSION, f"Version mismatch: got 0x{frame[1]:02X}, expected 0x{SIL_FRAME_VERSION:02X}"
    print(f"[OK] Version: 0x{frame[1]:02X}")
    
    # Check length
    length_msb = frame[2]
    length_lsb = frame[3]
    data_length = (length_msb << 8) | length_lsb
    print(f"[OK] Data length: {data_length} bytes (MSB: 0x{length_msb:02X}, LSB: 0x{length_lsb:02X})")
    
    # Extract data payload
    data_start = 4
    data_end = data_start + data_length
    data_payload = frame[data_start:data_end]
    
    # Verify CRC
    crc_start = data_end
    crc_msb = frame[crc_start]
    crc_lsb = frame[crc_start + 1]
    received_crc = (crc_msb << 8) | crc_lsb
    
    calculated_crc = crc16_ccitt_be(data_payload, 0xFFFF)
    assert received_crc == calculated_crc, f"CRC mismatch: got 0x{received_crc:04X}, expected 0x{calculated_crc:04X}"
    print(f"[OK] CRC: 0x{received_crc:04X} (verified)")
    
    # Check footer
    footer_idx = crc_start + 2
    assert frame[footer_idx] == SIL_FRAME_FOOTER, f"Footer mismatch: got 0x{frame[footer_idx]:02X}, expected 0x{SIL_FRAME_FOOTER:02X}"
    print(f"[OK] Footer: 0x{frame[footer_idx]:02X}")
    
    # Verify frame size
    expected_size = SIL_FRAME_OVERHEAD + data_length
    assert len(frame) == expected_size, f"Frame size mismatch: got {len(frame)}, expected {expected_size}"
    print(f"[OK] Frame size: {len(frame)} bytes (matches expected)")
    
    # Print frame structure
    print(f"\nFrame Structure:")
    print(f"  [0x{frame[0]:02X}] Header")
    print(f"  [0x{frame[1]:02X}] Version")
    print(f"  [0x{frame[2]:02X} 0x{frame[3]:02X}] Length ({data_length} bytes)")
    print(f"  [{len(data_payload)} bytes] Data payload")
    print(f"  [0x{crc_msb:02X} 0x{crc_lsb:02X}] CRC16")
    print(f"  [0x{frame[footer_idx]:02X}] Footer")
    
    print("\n" + "=" * 80)
    print("[OK] All frame format checks passed!")
    print("=" * 80 + "\n")
    
    return frame


def test_data_layout():
    """Test that data layout matches MCU parsing order."""
    print("\n" + "=" * 80)
    print("MCU Protocol Data Layout Test")
    print("=" * 80)
    
    encoder = SILFrameEncoder()
    
    # Create test data with known values
    vcell_mv = np.array([3000 + i * 10 for i in range(16)], dtype=np.uint16)
    tcell_cc = np.array([200 + i for i in range(16)], dtype=np.int16)
    pack_current_ma = 50000
    pack_voltage_mv = 52000
    
    frame = encoder.encode_frame(
        vcell_mv=vcell_mv,
        tcell_cc=tcell_cc,
        pack_current_ma=pack_current_ma,
        pack_voltage_mv=pack_voltage_mv,
        status_flags=0,
        timestamp_ms=0
    )
    
    # Extract data payload
    data_length = (frame[2] << 8) | frame[3]
    data_payload = frame[4:4+data_length]
    
    print(f"\nData payload size: {len(data_payload)} bytes")
    print(f"Expected sections:")
    print(f"  1. Cell voltages: 16 cells × 2 bytes = 32 bytes")
    print(f"  2. Module voltages: 1 module × 4 bytes = 4 bytes")
    print(f"  3. Temperatures: 16 sensors × 2 bytes = 32 bytes")
    print(f"  4. Balancing feedback: 1 module × 2 bytes = 2 bytes")
    print(f"  5. Open wire: 2 bytes count + 17 bytes array = 19 bytes")
    print(f"  6. GPIO voltages: 8 GPIOs × 2 bytes = 16 bytes")
    print(f"  7. GPA voltages: 4 GPAs × 2 bytes = 8 bytes")
    print(f"  8. Current sensor: 6 values × 4 bytes × 1 string = 24 bytes")
    print(f"  9. Pack values: 3 + 3×1 strings = 6 values × 4 bytes = 24 bytes")
    print(f"  10. Digital inputs: ~100+ bytes")
    
    # Verify first cell voltage (big-endian)
    offset = 0
    first_cell_msb = data_payload[offset]
    first_cell_lsb = data_payload[offset + 1]
    first_cell_value = (first_cell_msb << 8) | first_cell_lsb
    expected_first = vcell_mv[0]
    
    print(f"\nFirst cell voltage:")
    print(f"  Bytes: 0x{first_cell_msb:02X} 0x{first_cell_lsb:02X}")
    print(f"  Value: {first_cell_value} mV (expected: {expected_first} mV)")
    assert first_cell_value == expected_first, f"First cell voltage mismatch"
    print(f"  [OK] Correct")
    
    print("\n" + "=" * 80)
    print("[OK] Data layout test passed!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    try:
        frame = test_mcu_frame_format()
        test_data_layout()
        print("\n" + "=" * 80)
        print("All MCU protocol tests passed!")
        print("=" * 80 + "\n")
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

