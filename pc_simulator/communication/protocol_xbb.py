"""
UART Protocol for XBB Communication

Frame Format:
[0xA5] [0x33] [SubIndex: 0x0000] [DataLen: 80] [Data: 80 bytes] [0xB5] [CRC8]

Data Structure (20 int32 values, big-endian, 80 bytes total):
- pack_current_A: 4 bytes (int32, milli_A)
- pack_voltage_V: 4 bytes (int32, milli_V)
- temp_cell_C: 4 bytes (int32, milli_degC)
- temp_pcb_C: 4 bytes (int32, milli_degC)
- cell_1_V through cell_16_V: 16 × 4 = 64 bytes (int32, milli_V each)

Total Frame Size: 88 bytes
CRC8 calculated over all bytes from 0xA5 through 0xB5 (excluding CRC8 byte itself)
"""

import struct
from typing import Optional
import numpy as np


# Frame markers
XBB_FRAME_HEADER = 0xA5
XBB_FRAME_MSG_ID = 0x33
XBB_FRAME_FOOTER = 0xB5
XBB_SUBINDEX = 0x0000
XBB_DATA_LENGTH = 80  # 20 int32 values × 4 bytes = 80 bytes


# CRC8 Table (provided by user)
CRC_TABLE = [
    0x00, 0x07, 0x0E, 0x09, 0x1C, 0x1B, 0x12, 0x15,
    0x38, 0x3F, 0x36, 0x31, 0x24, 0x23, 0x2A, 0x2D,
    0x70, 0x77, 0x7E, 0x79, 0x6C, 0x6B, 0x62, 0x65,
    0x48, 0x4F, 0x46, 0x41, 0x54, 0x53, 0x5A, 0x5D,
    0xE0, 0xE7, 0xEE, 0xE9, 0xFC, 0xFB, 0xF2, 0xF5,
    0xD8, 0xDF, 0xD6, 0xD1, 0xC4, 0xC3, 0xCA, 0xCD,
    0x90, 0x97, 0x9E, 0x99, 0x8C, 0x8B, 0x82, 0x85,
    0xA8, 0xAF, 0xA6, 0xA1, 0xB4, 0xB3, 0xBA, 0xBD,
    0xC7, 0xC0, 0xC9, 0xCE, 0xDB, 0xDC, 0xD5, 0xD2,
    0xFF, 0xF8, 0xF1, 0xF6, 0xE3, 0xE4, 0xED, 0xEA,
    0xB7, 0xB0, 0xB9, 0xBE, 0xAB, 0xAC, 0xA5, 0xA2,
    0x8F, 0x88, 0x81, 0x86, 0x93, 0x94, 0x9D, 0x9A,
    0x27, 0x20, 0x29, 0x2E, 0x3B, 0x3C, 0x35, 0x32,
    0x1F, 0x18, 0x11, 0x16, 0x03, 0x04, 0x0D, 0x0A,
    0x57, 0x50, 0x59, 0x5E, 0x4B, 0x4C, 0x45, 0x42,
    0x6F, 0x68, 0x61, 0x66, 0x73, 0x74, 0x7D, 0x7A,
    0x89, 0x8E, 0x87, 0x80, 0x95, 0x92, 0x9B, 0x9C,
    0xB1, 0xB6, 0xBF, 0xB8, 0xAD, 0xAA, 0xA3, 0xA4,
    0xF9, 0xFE, 0xF7, 0xF0, 0xE5, 0xE2, 0xEB, 0xEC,
    0xC1, 0xC6, 0xCF, 0xC8, 0xDD, 0xDA, 0xD3, 0xD4,
    0x69, 0x6E, 0x67, 0x60, 0x75, 0x72, 0x7B, 0x7C,
    0x51, 0x56, 0x5F, 0x58, 0x4D, 0x4A, 0x43, 0x44,
    0x19, 0x1E, 0x17, 0x10, 0x05, 0x02, 0x0B, 0x0C,
    0x21, 0x26, 0x2F, 0x28, 0x3D, 0x3A, 0x33, 0x34,
    0x4E, 0x49, 0x40, 0x47, 0x52, 0x55, 0x5C, 0x5B,
    0x76, 0x71, 0x78, 0x7F, 0x6A, 0x6D, 0x64, 0x63,
    0x3E, 0x39, 0x30, 0x37, 0x22, 0x25, 0x2C, 0x2B,
    0x06, 0x01, 0x08, 0x0F, 0x1A, 0x1D, 0x14, 0x13,
    0xAE, 0xA9, 0xA0, 0xA7, 0xB2, 0xB5, 0xBC, 0xBB,
    0x96, 0x91, 0x98, 0x9F, 0x8A, 0x8D, 0x84, 0x83,
    0xDE, 0xD9, 0xD0, 0xD7, 0xC2, 0xC5, 0xCC, 0xCB,
    0xE6, 0xE1, 0xE8, 0xEF, 0xFA, 0xFD, 0xF4, 0xF3
]


def xbb_generate_crc8(data: bytes) -> int:
    """
    Generate CRC8 checksum using the provided CRC table.
    
    Args:
        data: Data bytes to calculate CRC over
        
    Returns:
        CRC8 value (uint8)
    """
    val = 0
    for byte in data:
        val = CRC_TABLE[val ^ byte]
    return val


def pack_int32_be(value: int) -> bytes:
    """Pack int32 as big-endian."""
    return struct.pack('>i', int(value))


class XBBFrameEncoder:
    """
    Encodes battery pack data into XBB frame format.
    
    Data is converted to milli-units with sign preservation:
    - Current: A → milli_A (int32)
    - Voltage: V → milli_V (int32)
    - Temperature: °C → milli_degC (int32)
    """
    
    @staticmethod
    def encode_frame(
        pack_current_ma: int,      # Pack current in milli-Amperes (signed)
        pack_voltage_mv: int,      # Pack voltage in milli-Volts (unsigned, but sent as signed int32)
        temp_cell_c: float,        # Cell temperature in °C (converted to milli_degC)
        temp_pcb_c: float,         # PCB temperature in °C (converted to milli_degC)
        cell_voltages_mv: np.ndarray  # Cell voltages in milli-Volts (array[16])
    ) -> bytes:
        """
        Encode XBB frame with battery pack data.
        
        Args:
            pack_current_ma: Pack current in milli-Amperes (signed)
            pack_voltage_mv: Pack voltage in milli-Volts (signed int32)
            temp_cell_c: Cell temperature in °C (will be converted to milli_degC)
            temp_pcb_c: PCB temperature in °C (will be converted to milli_degC)
            cell_voltages_mv: Cell voltages in milli-Volts (array[16])
        
        Returns:
            Encoded frame bytes (88 bytes total)
        """
        # Validate inputs
        if len(cell_voltages_mv) != 16:
            raise ValueError(f"cell_voltages_mv must have 16 elements, got {len(cell_voltages_mv)}")
        
        # Convert temperatures to milli_degC (preserve sign)
        temp_cell_milli_degc = int(round(temp_cell_c * 1000.0))
        temp_pcb_milli_degc = int(round(temp_pcb_c * 1000.0))
        
        # Convert cell voltages to int32 (already in milli-V, preserve sign)
        # Note: voltages are typically positive, but we use signed int32 to match spec
        cell_voltages_int32 = cell_voltages_mv.astype(np.int32)
        
        # Pack data payload (20 int32 values, big-endian, 80 bytes total)
        data_payload = bytearray()
        
        # 1. pack_current_A (4 bytes, int32, milli_A)
        data_payload.extend(pack_int32_be(pack_current_ma))
        
        # 2. pack_voltage_V (4 bytes, int32, milli_V)
        data_payload.extend(pack_int32_be(pack_voltage_mv))
        
        # 3. temp_cell_C (4 bytes, int32, milli_degC)
        data_payload.extend(pack_int32_be(temp_cell_milli_degc))
        
        # 4. temp_pcb_C (4 bytes, int32, milli_degC)
        data_payload.extend(pack_int32_be(temp_pcb_milli_degc))
        
        # 5. cell_1_V through cell_16_V (16 × 4 = 64 bytes, int32, milli_V each)
        for cell_voltage in cell_voltages_int32:
            data_payload.extend(pack_int32_be(cell_voltage))
        
        # Verify data length
        if len(data_payload) != XBB_DATA_LENGTH:
            raise ValueError(f"Data payload length mismatch: got {len(data_payload)}, expected {XBB_DATA_LENGTH}")
        
        # Build frame: [0xA5] [0x33] [SubIndex: 0x0000] [DataLen: 80] [Data: 80 bytes] [0xB5] [CRC8]
        frame = bytearray()
        frame.append(XBB_FRAME_HEADER)  # 0xA5
        frame.append(XBB_FRAME_MSG_ID)  # 0x33
        frame.extend(struct.pack('>H', XBB_SUBINDEX))  # SubIndex: 0x0000 (big-endian, 2 bytes)
        frame.extend(struct.pack('>H', XBB_DATA_LENGTH))  # DataLen: 80 (big-endian, 2 bytes)
        frame.extend(data_payload)  # Data: 80 bytes
        frame.append(XBB_FRAME_FOOTER)  # 0xB5
        
        # Calculate CRC8 over all bytes from 0xA5 through 0xB5 (excluding CRC8 byte itself)
        crc_data = bytes(frame)
        crc8 = xbb_generate_crc8(crc_data)
        
        # Append CRC8
        frame.append(crc8)
        
        return bytes(frame)
    
    @staticmethod
    def print_frame_info(
        pack_current_ma: int,
        pack_voltage_mv: int,
        temp_cell_c: float,
        temp_pcb_c: float,
        cell_voltages_mv: np.ndarray,
        frame: bytes
    ):
        """
        Print frame data and hex representation for verification.
        
        Args:
            pack_current_ma: Pack current in milli-Amperes
            pack_voltage_mv: Pack voltage in milli-Volts
            temp_cell_c: Cell temperature in °C
            temp_pcb_c: PCB temperature in °C
            cell_voltages_mv: Cell voltages in milli-Volts (array[16])
            frame: Encoded frame bytes
        """
        print("\n" + "=" * 80)
        print("XBB Frame Data")
        print("=" * 80)
        
        # Print values
        print(f"\nValues:")
        print(f"  Pack Current: {pack_current_ma} milli_A ({pack_current_ma / 1000.0:.3f} A)")
        print(f"  Pack Voltage: {pack_voltage_mv} milli_V ({pack_voltage_mv / 1000.0:.3f} V)")
        print(f"  Cell Temperature: {int(round(temp_cell_c * 1000.0))} milli_degC ({temp_cell_c:.3f} °C)")
        print(f"  PCB Temperature: {int(round(temp_pcb_c * 1000.0))} milli_degC ({temp_pcb_c:.3f} °C)")
        print(f"\n  Cell Voltages (milli_V):")
        for i in range(0, 16, 4):
            cells = [f"Cell {j+1:2d}: {cell_voltages_mv[j]:6d} mV" for j in range(i, min(i+4, 16))]
            print("    " + "  |  ".join(cells))
        
        # Print hex representation
        print(f"\nFrame Hex (88 bytes):")
        hex_str = ' '.join([f'{b:02X}' for b in frame])
        # Print in groups of 16 bytes per line for readability
        for i in range(0, len(frame), 16):
            line_bytes = frame[i:i+16]
            line_hex = ' '.join([f'{b:02X}' for b in line_bytes])
            print(f"  [{i:03d}-{i+len(line_bytes)-1:03d}]: {line_hex}")
        
        # Print frame structure breakdown
        print(f"\nFrame Structure:")
        print(f"  [0x{frame[0]:02X}] Header (0xA5)")
        print(f"  [0x{frame[1]:02X}] Message ID (0x33)")
        subindex = struct.unpack('>H', frame[2:4])[0]
        print(f"  [0x{subindex:04X}] SubIndex ({subindex})")
        datalen = struct.unpack('>H', frame[4:6])[0]
        print(f"  [0x{datalen:04X}] DataLen ({datalen} bytes)")
        print(f"  [{len(frame)-8} bytes] Data payload")
        print(f"  [0x{frame[-2]:02X}] Footer (0xB5)")
        print(f"  [0x{frame[-1]:02X}] CRC8")
        
        print("=" * 80 + "\n")

