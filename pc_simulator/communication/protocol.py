"""
UART Protocol Definitions

Binary UART protocol for SIL BMS communication:
- Frame structure: SOF | msg_id | len | seq | payload | CRC16 | EOF
- Message types: AFE_MEAS_FRAME (PC→MCU), BMS_APP_FRAME (MCU→PC)
- CRC16-CCITT algorithm
"""

import struct
from typing import Tuple, Optional
import numpy as np


# Frame markers
SOF = 0xA5  # Start of Frame
EOF = 0xAA  # End of Frame

# Message IDs
MSG_ID_AFE_MEAS = 0x01  # AFE_MEAS_FRAME (PC→MCU)
MSG_ID_BMS_APP = 0x02   # BMS_APP_FRAME (MCU→PC)


def crc16_ccitt(data: bytes, initial: int = 0xFFFF) -> int:
    """
    Calculate CRC16-CCITT checksum.
    
    Parameters:
        polynomial: 0x1021
        initial: 0xFFFF
        no XOR out
    
    Args:
        data: Data bytes
        initial: Initial CRC value (default: 0xFFFF)
    
    Returns:
        CRC16-CCITT value (uint16)
    """
    polynomial = 0x1021
    crc = initial
    
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ polynomial
            else:
                crc <<= 1
            crc &= 0xFFFF
    
    return crc


class AFEMeasFrame:
    """AFE_MEAS_FRAME structure (PC→MCU)."""
    
    MSG_ID = MSG_ID_AFE_MEAS
    
    # Payload structure (packed, little-endian)
    # uint32 timestamp_ms
    # uint16 vcell_mV[16]
    # int16  tcell_cC[16]  (centi-°C)
    # int32  pack_current_mA
    # uint32 pack_voltage_mV
    # uint32 status_flags
    
    PAYLOAD_FORMAT = '<I' + 'H' * 16 + 'h' * 16 + 'iII'  # Little-endian
    PAYLOAD_SIZE = 4 + 16*2 + 16*2 + 4 + 4 + 4  # 80 bytes
    
    @staticmethod
    def encode(
        timestamp_ms: int,
        vcell_mv: np.ndarray,
        tcell_cc: np.ndarray,
        pack_current_ma: int,
        pack_voltage_mv: int,
        status_flags: int,
        sequence: int
    ) -> bytes:
        """
        Encode AFE_MEAS_FRAME to bytes.
        
        Args:
            timestamp_ms: Timestamp in milliseconds
            vcell_mv: Cell voltages in mV (array[16])
            tcell_cc: Cell temperatures in centi-°C (array[16])
            pack_current_ma: Pack current in mA
            pack_voltage_mv: Pack voltage in mV
            status_flags: Status flags (uint32)
            sequence: Sequence number
        
        Returns:
            Encoded frame bytes
        """
        # Validate inputs
        if len(vcell_mv) != 16:
            raise ValueError(f"vcell_mv must have 16 elements, got {len(vcell_mv)}")
        if len(tcell_cc) != 16:
            raise ValueError(f"tcell_cc must have 16 elements, got {len(tcell_cc)}")
        
        # Pack payload
        payload = struct.pack(
            AFEMeasFrame.PAYLOAD_FORMAT,
            timestamp_ms,
            *vcell_mv.astype(np.uint16),
            *tcell_cc.astype(np.int16),
            pack_current_ma,
            pack_voltage_mv,
            status_flags
        )
        
        # Build frame: SOF | msg_id | len | seq | payload | CRC16 | EOF
        msg_id = AFEMeasFrame.MSG_ID
        length = len(payload)
        seq = sequence & 0xFFFF  # Wrap at 65535
        
        # Calculate CRC on: msg_id | len | seq | payload
        crc_data = struct.pack('<BBH', msg_id, length, seq) + payload
        crc = crc16_ccitt(crc_data)
        
        # Assemble frame
        frame = struct.pack('<B', SOF) + \
                struct.pack('<BBH', msg_id, length, seq) + \
                payload + \
                struct.pack('<H', crc) + \
                struct.pack('<B', EOF)
        
        return frame
    
    @staticmethod
    def decode(frame: bytes) -> Optional[dict]:
        """
        Decode AFE_MEAS_FRAME from bytes.
        
        Args:
            frame: Frame bytes
        
        Returns:
            Dictionary with decoded data or None if invalid
        """
        # Check minimum frame size
        if len(frame) < 7:  # SOF + msg_id + len + seq(2) + CRC(2) + EOF
            return None
        
        # Check SOF and EOF
        if frame[0] != SOF or frame[-1] != EOF:
            return None
        
        # Extract header
        msg_id, length, seq = struct.unpack('<BBH', frame[1:5])
        
        if msg_id != AFEMeasFrame.MSG_ID:
            return None
        
        # Check frame size
        expected_size = 1 + 4 + length + 2 + 1  # SOF + header + payload + CRC + EOF
        if len(frame) != expected_size:
            return None
        
        # Extract payload
        payload = frame[5:5+length]
        
        # Verify CRC
        crc_data = frame[1:5+length]  # msg_id | len | seq | payload
        expected_crc = crc16_ccitt(crc_data)
        received_crc = struct.unpack('<H', frame[5+length:5+length+2])[0]
        
        if expected_crc != received_crc:
            return None
        
        # Unpack payload
        unpacked = struct.unpack(AFEMeasFrame.PAYLOAD_FORMAT, payload)
        
        timestamp_ms = unpacked[0]
        vcell_mv = np.array(unpacked[1:17], dtype=np.uint16)
        tcell_cc = np.array(unpacked[17:33], dtype=np.int16)
        pack_current_ma = unpacked[33]
        pack_voltage_mv = unpacked[34]
        status_flags = unpacked[35]
        
        return {
            'timestamp_ms': timestamp_ms,
            'vcell_mv': vcell_mv,
            'tcell_cc': tcell_cc,
            'pack_current_ma': pack_current_ma,
            'pack_voltage_mv': pack_voltage_mv,
            'status_flags': status_flags,
            'sequence': seq
        }


def validate_afe_meas_data(data: dict) -> Tuple[bool, Optional[str]]:
    """
    Validate AFE measurement data before encoding.
    
    Args:
        data: Dictionary with measurement data
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ['timestamp_ms', 'vcell_mv', 'tcell_cc', 'pack_current_ma', 
                      'pack_voltage_mv', 'status_flags']
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate vcell_mv
    vcell = data['vcell_mv']
    if not isinstance(vcell, np.ndarray) or len(vcell) != 16:
        return False, "vcell_mv must be numpy array with 16 elements"
    if np.any(vcell < 0) or np.any(vcell > 65535):
        return False, "vcell_mv values must be in range [0, 65535] mV"
    
    # Validate tcell_cc
    tcell = data['tcell_cc']
    if not isinstance(tcell, np.ndarray) or len(tcell) != 16:
        return False, "tcell_cc must be numpy array with 16 elements"
    if np.any(tcell < -32768) or np.any(tcell > 32767):
        return False, "tcell_cc values must be in range [-32768, 32767] centi-°C"
    
    # Validate pack_current_ma
    current = data['pack_current_ma']
    if not isinstance(current, (int, float)):
        return False, "pack_current_ma must be numeric"
    if current < -2147483648 or current > 2147483647:
        return False, "pack_current_ma must be in range [-2147483648, 2147483647] mA"
    
    # Validate pack_voltage_mv
    voltage = data['pack_voltage_mv']
    if not isinstance(voltage, (int, float)):
        return False, "pack_voltage_mv must be numeric"
    if voltage < 0 or voltage > 4294967295:
        return False, "pack_voltage_mv must be in range [0, 4294967295] mV"
    
    # Validate status_flags
    flags = data['status_flags']
    if not isinstance(flags, int):
        return False, "status_flags must be integer"
    if flags < 0 or flags > 4294967295:
        return False, "status_flags must be in range [0, 4294967295]"
    
    return True, None

