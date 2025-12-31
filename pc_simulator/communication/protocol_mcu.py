"""
UART Protocol for MCU SIL Communication

Matches MCU-side frame format:
- Frame: [0xAA][0x01][LENGTH_MSB][LENGTH_LSB][DATA...][CRC16_MSB][CRC16_LSB][0x55]
- All data: BIG-ENDIAN (network byte order)
- CRC16-CCITT on data only
"""

import struct
from typing import Optional
import numpy as np

# Frame markers (matching MCU)
SIL_FRAME_HEADER = 0xAA
SIL_FRAME_FOOTER = 0x55
SIL_FRAME_VERSION = 0x01
SIL_FRAME_OVERHEAD = 7  # header + version + length(2) + crc(2) + footer

# BMS Configuration (defaults - should match battery_system_cfg.h)
BS_NR_OF_STRINGS = 1
BS_NR_OF_MODULES_PER_STRING = 1
BS_NR_OF_CELL_BLOCKS_PER_MODULE = 16
BS_NR_OF_TEMP_SENSORS_PER_MODULE = 16
SLV_NR_OF_GPIOS_PER_MODULE = 8
SLV_NR_OF_GPAS_PER_MODULE = 4
BS_NR_OF_VOLTAGES_FROM_CURRENT_SENSOR = 2
MCU_ADC1_MAX_NR_CHANNELS = 15


def crc16_ccitt_be(data: bytes, initial: int = 0xFFFF) -> int:
    """
    Calculate CRC16-CCITT checksum (matches MCU implementation).
    
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


def pack_int16_be(value: int) -> bytes:
    """Pack int16 as big-endian."""
    return struct.pack('>h', int(value))


def pack_uint16_be(value: int) -> bytes:
    """Pack uint16 as big-endian."""
    return struct.pack('>H', int(value) & 0xFFFF)


def pack_int32_be(value: int) -> bytes:
    """Pack int32 as big-endian."""
    return struct.pack('>i', int(value))


def pack_uint32_be(value: int) -> bytes:
    """Pack uint32 as big-endian."""
    return struct.pack('>I', int(value) & 0xFFFFFFFF)


def pack_float_be(value: float) -> bytes:
    """Pack float as big-endian."""
    return struct.pack('>f', float(value))


class SILFrameEncoder:
    """
    Encodes BMS data into MCU-compatible SIL frame format.
    
    Data layout matches UART_SIL_ParseAndUpdateDatabase parsing order:
    1. Cell voltages [strings][modules][cells] (int16, 2 bytes each)
    2. Module voltages [strings][modules] (uint32, 4 bytes each)
    3. Temperatures [strings][modules][sensors] (int16, 2 bytes each)
    4. Balancing feedback [strings][modules] (uint16, 2 bytes each)
    5. Open wire count [strings] (uint16, 2 bytes) + array (uint8, 1 byte each)
    6. GPIO voltages [strings][modules*GPIOs] (int16, 2 bytes each)
    7. GPA voltages [strings][modules*GPAs] (int16, 2 bytes each)
    8. Current sensor data (int32, 4 bytes each):
       - Current [strings]
       - Sensor temp [strings]
       - Power [strings]
       - Current counter [strings]
       - Energy counter [strings]
       - High voltage taps [strings][taps]
    9. Pack values (int32, 4 bytes each):
       - Pack current
       - Battery voltage
       - HV bus voltage
       - String voltage [strings]
       - String current [strings]
       - String power [strings]
    10. Digital inputs (various)
    """
    
    def __init__(
        self,
        num_strings: int = BS_NR_OF_STRINGS,
        num_modules: int = BS_NR_OF_MODULES_PER_STRING,
        num_cells: int = BS_NR_OF_CELL_BLOCKS_PER_MODULE,
        num_temp_sensors: int = BS_NR_OF_TEMP_SENSORS_PER_MODULE
    ):
        self.num_strings = num_strings
        self.num_modules = num_modules
        self.num_cells = num_cells
        self.num_temp_sensors = num_temp_sensors
        
        # Initialize counters for current sensor
        self.current_counter_As = np.zeros(num_strings, dtype=np.float64)
        self.energy_counter_Wh = np.zeros(num_strings, dtype=np.float64)
        self.last_timestamp_ms = 0
    
    def encode_frame(
        self,
        vcell_mv: np.ndarray,  # [16] cell voltages
        tcell_cc: np.ndarray,  # [16] cell temperatures (centi-°C)
        pack_current_ma: int,
        pack_voltage_mv: int,
        status_flags: int = 0,
        timestamp_ms: int = 0,
        # Optional simulated data
        balancing_feedback: Optional[np.ndarray] = None,
        open_wire_mask: Optional[np.ndarray] = None,
        gpio_voltages: Optional[np.ndarray] = None,
        gpa_voltages: Optional[np.ndarray] = None,
        sensor_temp_ddegc: Optional[int] = None,
        hv_bus_voltage_mv: Optional[int] = None
    ) -> bytes:
        """
        Encode complete BMS data frame matching MCU format.
        
        Args:
            vcell_mv: Cell voltages in mV [16]
            tcell_cc: Cell temperatures in centi-°C [16]
            pack_current_ma: Pack current in mA
            pack_voltage_mv: Pack voltage in mV
            status_flags: Status flags (for open wire detection)
            timestamp_ms: Timestamp in milliseconds
            balancing_feedback: Optional balancing feedback [strings][modules]
            open_wire_mask: Optional open wire mask [16] (1=open wire)
            gpio_voltages: Optional GPIO voltages
            gpa_voltages: Optional GPA voltages
            sensor_temp_ddegc: Optional current sensor temperature (deci-°C)
            hv_bus_voltage_mv: Optional HV bus voltage in mV
        
        Returns:
            Complete frame bytes ready to send
        """
        # Build data payload
        data_payload = bytearray()
        
        # ===== Section 1: Cell Voltages =====
        # Reshape to [strings][modules][cells]
        vcell_3d = vcell_mv.reshape(self.num_strings, self.num_modules, self.num_cells)
        for s in range(self.num_strings):
            for m in range(self.num_modules):
                for c in range(self.num_cells):
                    # Check for open wire (status_flags bit 0-15)
                    if status_flags & (1 << c):
                        data_payload.extend(pack_int16_be(0))  # Open wire = 0
                    else:
                        data_payload.extend(pack_int16_be(int(vcell_3d[s, m, c])))
        
        # Module voltages (sum of cells per module)
        for s in range(self.num_strings):
            for m in range(self.num_modules):
                module_voltage = int(np.sum(vcell_3d[s, m, :]))
                data_payload.extend(pack_uint32_be(module_voltage))
        
        # ===== Section 2: Temperatures =====
        tcell_3d = tcell_cc.reshape(self.num_strings, self.num_modules, self.num_temp_sensors)
        for s in range(self.num_strings):
            for m in range(self.num_modules):
                for t in range(self.num_temp_sensors):
                    # Check for NTC fault (status_flags bit 16-31)
                    if status_flags & (1 << (16 + t)):
                        data_payload.extend(pack_int16_be(0))  # NTC fault = 0
                    else:
                        data_payload.extend(pack_int16_be(int(tcell_3d[s, m, t])))
        
        # ===== Section 3: Balancing Feedback =====
        if balancing_feedback is not None:
            for s in range(self.num_strings):
                for m in range(self.num_modules):
                    data_payload.extend(pack_uint16_be(int(balancing_feedback[s, m])))
        else:
            # Default: all zeros
            for s in range(self.num_strings):
                for m in range(self.num_modules):
                    data_payload.extend(pack_uint16_be(0))
        
        # ===== Section 4: Open Wire =====
        for s in range(self.num_strings):
            # Count open wires
            if open_wire_mask is not None:
                nr_open_wires = int(np.sum(open_wire_mask))
            else:
                # Count from status_flags bits 0-15
                nr_open_wires = bin(status_flags & 0xFFFF).count('1')
            
            data_payload.extend(pack_uint16_be(nr_open_wires))
            
            # Open wire array: [modules * (cells + 1)]
            ow_array_size = self.num_modules * (self.num_cells + 1)
            for i in range(ow_array_size):
                if i < self.num_cells:
                    # Check status_flags bit
                    is_open = bool(status_flags & (1 << i))
                    data_payload.append(1 if is_open else 0)
                else:
                    # Inter-module connections (not used for single module)
                    data_payload.append(0)
        
        # ===== Section 5: GPIO Voltages =====
        if gpio_voltages is not None:
            for s in range(self.num_strings):
                for idx in range(self.num_modules * SLV_NR_OF_GPIOS_PER_MODULE):
                    if idx < len(gpio_voltages):
                        data_payload.extend(pack_int16_be(int(gpio_voltages[idx])))
                    else:
                        data_payload.extend(pack_int16_be(0))
        else:
            # Default: all zeros
            for s in range(self.num_strings):
                for _ in range(self.num_modules * SLV_NR_OF_GPIOS_PER_MODULE):
                    data_payload.extend(pack_int16_be(0))
        
        # ===== Section 6: GPA Voltages =====
        if gpa_voltages is not None:
            for s in range(self.num_strings):
                for idx in range(self.num_modules * SLV_NR_OF_GPAS_PER_MODULE):
                    if idx < len(gpa_voltages):
                        data_payload.extend(pack_int16_be(int(gpa_voltages[idx])))
                    else:
                        data_payload.extend(pack_int16_be(0))
        else:
            # Default: all zeros
            for s in range(self.num_strings):
                for _ in range(self.num_modules * SLV_NR_OF_GPAS_PER_MODULE):
                    data_payload.extend(pack_int16_be(0))
        
        # ===== Section 7: Current Sensor Data =====
        # Update counters
        dt_s = (timestamp_ms - self.last_timestamp_ms) / 1000.0 if self.last_timestamp_ms > 0 else 0.02
        current_A = pack_current_ma / 1000.0
        power_W = (pack_voltage_mv * pack_current_ma) / 1000000.0
        
        for s in range(self.num_strings):
            # Current (mA)
            data_payload.extend(pack_int32_be(pack_current_ma))
            
            # Sensor temperature (deci-°C)
            sensor_temp = sensor_temp_ddegc if sensor_temp_ddegc is not None else 250  # 25.0°C default
            data_payload.extend(pack_int32_be(sensor_temp))
            
            # Power (W)
            data_payload.extend(pack_int32_be(int(power_W)))
            
            # Current counter (As) - Coulomb counting
            self.current_counter_As[s] += current_A * dt_s
            data_payload.extend(pack_int32_be(int(self.current_counter_As[s])))
            
            # Energy counter (Wh)
            self.energy_counter_Wh[s] += power_W * dt_s / 3600.0
            data_payload.extend(pack_int32_be(int(self.energy_counter_Wh[s])))
            
            # High voltage taps (mV)
            for hv_idx in range(BS_NR_OF_VOLTAGES_FROM_CURRENT_SENSOR):
                # Simulate HV taps (could be pack voltage / 2, etc.)
                hv_value = pack_voltage_mv // 2 if hv_idx == 0 else pack_voltage_mv // 4
                data_payload.extend(pack_int32_be(hv_value))
        
        self.last_timestamp_ms = timestamp_ms
        
        # ===== Section 8: Pack/String Values =====
        # Pack current (mA)
        data_payload.extend(pack_int32_be(pack_current_ma))
        
        # Battery voltage (mV)
        data_payload.extend(pack_int32_be(pack_voltage_mv))
        
        # HV bus voltage (mV)
        hv_bus = hv_bus_voltage_mv if hv_bus_voltage_mv is not None else pack_voltage_mv
        data_payload.extend(pack_int32_be(hv_bus))
        
        # String voltage (mV) - one per string
        for s in range(self.num_strings):
            string_voltage = int(np.sum(vcell_3d[s, :, :]))
            data_payload.extend(pack_int32_be(string_voltage))
        
        # String current (mA) - one per string
        for s in range(self.num_strings):
            data_payload.extend(pack_int32_be(pack_current_ma))
        
        # String power (W) - one per string
        for s in range(self.num_strings):
            string_voltage = int(np.sum(vcell_3d[s, :, :]))
            string_power = int((string_voltage * pack_current_ma) / 1000)
            data_payload.extend(pack_int32_be(string_power))
        
        # ===== Section 9: Digital/Analog Inputs =====
        # Contactor feedback (4 bytes, packed)
        data_payload.extend(pack_uint32_be(0))  # Default: all contactors open
        
        # Interlock state (1 byte)
        data_payload.append(0)  # Default: interlock OK
        
        # Interlock voltages (2 x float, 4 bytes each)
        data_payload.extend(pack_float_be(0.0))  # HS
        data_payload.extend(pack_float_be(0.0))  # LS
        
        # Interlock currents (2 x float, 4 bytes each)
        data_payload.extend(pack_float_be(0.0))  # HS
        data_payload.extend(pack_float_be(0.0))  # LS
        
        # IMD resistance (4 bytes, uint32)
        data_payload.extend(pack_uint32_be(0))  # Default: no fault
        
        # IMD flags (1 byte)
        data_payload.append(0)  # Default: no fault
        
        # IMD sample voltages (2 x float, 4 bytes each)
        data_payload.extend(pack_float_be(0.0))
        data_payload.extend(pack_float_be(0.0))
        
        # HTSEN temperature (2 bytes, int16, deci-°C)
        data_payload.extend(pack_int16_be(250))  # 25.0°C default
        
        # HTSEN humidity (1 byte, uint8, percentage)
        data_payload.append(50)  # 50% default
        
        # MCU ADC1 voltages (MCU_ADC1_MAX_NR_CHANNELS x float, 4 bytes each)
        for _ in range(MCU_ADC1_MAX_NR_CHANNELS):
            data_payload.extend(pack_float_be(0.0))
        
        # Aerosol sensor (5 bytes: 1 status + 2 PM + 1 flags + 1 CRC)
        data_payload.extend(bytes([0, 0, 0, 0, 0]))
        
        # ===== Build Complete Frame =====
        data_length = len(data_payload)
        
        # Calculate CRC on data only (matching MCU: CRC on data from offset 4)
        crc = crc16_ccitt_be(data_payload, 0xFFFF)
        
        # Build frame: [HEADER][VERSION][LENGTH_MSB][LENGTH_LSB][DATA...][CRC16_MSB][CRC16_LSB][FOOTER]
        frame = bytearray()
        frame.append(SIL_FRAME_HEADER)  # 0xAA
        frame.append(SIL_FRAME_VERSION)  # 0x01
        frame.extend(pack_uint16_be(data_length))  # Length (big-endian)
        frame.extend(data_payload)  # Data
        frame.extend(pack_uint16_be(crc))  # CRC (big-endian)
        frame.append(SIL_FRAME_FOOTER)  # 0x55
        
        return bytes(frame)
    
    def reset_counters(self):
        """Reset current and energy counters."""
        self.current_counter_As.fill(0)
        self.energy_counter_Wh.fill(0)
        self.last_timestamp_ms = 0

