"""
UART Transmitter for MCU-compatible SIL frames

Extends base UARTTransmitter to use MCU frame format.
"""

import queue
import time
import logging
from typing import Optional, Dict
import numpy as np
from communication.uart_tx import UARTTransmitter
from communication.protocol_mcu import SILFrameEncoder


class MCUCompatibleUARTTransmitter(UARTTransmitter):
    """
    UART Transmitter that sends MCU-compatible SIL frames.
    Extends base UARTTransmitter with MCU frame format.
    """
    
    def __init__(
        self,
        port: str,
        baudrate: int = 921600,
        frame_rate_hz: float = 50.0,
        timeout: float = 1.0,
        retry_max: int = 3,
        retry_backoff: float = 0.1,
        verbose: bool = False,
        num_strings: int = 1,
        num_modules: int = 1,
        num_cells: int = 16,
        num_temp_sensors: int = 16
    ):
        """
        Initialize MCU-compatible UART transmitter.
        
        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate: Baud rate (default: 921600)
            frame_rate_hz: Frame transmission rate in Hz (default: 50.0)
            timeout: Serial port timeout in seconds (default: 1.0)
            retry_max: Maximum retry attempts (default: 3)
            retry_backoff: Retry backoff multiplier (default: 0.1)
            verbose: Enable verbose logging (default: False)
            num_strings: Number of battery strings (default: 1)
            num_modules: Number of modules per string (default: 1)
            num_cells: Number of cells per module (default: 16)
            num_temp_sensors: Number of temperature sensors per module (default: 16)
        """
        # Initialize base class
        super().__init__(
            port=port,
            baudrate=baudrate,
            frame_rate_hz=frame_rate_hz,
            timeout=timeout,
            retry_max=retry_max,
            retry_backoff=retry_backoff,
            verbose=verbose
        )
        
        # Create MCU frame encoder
        self.frame_encoder = SILFrameEncoder(
            num_strings=num_strings,
            num_modules=num_modules,
            num_cells=num_cells,
            num_temp_sensors=num_temp_sensors
        )
    
    def send_frame(self, frame_data: dict) -> bool:
        """
        Send frame using MCU-compatible format.
        
        Args:
            frame_data: Dictionary with keys:
                - vcell_mv: array[16] cell voltages in mV
                - tcell_cc: array[16] cell temperatures in centi-°C
                - pack_current_ma: pack current in mA
                - pack_voltage_mv: pack voltage in mV
                - status_flags: fault flags (optional, default: 0)
                - timestamp_ms: timestamp in ms (optional)
                - balancing_feedback: optional balancing feedback array
                - open_wire_mask: optional open wire mask array
                - gpio_voltages: optional GPIO voltages array
                - gpa_voltages: optional GPA voltages array
                - sensor_temp_ddegc: optional sensor temperature (deci-°C)
                - hv_bus_voltage_mv: optional HV bus voltage in mV
        
        Returns:
            True if frame queued successfully
        """
        # Validate required fields
        required_fields = ['vcell_mv', 'tcell_cc', 'pack_current_ma', 'pack_voltage_mv']
        for field in required_fields:
            if field not in frame_data:
                self._logger.error(f"Missing required field: {field}")
                return False
        
        # Encode frame using MCU format
        try:
            frame_bytes = self.frame_encoder.encode_frame(
                vcell_mv=frame_data['vcell_mv'],
                tcell_cc=frame_data['tcell_cc'],
                pack_current_ma=frame_data['pack_current_ma'],
                pack_voltage_mv=frame_data['pack_voltage_mv'],
                status_flags=frame_data.get('status_flags', 0),
                timestamp_ms=frame_data.get('timestamp_ms', 0),
                balancing_feedback=frame_data.get('balancing_feedback'),
                open_wire_mask=frame_data.get('open_wire_mask'),
                gpio_voltages=frame_data.get('gpio_voltages'),
                gpa_voltages=frame_data.get('gpa_voltages'),
                sensor_temp_ddegc=frame_data.get('sensor_temp_ddegc'),
                hv_bus_voltage_mv=frame_data.get('hv_bus_voltage_mv')
            )
            
            # Queue frame for transmission (using base class queue format)
            try:
                self._tx_queue.put_nowait({
                    'frame': frame_bytes,
                    'timestamp': time.time()
                })
                return True
            except queue.Full:
                self._logger.warning("Frame queue full, dropping frame")
                return False
                
        except Exception as e:
            self._logger.error(f"Error encoding frame: {e}")
            return False
    
    def reset_counters(self):
        """Reset current and energy counters."""
        self.frame_encoder.reset_counters()

