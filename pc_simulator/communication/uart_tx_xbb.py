"""
UART Transmitter for XBB Protocol

Sends battery pack data using XBB frame format at 1Hz.
"""

import queue
import time
import logging
from typing import Optional, Dict
import numpy as np
from communication.uart_tx import UARTTransmitter
from communication.protocol_xbb import XBBFrameEncoder


class XBBUARTTransmitter(UARTTransmitter):
    """
    UART Transmitter that sends XBB protocol frames.
    Extends base UARTTransmitter with XBB frame format.
    """
    
    def __init__(
        self,
        port: str,
        baudrate: int = 921600,
        frame_rate_hz: float = 1.0,  # Default 1Hz for XBB protocol
        timeout: float = 1.0,
        retry_max: int = 3,
        retry_backoff: float = 0.1,
        verbose: bool = False,
        print_frames: bool = True
    ):
        """
        Initialize XBB UART transmitter.
        
        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate: Baud rate (default: 921600)
            frame_rate_hz: Frame transmission rate in Hz (default: 1.0 for XBB)
            timeout: Serial port timeout in seconds (default: 1.0)
            retry_max: Maximum retry attempts (default: 3)
            retry_backoff: Retry backoff multiplier (default: 0.1)
            verbose: Enable verbose logging (default: False)
            print_frames: Print frame data and hex (default: True)
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
        
        self._print_frames = print_frames
    
    def send_frame(self, frame_data: dict) -> bool:
        """
        Send frame using XBB protocol format.
        
        Args:
            frame_data: Dictionary with keys:
                - pack_current_ma: pack current in milli-Amperes (signed)
                - pack_voltage_mv: pack voltage in milli-Volts
                - temp_cell_c: cell temperature in °C (float)
                - temp_pcb_c: PCB temperature in °C (float)
                - cell_voltages_mv: cell voltages in milli-Volts (array[16])
        
        Returns:
            True if frame queued successfully
        """
        # Validate required fields
        required_fields = ['pack_current_ma', 'pack_voltage_mv', 'temp_cell_c', 'temp_pcb_c', 'cell_voltages_mv']
        for field in required_fields:
            if field not in frame_data:
                self._logger.error(f"Missing required field: {field}")
                return False
        
        # Validate cell voltages array
        cell_voltages = frame_data['cell_voltages_mv']
        if not isinstance(cell_voltages, np.ndarray) or len(cell_voltages) != 16:
            self._logger.error(f"cell_voltages_mv must be numpy array with 16 elements, got {type(cell_voltages)} with length {len(cell_voltages) if hasattr(cell_voltages, '__len__') else 'N/A'}")
            return False
        
        # Encode frame using XBB format
        try:
            frame_bytes = XBBFrameEncoder.encode_frame(
                pack_current_ma=int(frame_data['pack_current_ma']),
                pack_voltage_mv=int(frame_data['pack_voltage_mv']),
                temp_cell_c=float(frame_data['temp_cell_c']),
                temp_pcb_c=float(frame_data['temp_pcb_c']),
                cell_voltages_mv=cell_voltages
            )
            
            # Print frame info if enabled
            if self._print_frames:
                XBBFrameEncoder.print_frame_info(
                    pack_current_ma=int(frame_data['pack_current_ma']),
                    pack_voltage_mv=int(frame_data['pack_voltage_mv']),
                    temp_cell_c=float(frame_data['temp_cell_c']),
                    temp_pcb_c=float(frame_data['temp_pcb_c']),
                    cell_voltages_mv=cell_voltages,
                    frame=frame_bytes
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
            self._logger.error(f"Error encoding XBB frame: {e}")
            return False

