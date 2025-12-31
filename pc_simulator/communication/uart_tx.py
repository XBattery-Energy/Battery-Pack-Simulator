"""
UART Transmitter for PC Simulator

Sends AFE_MEAS_FRAME to MCU via UART with:
- Frame encoding and CRC16-CCITT
- Rate limiting (configurable Hz)
- Thread-safe queue
- Error handling and retry logic
- Statistics and logging
"""

import serial
import serial.tools.list_ports
import threading
import queue
import time
import logging
from typing import Optional, Dict
import numpy as np
from communication.protocol import (
    AFEMeasFrame,
    validate_afe_meas_data,
    crc16_ccitt
)


class UARTTransmitter:
    """
    UART Transmitter for AFE measurement frames.
    
    Features:
    - Frame encoding with CRC16-CCITT
    - Rate limiting (configurable Hz)
    - Thread-safe queue for frame transmission
    - Error handling with retry logic
    - Statistics tracking
    - Optional verbose logging
    """
    
    def __init__(
        self,
        port: str,
        baudrate: int = 921600,
        frame_rate_hz: float = 50.0,
        timeout: float = 1.0,
        retry_max: int = 3,
        retry_backoff: float = 0.1,
        verbose: bool = False
    ):
        """
        Initialize UART transmitter.
        
        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate: Baud rate (default: 921600, recommend for 50 Hz)
            frame_rate_hz: Frame transmission rate in Hz (default: 50.0)
            timeout: Serial port timeout in seconds (default: 1.0)
            retry_max: Maximum retry attempts (default: 3)
            retry_backoff: Retry backoff multiplier (default: 0.1)
            verbose: Enable verbose logging (default: False)
        """
        self._port = port
        self._baudrate = baudrate
        self._frame_rate_hz = frame_rate_hz
        self._frame_interval_sec = 1.0 / frame_rate_hz if frame_rate_hz > 0 else 0.0
        self._timeout = timeout
        self._retry_max = retry_max
        self._retry_backoff = retry_backoff
        self._verbose = verbose
        
        # Serial port
        self._serial: Optional[serial.Serial] = None
        
        # Threading
        self._tx_thread: Optional[threading.Thread] = None
        self._tx_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # Statistics
        self._sent_count = 0
        self._error_count = 0
        self._last_error: Optional[str] = None
        self._sequence = 0
        self._last_send_time = 0.0
        
        # Logging
        self._logger = logging.getLogger(__name__)
        if verbose:
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.INFO)
    
    def _open_serial_port(self) -> bool:
        """
        Open serial port with error handling.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self._serial is not None and self._serial.is_open:
                return True
            
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                timeout=self._timeout,
                write_timeout=self._timeout
            )
            
            if self._verbose:
                self._logger.debug(f"Opened serial port: {self._port} at {self._baudrate} baud")
            
            return True
        
        except serial.SerialException as e:
            self._last_error = f"Serial port error: {e}"
            self._logger.error(self._last_error)
            return False
        
        except Exception as e:
            self._last_error = f"Unexpected error opening serial port: {e}"
            self._logger.error(self._last_error)
            return False
    
    def _close_serial_port(self):
        """Close serial port."""
        if self._serial is not None and self._serial.is_open:
            try:
                self._serial.close()
                if self._verbose:
                    self._logger.debug(f"Closed serial port: {self._port}")
            except Exception as e:
                self._logger.warning(f"Error closing serial port: {e}")
    
    def _send_frame_with_retry(self, frame: bytes) -> bool:
        """
        Send frame with retry logic.
        
        Args:
            frame: Frame bytes to send
        
        Returns:
            True if successful, False otherwise
        """
        if self._serial is None or not self._serial.is_open:
            if not self._open_serial_port():
                return False
        
        for attempt in range(self._retry_max):
            try:
                bytes_written = self._serial.write(frame)
                
                if bytes_written != len(frame):
                    raise serial.SerialTimeoutException(
                        f"Only wrote {bytes_written} of {len(frame)} bytes"
                    )
                
                self._serial.flush()  # Ensure data is sent
                
                if self._verbose:
                    self._logger.debug(f"Sent frame: {len(frame)} bytes, sequence: {self._sequence}")
                
                return True
            
            except serial.SerialTimeoutException as e:
                self._last_error = f"Send timeout (attempt {attempt + 1}/{self._retry_max}): {e}"
                self._logger.warning(self._last_error)
                
                if attempt < self._retry_max - 1:
                    # Exponential backoff
                    backoff_time = self._retry_backoff * (2 ** attempt)
                    time.sleep(backoff_time)
                    
                    # Try to reopen port
                    self._close_serial_port()
                    if not self._open_serial_port():
                        break
            
            except serial.SerialException as e:
                self._last_error = f"Serial error (attempt {attempt + 1}/{self._retry_max}): {e}"
                self._logger.warning(self._last_error)
                
                if attempt < self._retry_max - 1:
                    # Exponential backoff
                    backoff_time = self._retry_backoff * (2 ** attempt)
                    time.sleep(backoff_time)
                    
                    # Try to reopen port
                    self._close_serial_port()
                    if not self._open_serial_port():
                        break
            
            except Exception as e:
                self._last_error = f"Unexpected error (attempt {attempt + 1}/{self._retry_max}): {e}"
                self._logger.error(self._last_error)
                break
        
        self._error_count += 1
        return False
    
    def _tx_thread_worker(self):
        """Transmission thread worker function."""
        if self._verbose:
            self._logger.debug("TX thread started")
        
        while not self._stop_event.is_set():
            try:
                # Get frame from queue (with timeout to check stop event)
                try:
                    frame_data = self._tx_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Rate limiting: wait if needed
                current_time = time.time()
                time_since_last = current_time - self._last_send_time
                
                if time_since_last < self._frame_interval_sec:
                    sleep_time = self._frame_interval_sec - time_since_last
                    time.sleep(sleep_time)
                
                # Send frame
                success = self._send_frame_with_retry(frame_data['frame'])
                
                if success:
                    self._sent_count += 1
                    self._last_send_time = time.time()
                else:
                    self._error_count += 1
                
                # Mark task as done
                self._tx_queue.task_done()
            
            except Exception as e:
                self._last_error = f"TX thread error: {e}"
                self._logger.error(self._last_error)
                self._error_count += 1
        
        # Close serial port when thread stops
        self._close_serial_port()
        
        if self._verbose:
            self._logger.debug("TX thread stopped")
    
    def send_frame(self, afe_meas_data: dict) -> bool:
        """
        Send AFE measurement frame.
        
        Args:
            afe_meas_data: Dictionary with measurement data:
                - timestamp_ms: int
                - vcell_mv: numpy array[16] (uint16, mV)
                - tcell_cc: numpy array[16] (int16, centi-°C)
                - pack_current_ma: int (mA)
                - pack_voltage_mv: int (mV)
                - status_flags: int (uint32)
        
        Returns:
            True if frame queued successfully, False otherwise
        """
        # Validate data
        is_valid, error_msg = validate_afe_meas_data(afe_meas_data)
        if not is_valid:
            self._last_error = f"Frame validation failed: {error_msg}"
            self._logger.error(self._last_error)
            self._error_count += 1
            return False
        
        # Encode frame
        try:
            frame = AFEMeasFrame.encode(
                timestamp_ms=afe_meas_data['timestamp_ms'],
                vcell_mv=afe_meas_data['vcell_mv'],
                tcell_cc=afe_meas_data['tcell_cc'],
                pack_current_ma=int(afe_meas_data['pack_current_ma']),
                pack_voltage_mv=int(afe_meas_data['pack_voltage_mv']),
                status_flags=afe_meas_data['status_flags'],
                sequence=self._sequence
            )
            
            # Increment sequence (wrap at 65535)
            self._sequence = (self._sequence + 1) & 0xFFFF
            
        except Exception as e:
            self._last_error = f"Frame encoding error: {e}"
            self._logger.error(self._last_error)
            self._error_count += 1
            return False
        
        # Queue frame for transmission
        try:
            self._tx_queue.put_nowait({
                'frame': frame,
                'timestamp': time.time()
            })
            return True
        
        except queue.Full:
            self._last_error = "TX queue full"
            self._logger.warning(self._last_error)
            self._error_count += 1
            return False
    
    def start(self) -> bool:
        """
        Start transmission thread.
        
        Returns:
            True if started successfully, False otherwise
        """
        with self._lock:
            if self._tx_thread is not None and self._tx_thread.is_alive():
                self._logger.warning("TX thread already running")
                return False
            
            # Open serial port
            if not self._open_serial_port():
                return False
            
            # Reset stop event
            self._stop_event.clear()
            
            # Start thread
            self._tx_thread = threading.Thread(target=self._tx_thread_worker, daemon=True)
            self._tx_thread.start()
            
            if self._verbose:
                self._logger.info(f"UART transmitter started: {self._port} at {self._baudrate} baud, {self._frame_rate_hz} Hz")
            
            return True
    
    def stop(self):
        """Stop transmission thread."""
        with self._lock:
            if self._tx_thread is None or not self._tx_thread.is_alive():
                return
            
            # Signal stop
            self._stop_event.set()
            
            # Wait for thread to finish (with timeout)
            self._tx_thread.join(timeout=2.0)
            
            if self._tx_thread.is_alive():
                self._logger.warning("TX thread did not stop within timeout")
            
            # Close serial port
            self._close_serial_port()
            
            if self._verbose:
                self._logger.info("UART transmitter stopped")
    
    def get_statistics(self) -> dict:
        """
        Get transmission statistics.
        
        Returns:
            Dictionary with statistics:
            - sent_count: Number of frames sent successfully
            - error_count: Number of errors
            - last_error: Last error message (if any)
            - sequence: Current sequence number
            - queue_size: Current queue size
        """
        with self._lock:
            return {
                'sent_count': self._sent_count,
                'error_count': self._error_count,
                'last_error': self._last_error,
                'sequence': self._sequence,
                'queue_size': self._tx_queue.qsize()
            }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        with self._lock:
            self._sent_count = 0
            self._error_count = 0
            self._last_error = None
            # Note: sequence is not reset (should continue from current value)
    
    @staticmethod
    def list_available_ports() -> list:
        """
        List available serial ports.
        
        Returns:
            List of available port names
        """
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]

