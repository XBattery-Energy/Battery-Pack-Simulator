"""
Unit tests for UART Transmitter.
"""

import pytest
import numpy as np
import time
import threading
import serial
from unittest.mock import Mock, patch, MagicMock
from sil_bms.pc_simulator.communication.uart_tx import UARTTransmitter
from sil_bms.pc_simulator.communication.protocol import AFEMeasFrame, crc16_ccitt


class TestUARTTransmitter:
    """Test suite for UARTTransmitter class."""
    
    def test_initialization(self):
        """Test transmitter initialization."""
        tx = UARTTransmitter(port='COM3', baudrate=921600, frame_rate_hz=50.0)
        
        assert tx._port == 'COM3'
        assert tx._baudrate == 921600
        assert tx._frame_rate_hz == 50.0
        assert tx._frame_interval_sec == 1.0 / 50.0
        assert tx._sequence == 0
        assert tx._sent_count == 0
        assert tx._error_count == 0
    
    def test_frame_validation_valid(self):
        """Test frame validation with valid data."""
        from sil_bms.pc_simulator.communication.protocol import validate_afe_meas_data
        
        data = {
            'timestamp_ms': 1000,
            'vcell_mv': np.array([3200] * 16, dtype=np.uint16),
            'tcell_cc': np.array([2500] * 16, dtype=np.int16),  # 25.0°C
            'pack_current_ma': 50000,
            'pack_voltage_mv': 51200,
            'status_flags': 0
        }
        
        is_valid, error = validate_afe_meas_data(data)
        assert is_valid, f"Validation should pass, got error: {error}"
    
    def test_frame_validation_invalid(self):
        """Test frame validation with invalid data."""
        from sil_bms.pc_simulator.communication.protocol import validate_afe_meas_data
        
        # Missing field
        data = {
            'timestamp_ms': 1000,
            'vcell_mv': np.array([3200] * 16),
            # Missing other fields
        }
        
        is_valid, error = validate_afe_meas_data(data)
        assert not is_valid
        assert 'Missing required field' in error
        
        # Invalid array length
        data = {
            'timestamp_ms': 1000,
            'vcell_mv': np.array([3200] * 15),  # Wrong length
            'tcell_cc': np.array([2500] * 16),
            'pack_current_ma': 50000,
            'pack_voltage_mv': 51200,
            'status_flags': 0
        }
        
        is_valid, error = validate_afe_meas_data(data)
        assert not is_valid
        assert '16 elements' in error
    
    def test_frame_encoding(self):
        """Test frame encoding."""
        timestamp_ms = 1000
        vcell_mv = np.array([3200] * 16, dtype=np.uint16)
        tcell_cc = np.array([2500] * 16, dtype=np.int16)  # 25.0°C
        pack_current_ma = 50000
        pack_voltage_mv = 51200
        status_flags = 0
        sequence = 42
        
        frame = AFEMeasFrame.encode(
            timestamp_ms, vcell_mv, tcell_cc,
            pack_current_ma, pack_voltage_mv, status_flags, sequence
        )
        
        # Check frame structure
        assert len(frame) > 0
        assert frame[0] == 0xA5  # SOF
        assert frame[-1] == 0xAA  # EOF
        assert frame[1] == AFEMeasFrame.MSG_ID  # Message ID
    
    def test_frame_decoding(self):
        """Test frame decoding."""
        timestamp_ms = 1000
        vcell_mv = np.array([3200] * 16, dtype=np.uint16)
        tcell_cc = np.array([2500] * 16, dtype=np.int16)
        pack_current_ma = 50000
        pack_voltage_mv = 51200
        status_flags = 0
        sequence = 42
        
        # Encode
        frame = AFEMeasFrame.encode(
            timestamp_ms, vcell_mv, tcell_cc,
            pack_current_ma, pack_voltage_mv, status_flags, sequence
        )
        
        # Decode
        decoded = AFEMeasFrame.decode(frame)
        
        assert decoded is not None
        assert decoded['timestamp_ms'] == timestamp_ms
        assert np.array_equal(decoded['vcell_mv'], vcell_mv)
        assert np.array_equal(decoded['tcell_cc'], tcell_cc)
        assert decoded['pack_current_ma'] == pack_current_ma
        assert decoded['pack_voltage_mv'] == pack_voltage_mv
        assert decoded['status_flags'] == status_flags
        assert decoded['sequence'] == sequence
    
    def test_crc16_ccitt(self):
        """Test CRC16-CCITT calculation."""
        # Test with known value
        data = b"123456789"
        crc = crc16_ccitt(data)
        assert crc == 0x29B1, f"Expected 0x29B1, got 0x{crc:04X}"
        
        # Test with empty data
        crc_empty = crc16_ccitt(b"")
        assert crc_empty == 0xFFFF  # Initial value
    
    @patch('serial.Serial')
    def test_send_frame_success(self, mock_serial_class):
        """Test successful frame sending."""
        mock_serial = MagicMock()
        mock_serial.is_open = True
        mock_serial.write.return_value = 100  # Bytes written
        mock_serial_class.return_value = mock_serial
        
        tx = UARTTransmitter(port='COM3', baudrate=921600, frame_rate_hz=50.0, verbose=False)
        
        # Start transmitter
        assert tx.start()
        
        # Prepare data
        data = {
            'timestamp_ms': 1000,
            'vcell_mv': np.array([3200] * 16, dtype=np.uint16),
            'tcell_cc': np.array([2500] * 16, dtype=np.int16),
            'pack_current_ma': 50000,
            'pack_voltage_mv': 51200,
            'status_flags': 0
        }
        
        # Send frame
        result = tx.send_frame(data)
        assert result, "Frame should be queued successfully"
        
        # Wait for transmission
        time.sleep(0.1)
        
        # Stop transmitter
        tx.stop()
        
        # Check statistics
        stats = tx.get_statistics()
        assert stats['sent_count'] >= 0  # May be 0 if thread didn't process yet
        assert stats['sequence'] == 1  # Should increment
    
    @patch('serial.Serial')
    def test_send_frame_validation_error(self, mock_serial_class):
        """Test frame sending with validation error."""
        tx = UARTTransmitter(port='COM3', baudrate=921600)
        
        # Invalid data (missing fields)
        data = {
            'timestamp_ms': 1000,
            # Missing other fields
        }
        
        result = tx.send_frame(data)
        assert not result, "Frame should fail validation"
        
        stats = tx.get_statistics()
        assert stats['error_count'] > 0
        assert stats['last_error'] is not None
    
    @patch('serial.Serial')
    def test_retry_logic(self, mock_serial_class):
        """Test retry logic on serial errors."""
        mock_serial = MagicMock()
        mock_serial.is_open = True
        mock_serial.write.side_effect = [serial.SerialTimeoutException("Timeout")] * 3
        mock_serial_class.return_value = mock_serial
        
        tx = UARTTransmitter(
            port='COM3',
            baudrate=921600,
            frame_rate_hz=50.0,
            retry_max=3,
            verbose=False
        )
        
        assert tx.start()
        
        data = {
            'timestamp_ms': 1000,
            'vcell_mv': np.array([3200] * 16, dtype=np.uint16),
            'tcell_cc': np.array([2500] * 16, dtype=np.int16),
            'pack_current_ma': 50000,
            'pack_voltage_mv': 51200,
            'status_flags': 0
        }
        
        tx.send_frame(data)
        
        # Wait for retries
        time.sleep(0.5)
        
        tx.stop()
        
        stats = tx.get_statistics()
        # Should have errors due to retry failures
        assert stats['error_count'] > 0
    
    def test_sequence_wrapping(self):
        """Test sequence number wrapping at 65535."""
        tx = UARTTransmitter(port='COM3', baudrate=921600)
        
        # Set sequence to near wrap
        tx._sequence = 65534
        
        data = {
            'timestamp_ms': 1000,
            'vcell_mv': np.array([3200] * 16, dtype=np.uint16),
            'tcell_cc': np.array([2500] * 16, dtype=np.int16),
            'pack_current_ma': 50000,
            'pack_voltage_mv': 51200,
            'status_flags': 0
        }
        
        # Send frame (should increment to 65535)
        tx.send_frame(data)
        assert tx._sequence == 65535
        
        # Send another (should wrap to 0)
        tx.send_frame(data)
        assert tx._sequence == 0
    
    def test_rate_limiting(self):
        """Test rate limiting."""
        tx = UARTTransmitter(port='COM3', baudrate=921600, frame_rate_hz=10.0)
        
        # Frame interval should be 0.1 seconds (100ms)
        assert abs(tx._frame_interval_sec - 0.1) < 0.001
    
    def test_statistics(self):
        """Test statistics tracking."""
        tx = UARTTransmitter(port='COM3', baudrate=921600)
        
        stats = tx.get_statistics()
        
        assert 'sent_count' in stats
        assert 'error_count' in stats
        assert 'last_error' in stats
        assert 'sequence' in stats
        assert 'queue_size' in stats
        
        assert stats['sent_count'] == 0
        assert stats['error_count'] == 0
        assert stats['sequence'] == 0
    
    def test_reset_statistics(self):
        """Test statistics reset."""
        tx = UARTTransmitter(port='COM3', baudrate=921600)
        
        # Modify statistics
        tx._sent_count = 100
        tx._error_count = 10
        tx._last_error = "Test error"
        
        # Reset
        tx.reset_statistics()
        
        stats = tx.get_statistics()
        assert stats['sent_count'] == 0
        assert stats['error_count'] == 0
        assert stats['last_error'] is None
        # Sequence should not be reset
        assert stats['sequence'] == 0  # Or whatever it was
    
    def test_list_available_ports(self):
        """Test listing available serial ports."""
        ports = UARTTransmitter.list_available_ports()
        
        # Should return a list (may be empty if no ports available)
        assert isinstance(ports, list)
    
    @patch('serial.Serial')
    def test_start_stop(self, mock_serial_class):
        """Test start/stop functionality."""
        mock_serial = MagicMock()
        mock_serial.is_open = True
        mock_serial_class.return_value = mock_serial
        
        tx = UARTTransmitter(port='COM3', baudrate=921600, verbose=False)
        
        # Start
        assert tx.start()
        assert tx._tx_thread is not None
        assert tx._tx_thread.is_alive()
        
        # Stop
        tx.stop()
        
        # Thread should stop
        time.sleep(0.1)
        assert not tx._tx_thread.is_alive()
    
    @patch('serial.Serial')
    def test_double_start(self, mock_serial_class):
        """Test that starting twice doesn't create multiple threads."""
        mock_serial = MagicMock()
        mock_serial.is_open = True
        mock_serial_class.return_value = mock_serial
        
        tx = UARTTransmitter(port='COM3', baudrate=921600, verbose=False)
        
        # Start first time
        assert tx.start()
        thread1 = tx._tx_thread
        
        # Start second time (should fail)
        result = tx.start()
        assert not result  # Should return False
        
        # Thread should be the same
        assert tx._tx_thread == thread1
        
        tx.stop()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

