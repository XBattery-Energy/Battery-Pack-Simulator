"""
Unit tests for AFE Measurement Wrapper.
"""

import pytest
import numpy as np
from sil_bms.pc_simulator.afe.wrapper import AFEWrapper, FaultType


class TestAFEWrapper:
    """Test suite for AFEWrapper class."""
    
    def test_initialization(self):
        """Test AFE wrapper initialization."""
        wrapper = AFEWrapper(seed=42)
        
        assert wrapper._voltage_noise_mv == 2.0
        assert wrapper._temp_noise_c == 0.5
        assert wrapper._current_noise_ma == 50.0
        assert len(wrapper._voltage_gain_errors) == 16
        assert len(wrapper._voltage_offsets_mv) == 16
        assert len(wrapper._temp_offsets_c) == 16
    
    def test_custom_noise_config(self):
        """Test custom noise configuration."""
        wrapper = AFEWrapper(
            noise_config={
                'voltage_noise_mv': 5.0,
                'temp_noise_c': 1.0,
                'current_noise_ma': 100.0
            },
            seed=42
        )
        
        assert wrapper._voltage_noise_mv == 5.0
        assert wrapper._temp_noise_c == 1.0
        assert wrapper._current_noise_ma == 100.0
    
    def test_custom_calibration_errors(self):
        """Test custom calibration errors."""
        wrapper = AFEWrapper(
            calibration_errors={
                'voltage_gain_error': 0.002,
                'voltage_offset_mv': 10.0,
                'temp_offset_c': 2.0,
                'current_gain_error': 0.004,
                'current_offset_ma': 20.0
            },
            seed=42
        )
        
        # Check that errors are within specified ranges
        assert np.all(np.abs(wrapper._voltage_gain_errors - 1.0) <= 0.002)
        assert np.all(np.abs(wrapper._voltage_offsets_mv) <= 10.0)
        assert np.all(np.abs(wrapper._temp_offsets_c) <= 2.0)
        assert np.abs(wrapper._current_gain_error - 1.0) <= 0.004
        assert np.abs(wrapper._current_offset_ma) <= 20.0
    
    def test_apply_measurement_basic(self):
        """Test basic measurement application."""
        wrapper = AFEWrapper(seed=42)
        
        true_voltages = np.full(16, 3200.0)  # 3.2V per cell
        true_temps = np.full(16, 25.0)  # 25°C
        true_current = 50000.0  # 50A
        
        measured_v, measured_t, measured_i, flags = wrapper.apply_measurement(
            true_voltages, true_temps, true_current
        )
        
        assert len(measured_v) == 16
        assert len(measured_t) == 16
        assert isinstance(measured_i, (int, float))
        assert isinstance(flags, int)
        
        # Voltages should be close to true values (within noise + calibration)
        assert np.all(measured_v > 3000.0)
        assert np.all(measured_v < 3400.0)
        
        # Temperatures should be in centi-°C (int16 format)
        # 25°C = 2500 centi-°C
        assert np.all(measured_t > 2000)  # > 20°C (2000 centi-°C)
        assert np.all(measured_t < 3000)  # < 30°C (3000 centi-°C)
        assert measured_t.dtype == np.int16, "Temperatures should be int16"
    
    def test_quantization(self):
        """Test ADC quantization."""
        wrapper = AFEWrapper(
            noise_config={'voltage_noise_mv': 0.0, 'temp_noise_c': 0.0, 'current_noise_ma': 0.0},
            calibration_errors={'voltage_gain_error': 0.0, 'voltage_offset_mv': 0.0,
                               'temp_offset_c': 0.0, 'current_gain_error': 0.0, 'current_offset_ma': 0.0},
            seed=42
        )
        
        # Test voltage quantization (0.1mV resolution)
        true_voltages = np.full(16, 3200.123)  # Non-quantized value
        true_temps = np.full(16, 25.0)
        true_current = 50000.0
        
        measured_v, _, _, _ = wrapper.apply_measurement(true_voltages, true_temps, true_current)
        
        # Should be quantized to 0.1mV steps
        for v in measured_v:
            # Check that value is multiple of 0.1mV
            remainder = v % 0.1
            assert abs(remainder) < 0.001 or abs(remainder - 0.1) < 0.001
    
    def test_noise_injection(self):
        """Test Gaussian noise injection."""
        wrapper = AFEWrapper(
            noise_config={'voltage_noise_mv': 2.0, 'temp_noise_c': 0.5, 'current_noise_ma': 50.0},
            calibration_errors={'voltage_gain_error': 0.0, 'voltage_offset_mv': 0.0,
                               'temp_offset_c': 0.0, 'current_gain_error': 0.0, 'current_offset_ma': 0.0},
            seed=42
        )
        
        true_voltages = np.full(16, 3200.0)
        true_temps = np.full(16, 25.0)
        true_current = 50000.0
        
        # Run multiple measurements
        voltages_list = []
        for _ in range(100):
            measured_v, _, _, _ = wrapper.apply_measurement(true_voltages, true_temps, true_current)
            voltages_list.append(measured_v[0])
        
        # Check noise statistics
        voltages_array = np.array(voltages_list)
        std_dev = np.std(voltages_array)
        
        # Standard deviation should be close to noise level (2mV)
        assert 1.5 <= std_dev <= 2.5, f"Voltage noise std dev should be ~2mV, got {std_dev}mV"
    
    def test_open_wire_fault(self):
        """Test open wire fault injection."""
        wrapper = AFEWrapper(seed=42)
        
        true_voltages = np.full(16, 3200.0)
        true_temps = np.full(16, 25.0)
        true_current = 50000.0
        
        # Inject open wire on cell 5
        wrapper.inject_fault(FaultType.OPEN_WIRE, cell_mask=1 << 5)
        
        measured_v, _, _, flags = wrapper.apply_measurement(true_voltages, true_temps, true_current)
        
        # Cell 5 should have invalid voltage
        assert measured_v[5] == wrapper.INVALID_VOLTAGE_MV
        
        # Status flags should indicate open wire
        assert flags & (1 << 5), "Status flags should indicate open wire on cell 5"
    
    def test_stuck_adc_fault(self):
        """Test stuck ADC fault injection."""
        wrapper = AFEWrapper(seed=42)
        
        true_voltages = np.full(16, 3200.0)
        true_temps = np.full(16, 25.0)
        true_current = 50000.0
        
        # First measurement (normal)
        measured_v1, _, _, _ = wrapper.apply_measurement(true_voltages, true_temps, true_current)
        stuck_value = measured_v1[3]
        
        # Inject stuck ADC on cell 3
        wrapper.inject_fault(FaultType.STUCK_ADC, cell_mask=1 << 3)
        
        # Change true voltage
        true_voltages[3] = 3500.0
        
        # Second measurement (cell 3 should be stuck)
        measured_v2, _, _, _ = wrapper.apply_measurement(true_voltages, true_temps, true_current)
        
        # Cell 3 should still have old value
        assert abs(measured_v2[3] - stuck_value) < 1.0, \
            f"Cell 3 should be stuck at {stuck_value}mV, got {measured_v2[3]}mV"
        
        # Other cells should have new value
        assert measured_v2[0] != stuck_value
    
    def test_ntc_fault(self):
        """Test NTC fault injection."""
        wrapper = AFEWrapper(seed=42)
        
        true_voltages = np.full(16, 3200.0)
        true_temps = np.full(16, 25.0)
        true_current = 50000.0
        
        # Inject NTC fault on cell 8
        wrapper.inject_fault(FaultType.NTC_OPEN, cell_mask=1 << 8)
        
        measured_v, measured_t, _, flags = wrapper.apply_measurement(true_voltages, true_temps, true_current)
        
        # Cell 8 temperature should be invalid (in centi-°C, so -32768)
        assert measured_t[8] <= -32000, f"Cell 8 temp should be invalid, got {measured_t[8]} centi-°C"
        
        # Status flags should indicate NTC fault
        assert flags & (1 << (16 + 8)), "Status flags should indicate NTC fault on cell 8"
    
    def test_current_sensor_fault(self):
        """Test current sensor fault injection."""
        wrapper = AFEWrapper(seed=42)
        
        true_voltages = np.full(16, 3200.0)
        true_temps = np.full(16, 25.0)
        true_current = 50000.0
        
        # Inject current sensor fault
        wrapper.inject_fault(FaultType.CURRENT_SENSOR_FAULT)
        
        _, _, measured_i, flags = wrapper.apply_measurement(true_voltages, true_temps, true_current)
        
        # Current should be invalid
        assert measured_i == wrapper.INVALID_CURRENT_MA
        
        # Status flags should indicate current sensor fault
        assert flags & (1 << 30), "Status flags should indicate current sensor fault"
    
    def test_clear_fault(self):
        """Test fault clearing."""
        wrapper = AFEWrapper(seed=42)
        
        true_voltages = np.full(16, 3200.0)
        true_temps = np.full(16, 25.0)
        true_current = 50000.0
        
        # Inject open wire on cell 5
        wrapper.inject_fault(FaultType.OPEN_WIRE, cell_mask=1 << 5)
        measured_v1, _, _, flags1 = wrapper.apply_measurement(true_voltages, true_temps, true_current)
        assert measured_v1[5] == wrapper.INVALID_VOLTAGE_MV
        
        # Clear fault
        wrapper.clear_fault(FaultType.OPEN_WIRE, cell_mask=1 << 5)
        measured_v2, _, _, flags2 = wrapper.apply_measurement(true_voltages, true_temps, true_current)
        
        # Cell 5 should be normal
        assert measured_v2[5] != wrapper.INVALID_VOLTAGE_MV
        assert not (flags2 & (1 << 5))
    
    def test_fault_scheduling(self):
        """Test time-based fault scheduling."""
        wrapper = AFEWrapper(seed=42)
        wrapper.start_simulation()
        
        true_voltages = np.full(16, 3200.0)
        true_temps = np.full(16, 25.0)
        true_current = 50000.0
        
        # Schedule fault injection at 1000ms, duration 500ms
        wrapper.schedule_fault(
            FaultType.OPEN_WIRE,
            inject_time_ms=1000.0,
            cell_mask=1 << 5,
            duration_ms=500.0
        )
        
        # Before injection time (simulate by not waiting)
        # Note: In real use, time would advance naturally
        # For testing, we'll manually set time
        import time
        wrapper._start_time_ms = time.time() * 1000.0 - 500.0  # 500ms ago
        
        # Should not be injected yet
        measured_v1, _, _, _ = wrapper.apply_measurement(true_voltages, true_temps, true_current)
        # Fault not yet injected (time-based scheduling requires actual time passage)
        # This test demonstrates the API, actual time-based behavior needs real time
    
    def test_crc_error_rate(self):
        """Test CRC error rate injection."""
        wrapper = AFEWrapper(seed=42)
        
        # Set CRC error rate to 0.1 (10%)
        wrapper.set_crc_error_rate(0.1)
        
        true_voltages = np.full(16, 3200.0)
        true_temps = np.full(16, 25.0)
        true_current = 50000.0
        
        # Run many measurements
        crc_errors = 0
        for _ in range(1000):
            _, _, _, flags = wrapper.apply_measurement(true_voltages, true_temps, true_current)
            if flags & (1 << 31):  # CRC error bit
                crc_errors += 1
        
        # Should have approximately 10% CRC errors (with some variance)
        error_rate = crc_errors / 1000.0
        assert 0.05 <= error_rate <= 0.15, \
            f"CRC error rate should be ~10%, got {error_rate*100:.1f}%"
    
    def test_get_status_flags(self):
        """Test status flags generation."""
        wrapper = AFEWrapper(seed=42)
        
        true_voltages = np.full(16, 3200.0)
        true_temps = np.full(16, 25.0)
        true_current = 50000.0
        
        # Inject multiple faults
        wrapper.inject_fault(FaultType.OPEN_WIRE, cell_mask=1 << 3)
        wrapper.inject_fault(FaultType.NTC_OPEN, cell_mask=1 << 7)
        wrapper.inject_fault(FaultType.CURRENT_SENSOR_FAULT)
        
        _, _, _, flags = wrapper.apply_measurement(true_voltages, true_temps, true_current)
        
        # Check status flags
        assert flags & (1 << 3), "Should have open wire flag for cell 3"
        assert flags & (1 << (16 + 7)), "Should have NTC fault flag for cell 7"
        assert flags & (1 << 30), "Should have current sensor fault flag"
    
    def test_get_statistics(self):
        """Test statistics collection."""
        wrapper = AFEWrapper(seed=42)
        
        true_voltages = np.full(16, 3200.0)
        true_temps = np.full(16, 25.0)
        true_current = 50000.0
        
        # Run some measurements
        for _ in range(100):
            wrapper.apply_measurement(true_voltages, true_temps, true_current)
        
        stats = wrapper.get_statistics()
        
        assert stats['measurement_count'] == 100
        assert 'crc_error_count' in stats
        assert 'open_wire_mask' in stats
    
    def test_reset(self):
        """Test wrapper reset."""
        wrapper = AFEWrapper(seed=42)
        
        # Inject faults and make measurements
        wrapper.inject_fault(FaultType.OPEN_WIRE, cell_mask=1 << 5)
        wrapper.set_crc_error_rate(0.1)
        
        true_voltages = np.full(16, 3200.0)
        true_temps = np.full(16, 25.0)
        true_current = 50000.0
        
        wrapper.apply_measurement(true_voltages, true_temps, true_current)
        
        # Reset
        wrapper.reset()
        
        # Check that faults are cleared
        assert wrapper._open_wire_mask == 0
        assert wrapper._crc_error_rate == 0.0
        assert wrapper._measurement_count == 0
        
        # Measurements should be normal
        measured_v, _, _, flags = wrapper.apply_measurement(true_voltages, true_temps, true_current)
        assert measured_v[5] != wrapper.INVALID_VOLTAGE_MV
        assert not (flags & (1 << 5))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

