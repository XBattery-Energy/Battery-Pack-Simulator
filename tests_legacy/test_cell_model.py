"""
Unit tests for LiFePO₄ cell ECM model.
"""

import pytest
import numpy as np
from sil_bms.pc_simulator.plant.cell_model import LiFePO4Cell


class TestLiFePO4Cell:
    """Test suite for LiFePO4Cell class."""
    
    def test_initialization(self):
        """Test cell initialization."""
        cell = LiFePO4Cell(capacity_ah=100.0, initial_soc=0.5, temperature_c=25.0)
        
        assert cell._capacity_nominal_ah == 100.0
        assert cell._soc == 0.5
        assert cell._temperature_c == 25.0
        assert cell._cycles == 0
        assert cell._v_rc == 0.0
    
    def test_ocv_soc_curve(self):
        """Test OCV-SOC relationship."""
        cell = LiFePO4Cell()
        
        # Test at 0% SOC
        ocv_0 = cell.get_ocv(soc_pct=0.0)
        assert 2.45 <= ocv_0 <= 2.55, f"OCV at 0% should be ~2.5V, got {ocv_0}V"
        
        # Test at 50% SOC (flat plateau)
        ocv_50 = cell.get_ocv(soc_pct=50.0)
        assert 3.20 <= ocv_50 <= 3.30, f"OCV at 50% should be ~3.25V, got {ocv_50}V"
        
        # Test at 100% SOC
        ocv_100 = cell.get_ocv(soc_pct=100.0)
        assert 3.60 <= ocv_100 <= 3.70, f"OCV at 100% should be ~3.65V, got {ocv_100}V"
        
        # Test monotonicity (OCV should increase with SOC)
        ocv_10 = cell.get_ocv(soc_pct=10.0)
        ocv_20 = cell.get_ocv(soc_pct=20.0)
        ocv_80 = cell.get_ocv(soc_pct=80.0)
        ocv_90 = cell.get_ocv(soc_pct=90.0)
        
        assert ocv_10 < ocv_20, "OCV should increase with SOC"
        assert ocv_80 < ocv_90, "OCV should increase with SOC"
    
    def test_ocv_temperature_effect(self):
        """Test OCV temperature coefficient."""
        cell = LiFePO4Cell()
        
        ocv_25 = cell.get_ocv(soc_pct=50.0, temperature_c=25.0)
        ocv_0 = cell.get_ocv(soc_pct=50.0, temperature_c=0.0)
        ocv_50 = cell.get_ocv(soc_pct=50.0, temperature_c=50.0)
        
        # OCV should decrease with temperature (negative coefficient)
        # At 0°C: OCV = OCV_25 + (-0.5mV/°C) * (0 - 25) = OCV_25 + 12.5mV
        # At 50°C: OCV = OCV_25 + (-0.5mV/°C) * (50 - 25) = OCV_25 - 12.5mV
        assert ocv_0 > ocv_25, "OCV should be higher at lower temperature"
        assert ocv_50 < ocv_25, "OCV should be lower at higher temperature"
        
        # Check approximate magnitude (12.5mV difference for 25°C change)
        diff_0 = (ocv_0 - ocv_25) * 1000  # Convert to mV
        diff_50 = (ocv_25 - ocv_50) * 1000  # Convert to mV
        
        assert 10.0 <= diff_0 <= 15.0, f"OCV difference at 0°C should be ~12.5mV, got {diff_0}mV"
        assert 10.0 <= diff_50 <= 15.0, f"OCV difference at 50°C should be ~12.5mV, got {diff_50}mV"
    
    def test_internal_resistance_soc_dependence(self):
        """Test R0 as function of SOC."""
        cell = LiFePO4Cell(temperature_c=25.0)
        
        r0_0 = cell.get_internal_resistance(soc_pct=0.0)
        r0_50 = cell.get_internal_resistance(soc_pct=50.0)
        r0_100 = cell.get_internal_resistance(soc_pct=100.0)
        
        # R0 should be higher at extremes (0% and 100%) than at 50%
        assert r0_0 > r0_50, "R0 should be higher at 0% SOC"
        assert r0_100 > r0_50, "R0 should be higher at 100% SOC"
        
        # Check approximate values
        # At 50%: R0 = 0.5 mΩ
        assert 0.4 <= r0_50 <= 0.6, f"R0 at 50% should be ~0.5mΩ, got {r0_50}mΩ"
        
        # At 0%: R0 = 0.5 * 1.5 = 0.75 mΩ
        assert 0.6 <= r0_0 <= 0.9, f"R0 at 0% should be ~0.75mΩ, got {r0_0}mΩ"
        
        # At 100%: R0 = 0.5 * 2.0 = 1.0 mΩ
        assert 0.8 <= r0_100 <= 1.2, f"R0 at 100% should be ~1.0mΩ, got {r0_100}mΩ"
    
    def test_internal_resistance_temperature_dependence(self):
        """Test R0 as function of temperature."""
        cell = LiFePO4Cell()
        
        r0_25 = cell.get_internal_resistance(soc_pct=50.0, temperature_c=25.0)
        r0_0 = cell.get_internal_resistance(soc_pct=50.0, temperature_c=0.0)
        r0_50 = cell.get_internal_resistance(soc_pct=50.0, temperature_c=50.0)
        
        # R0 should decrease with temperature (negative coefficient)
        assert r0_0 > r0_25, "R0 should be higher at lower temperature"
        assert r0_50 < r0_25, "R0 should be lower at higher temperature"
        
        # Check approximate magnitude
        # At 0°C: R0 = 0.5 * [1 - 0.005 * (0 - 25)] = 0.5 * 1.125 = 0.5625 mΩ
        # At 50°C: R0 = 0.5 * [1 - 0.005 * (50 - 25)] = 0.5 * 0.875 = 0.4375 mΩ
        assert 0.5 <= r0_0 <= 0.65, f"R0 at 0°C should be ~0.56mΩ, got {r0_0}mΩ"
        assert 0.35 <= r0_50 <= 0.5, f"R0 at 50°C should be ~0.44mΩ, got {r0_50}mΩ"
    
    def test_aging_capacity_fade(self):
        """Test capacity fade with aging."""
        cell = LiFePO4Cell(capacity_ah=100.0, cycles=0)
        
        capacity_0 = cell._capacity_actual_ah
        assert capacity_0 == 100.0, "Initial capacity should be 100Ah"
        
        # Set to 1000 cycles
        cell.set_aging(1000)
        capacity_1000 = cell._capacity_actual_ah
        
        # Capacity fade: Q = 100 * (1 - 0.0001 * sqrt(1000))
        # Q = 100 * (1 - 0.0001 * 31.62) = 100 * 0.9968 = 99.68 Ah
        expected = 100.0 * (1.0 - 0.0001 * np.sqrt(1000))
        assert abs(capacity_1000 - expected) < 0.1, \
            f"Capacity after 1000 cycles should be ~{expected}Ah, got {capacity_1000}Ah"
        
        # Set to 10000 cycles
        cell.set_aging(10000)
        capacity_10000 = cell._capacity_actual_ah
        
        # Should have more fade
        assert capacity_10000 < capacity_1000, "Capacity should decrease with more cycles"
    
    def test_aging_resistance_increase(self):
        """Test resistance increase with aging."""
        cell = LiFePO4Cell(temperature_c=25.0, cycles=0)
        
        r0_0 = cell.get_internal_resistance(soc_pct=50.0)
        
        # Set to 1000 cycles
        cell.set_aging(1000)
        r0_1000 = cell.get_internal_resistance(soc_pct=50.0)
        
        # Resistance increase: R0 = 0.5 * (1 + 0.001 * 1000) = 0.5 * 2.0 = 1.0 mΩ
        expected = 0.5 * (1.0 + 0.001 * 1000)
        assert abs(r0_1000 - expected) < 0.1, \
            f"R0 after 1000 cycles should be ~{expected}mΩ, got {r0_1000}mΩ"
        
        # Should be higher than initial
        assert r0_1000 > r0_0, "Resistance should increase with cycles"
    
    def test_soc_update_charge(self):
        """Test SOC update during charging."""
        cell = LiFePO4Cell(capacity_ah=100.0, initial_soc=0.5, temperature_c=25.0)
        
        initial_soc = cell._soc
        
        # Charge at 50A (0.5C) for 1 hour = 50Ah
        # This should increase SOC by 50%
        current_ma = 50000  # 50A
        dt_ms = 3600000  # 1 hour
        
        voltage, soc_pct = cell.update(current_ma, dt_ms)
        
        # SOC should increase
        assert cell._soc > initial_soc, "SOC should increase during charge"
        
        # Approximate check: 50Ah / 100Ah = 0.5 = 50% increase
        # But we start at 50%, so should end near 100%
        assert cell._soc >= 0.95, f"SOC should be near 100% after 1h charge, got {cell._soc*100}%"
    
    def test_soc_update_discharge(self):
        """Test SOC update during discharge."""
        cell = LiFePO4Cell(capacity_ah=100.0, initial_soc=0.8, temperature_c=25.0)
        
        initial_soc = cell._soc
        
        # Discharge at 100A (1C) for 0.5 hour = 50Ah
        current_ma = -100000  # -100A (negative = discharge)
        dt_ms = 1800000  # 0.5 hour
        
        voltage, soc_pct = cell.update(current_ma, dt_ms)
        
        # SOC should decrease
        assert cell._soc < initial_soc, "SOC should decrease during discharge"
        
        # Approximate check: 50Ah / 100Ah = 0.5 = 50% decrease
        # Start at 80%, end near 30%
        assert cell._soc <= 0.35, f"SOC should be near 30% after 0.5h discharge, got {cell._soc*100}%"
    
    def test_rc_network_transient(self):
        """Test RC network transient response."""
        cell = LiFePO4Cell(initial_soc=0.5, temperature_c=25.0)
        
        # Apply step current and observe RC voltage
        current_ma = 50000  # 50A
        dt_ms = 100  # 100ms steps
        
        # Initial RC voltage should be 0
        assert cell._v_rc == 0.0, "Initial RC voltage should be 0"
        
        # After first step
        cell.update(current_ma, dt_ms)
        v_rc_1 = cell._v_rc
        
        # RC voltage should build up
        assert v_rc_1 > 0, "RC voltage should increase with current"
        
        # After many steps (should approach steady state)
        for _ in range(100):
            cell.update(current_ma, dt_ms)
        
        v_rc_steady = cell._v_rc
        
        # Steady state: V_RC = I * R1 = 50A * 5mΩ = 0.25V
        expected_steady = 50.0 * 5e-3  # 0.25V
        assert abs(v_rc_steady - expected_steady) < 0.05, \
            f"RC voltage should approach {expected_steady}V, got {v_rc_steady}V"
    
    def test_voltage_calculation(self):
        """Test terminal voltage calculation."""
        cell = LiFePO4Cell(initial_soc=0.5, temperature_c=25.0)
        
        # At rest (no current), voltage should equal OCV
        voltage_rest, _ = cell.update(0.0, 1000)  # 1 second, no current
        
        ocv = cell.get_ocv()
        assert abs(voltage_rest - ocv * 1000) < 10, \
            f"Voltage at rest should equal OCV (~{ocv*1000}mV), got {voltage_rest}mV"
        
        # During charge, voltage should be higher than OCV
        voltage_charge, _ = cell.update(50000, 100)  # 50A charge
        
        assert voltage_charge > ocv * 1000, \
            "Voltage during charge should be higher than OCV"
        
        # During discharge, voltage should be lower than OCV
        cell.reset(soc_pct=50.0)
        voltage_discharge, _ = cell.update(-50000, 100)  # 50A discharge
        
        assert voltage_discharge < ocv * 1000, \
            "Voltage during discharge should be lower than OCV"
    
    def test_temperature_update(self):
        """Test thermal model update."""
        cell = LiFePO4Cell(initial_soc=0.5, temperature_c=25.0, ambient_temp_c=25.0)
        
        initial_temp = cell._temperature_c
        
        # Apply high current (should cause self-heating)
        current_ma = 100000  # 100A
        dt_ms = 1000  # 1 second
        
        cell.update(current_ma, dt_ms)
        
        # Temperature should increase due to self-heating
        # (may be small, but should be positive)
        assert cell._temperature_c >= initial_temp, \
            "Temperature should increase with high current"
    
    def test_capacity_temperature_effect(self):
        """Test capacity temperature coefficient."""
        cell_25 = LiFePO4Cell(capacity_ah=100.0, initial_soc=0.5, temperature_c=25.0)
        cell_0 = LiFePO4Cell(capacity_ah=100.0, initial_soc=0.5, temperature_c=0.0)
        cell_50 = LiFePO4Cell(capacity_ah=100.0, initial_soc=0.5, temperature_c=50.0)
        
        # Discharge same amount at different temperatures
        current_ma = -50000  # -50A
        dt_ms = 3600000  # 1 hour
        
        voltage_25, soc_25 = cell_25.update(current_ma, dt_ms)
        voltage_0, soc_0 = cell_0.update(current_ma, dt_ms)
        voltage_50, soc_50 = cell_50.update(current_ma, dt_ms)
        
        # At higher temperature, capacity is higher, so SOC decreases less
        # (more capacity means same current removes less SOC)
        assert soc_50 > soc_25, "Higher temperature should preserve more SOC (higher capacity)"
        assert soc_25 > soc_0, "Lower temperature should lose more SOC (lower capacity)"
    
    def test_soc_limits(self):
        """Test SOC clamping at limits."""
        cell = LiFePO4Cell(initial_soc=0.0, temperature_c=25.0)
        
        # Try to discharge below 0%
        voltage, soc = cell.update(-100000, 10000)  # High discharge
        
        assert cell._soc >= 0.0, "SOC should not go below 0%"
        assert soc >= 0.0, "SOC percent should not go below 0%"
        
        # Reset and charge to 100%
        cell.reset(soc_pct=100.0)
        
        # Try to charge above 100%
        voltage, soc = cell.update(100000, 10000)  # High charge
        
        assert cell._soc <= 1.0, "SOC should not go above 100%"
        assert soc <= 100.0, "SOC percent should not go above 100%"
    
    def test_get_state(self):
        """Test get_state() method."""
        cell = LiFePO4Cell(capacity_ah=100.0, initial_soc=0.6, temperature_c=30.0, cycles=500)
        
        state = cell.get_state()
        
        assert 'soc_pct' in state
        assert 'voltage_mv' in state
        assert 'temperature_c' in state
        assert 'capacity_ah' in state
        assert 'internal_resistance_mohm' in state
        assert 'cycles' in state
        assert 'rc_voltage_v' in state
        
        assert state['soc_pct'] == 60.0
        assert state['temperature_c'] == 30.0
        assert state['cycles'] == 500


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

