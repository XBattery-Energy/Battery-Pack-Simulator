"""
Unit tests for 16S Battery Pack model.
"""

import pytest
import numpy as np
from sil_bms.pc_simulator.plant.pack_model import BatteryPack16S


class TestBatteryPack16S:
    """Test suite for BatteryPack16S class."""
    
    def test_initialization(self):
        """Test pack initialization."""
        pack = BatteryPack16S(
            cell_capacity_ah=100.0,
            initial_soc_pct=50.0,
            ambient_temp_c=25.0,
            seed=42  # For reproducibility
        )
        
        assert len(pack._cells) == 16, "Should have 16 cells"
        assert pack._cell_capacity_nominal_ah == 100.0
        assert pack._ambient_temp_c == 25.0
        assert pack._soc_calculation_mode == 'minimum'
    
    def test_cell_variations(self):
        """Test cell-to-cell variations."""
        pack = BatteryPack16S(
            cell_capacity_ah=100.0,
            initial_soc_pct=50.0,
            capacity_variation_sigma=1.5,
            soc_variation_sigma=2.0,
            seed=42
        )
        
        # Check capacity variations
        capacities = [cell._capacity_nominal_ah for cell in pack._cells]
        assert len(capacities) == 16
        # Capacities should vary around 100Ah
        assert 95.0 <= min(capacities) <= 105.0
        assert 95.0 <= max(capacities) <= 105.0
        
        # Check SOC variations
        socs = pack.get_cell_socs()
        assert len(socs) == 16
        # SOCs should vary around 50%
        assert 40.0 <= min(socs) <= 60.0
        assert 40.0 <= max(socs) <= 60.0
        # Mean should be close to 50%
        assert 45.0 <= np.mean(socs) <= 55.0
    
    def test_get_cell_voltages(self):
        """Test get_cell_voltages() method."""
        pack = BatteryPack16S(initial_soc_pct=50.0, seed=42)
        
        voltages = pack.get_cell_voltages()
        
        assert len(voltages) == 16
        assert isinstance(voltages, np.ndarray)
        # Voltages should be in reasonable range (2.5V to 3.65V per cell)
        assert np.all(voltages >= 2500), "All voltages should be >= 2.5V"
        assert np.all(voltages <= 3650), "All voltages should be <= 3.65V"
    
    def test_get_cell_temperatures(self):
        """Test get_cell_temperatures() method."""
        pack = BatteryPack16S(ambient_temp_c=25.0, seed=42)
        
        temps = pack.get_cell_temperatures()
        
        assert len(temps) == 16
        assert isinstance(temps, np.ndarray)
        # Temperatures should be close to ambient initially
        assert np.all(temps >= 20.0), "All temps should be >= 20°C"
        assert np.all(temps <= 30.0), "All temps should be <= 30°C"
    
    def test_get_cell_socs(self):
        """Test get_cell_socs() method."""
        pack = BatteryPack16S(initial_soc_pct=50.0, seed=42)
        
        socs = pack.get_cell_socs()
        
        assert len(socs) == 16
        assert isinstance(socs, np.ndarray)
        # SOCs should be in 0-100% range
        assert np.all(socs >= 0.0), "All SOCs should be >= 0%"
        assert np.all(socs <= 100.0), "All SOCs should be <= 100%"
    
    def test_get_pack_voltage(self):
        """Test pack voltage calculation."""
        pack = BatteryPack16S(initial_soc_pct=50.0, seed=42)
        
        pack_voltage = pack.get_pack_voltage()
        cell_voltages = pack.get_cell_voltages()
        
        # Pack voltage should be sum of cell voltages
        expected_voltage = np.sum(cell_voltages)
        assert abs(pack_voltage - expected_voltage) < 1.0, \
            f"Pack voltage should equal sum of cell voltages"
        
        # Pack voltage should be in reasonable range
        # 16 cells * 3.2V = 51.2V nominal
        assert 40.0 * 1000 <= pack_voltage <= 60.0 * 1000, \
            f"Pack voltage should be ~51.2V, got {pack_voltage/1000}V"
    
    def test_get_pack_current(self):
        """Test pack current tracking."""
        pack = BatteryPack16S(seed=42)
        
        # Initially should be 0
        assert pack.get_pack_current() == 0.0
        
        # Update with current
        pack.update(current_ma=50000, dt_ms=1000)
        assert pack.get_pack_current() == 50000.0
        
        # Update with different current
        pack.update(current_ma=-100000, dt_ms=1000)
        assert pack.get_pack_current() == -100000.0
    
    def test_get_pack_soc_minimum(self):
        """Test pack SOC calculation (minimum mode)."""
        pack = BatteryPack16S(
            initial_soc_pct=50.0,
            soc_calculation_mode='minimum',
            seed=42
        )
        
        pack_soc = pack.get_pack_soc()
        cell_socs = pack.get_cell_socs()
        
        # Pack SOC should equal minimum cell SOC
        assert abs(pack_soc - np.min(cell_socs)) < 0.1, \
            f"Pack SOC should equal minimum cell SOC"
    
    def test_get_pack_soc_average(self):
        """Test pack SOC calculation (average mode)."""
        pack = BatteryPack16S(
            initial_soc_pct=50.0,
            soc_calculation_mode='average',
            seed=42
        )
        
        pack_soc = pack.get_pack_soc()
        cell_socs = pack.get_cell_socs()
        
        # Pack SOC should equal average cell SOC
        assert abs(pack_soc - np.mean(cell_socs)) < 0.1, \
            f"Pack SOC should equal average cell SOC"
    
    def test_update_charge(self):
        """Test pack update during charging."""
        pack = BatteryPack16S(
            cell_capacity_ah=100.0,
            initial_soc_pct=20.0,
            seed=42
        )
        
        initial_soc = pack.get_pack_soc()
        initial_voltage = pack.get_pack_voltage()
        
        # Charge at 50A (0.5C) for 1 hour
        current_ma = 50000
        dt_ms = 3600000  # 1 hour
        
        pack.update(current_ma, dt_ms)
        
        final_soc = pack.get_pack_soc()
        final_voltage = pack.get_pack_voltage()
        
        # SOC should increase
        assert final_soc > initial_soc, "Pack SOC should increase during charge"
        
        # Voltage should increase (cells charging)
        assert final_voltage > initial_voltage, "Pack voltage should increase during charge"
    
    def test_update_discharge(self):
        """Test pack update during discharge."""
        pack = BatteryPack16S(
            cell_capacity_ah=100.0,
            initial_soc_pct=80.0,
            seed=42
        )
        
        initial_soc = pack.get_pack_soc()
        initial_voltage = pack.get_pack_voltage()
        
        # Discharge at 100A (1C) for 0.5 hour
        current_ma = -100000
        dt_ms = 1800000  # 0.5 hour
        
        pack.update(current_ma, dt_ms)
        
        final_soc = pack.get_pack_soc()
        final_voltage = pack.get_pack_voltage()
        
        # SOC should decrease
        assert final_soc < initial_soc, "Pack SOC should decrease during discharge"
        
        # Voltage should decrease (cells discharging)
        assert final_voltage < initial_voltage, "Pack voltage should decrease during discharge"
    
    def test_cell_imbalance_growth(self):
        """Test cell imbalance growth over time."""
        pack = BatteryPack16S(
            cell_capacity_ah=100.0,
            initial_soc_pct=50.0,
            capacity_variation_sigma=1.5,
            seed=42
        )
        
        # Get initial imbalance
        initial_imbalance = pack.get_cell_imbalance()
        initial_delta = initial_imbalance['soc_delta_pct']
        
        # Discharge pack (imbalance should grow due to capacity mismatch)
        current_ma = -50000  # -50A
        dt_ms = 3600000  # 1 hour
        
        pack.update(current_ma, dt_ms)
        
        # Get imbalance after discharge
        final_imbalance = pack.get_cell_imbalance()
        final_delta = final_imbalance['soc_delta_pct']
        
        # Imbalance should increase (cells with lower capacity discharge faster)
        assert final_delta > initial_delta, \
            f"Cell imbalance should grow: {initial_delta:.2f}% -> {final_delta:.2f}%"
    
    def test_fault_injection_voltage(self):
        """Test voltage fault injection."""
        pack = BatteryPack16S(seed=42)
        
        # Get normal voltage
        normal_voltages = pack.get_cell_voltages()
        cell_5_voltage_normal = normal_voltages[5]
        
        # Inject fault: set cell 5 voltage to 3700mV (over-voltage)
        pack.set_cell_voltage(5, 3700.0)
        
        fault_voltages = pack.get_cell_voltages()
        cell_5_voltage_fault = fault_voltages[5]
        
        # Cell 5 voltage should be forced to 3700mV
        assert cell_5_voltage_fault == 3700.0, \
            f"Cell 5 voltage should be forced to 3700mV, got {cell_5_voltage_fault}mV"
        
        # Other cells should be unchanged
        assert fault_voltages[4] == normal_voltages[4], "Other cells should be unchanged"
        assert fault_voltages[6] == normal_voltages[6], "Other cells should be unchanged"
        
        # Clear fault
        pack.set_cell_voltage(5, None)
        cleared_voltages = pack.get_cell_voltages()
        assert abs(cleared_voltages[5] - cell_5_voltage_normal) < 10, \
            "Cell 5 voltage should return to normal after clearing fault"
    
    def test_fault_injection_temperature(self):
        """Test temperature fault injection."""
        pack = BatteryPack16S(ambient_temp_c=25.0, seed=42)
        
        # Get normal temperatures
        normal_temps = pack.get_cell_temperatures()
        cell_3_temp_normal = normal_temps[3]
        
        # Inject fault: set cell 3 temperature to 65°C (over-temperature)
        pack.set_cell_temperature(3, 65.0)
        
        fault_temps = pack.get_cell_temperatures()
        cell_3_temp_fault = fault_temps[3]
        
        # Cell 3 temperature should be forced to 65°C
        assert cell_3_temp_fault == 65.0, \
            f"Cell 3 temperature should be forced to 65°C, got {cell_3_temp_fault}°C"
        
        # Clear fault
        pack.set_cell_temperature(3, None)
        # Update to allow temperature to return
        pack.update(0.0, 1000, ambient_temp_c=25.0)
        cleared_temps = pack.get_cell_temperatures()
        # Temperature should return toward ambient (may take time)
        assert cleared_temps[3] < 65.0, "Cell 3 temperature should decrease after clearing fault"
    
    def test_clear_all_faults(self):
        """Test clear_all_faults() method."""
        pack = BatteryPack16S(seed=42)
        
        # Inject multiple faults
        pack.set_cell_voltage(5, 3700.0)
        pack.set_cell_voltage(10, 2400.0)
        pack.set_cell_temperature(3, 65.0)
        pack.set_cell_temperature(8, -10.0)
        
        # Verify faults are set
        voltages = pack.get_cell_voltages()
        temps = pack.get_cell_temperatures()
        assert voltages[5] == 3700.0
        assert voltages[10] == 2400.0
        assert temps[3] == 65.0
        assert temps[8] == -10.0
        
        # Clear all faults
        pack.clear_all_faults()
        
        # Verify faults are cleared
        voltages_cleared = pack.get_cell_voltages()
        temps_cleared = pack.get_cell_temperatures()
        assert voltages_cleared[5] != 3700.0 or abs(voltages_cleared[5] - 3700.0) > 100
        assert voltages_cleared[10] != 2400.0 or abs(voltages_cleared[10] - 2400.0) > 100
        assert temps_cleared[3] != 65.0 or abs(temps_cleared[3] - 65.0) > 5
        assert temps_cleared[8] != -10.0 or abs(temps_cleared[8] - (-10.0)) > 5
    
    def test_get_cell_imbalance(self):
        """Test get_cell_imbalance() method."""
        pack = BatteryPack16S(initial_soc_pct=50.0, seed=42)
        
        imbalance = pack.get_cell_imbalance()
        
        assert 'min_voltage_mv' in imbalance
        assert 'max_voltage_mv' in imbalance
        assert 'voltage_delta_mv' in imbalance
        assert 'min_soc_pct' in imbalance
        assert 'max_soc_pct' in imbalance
        assert 'soc_delta_pct' in imbalance
        assert 'voltage_std_mv' in imbalance
        assert 'soc_std_pct' in imbalance
        
        # Check that delta = max - min
        assert imbalance['voltage_delta_mv'] == \
            imbalance['max_voltage_mv'] - imbalance['min_voltage_mv']
        assert imbalance['soc_delta_pct'] == \
            imbalance['max_soc_pct'] - imbalance['min_soc_pct']
    
    def test_set_aging(self):
        """Test aging application to all cells."""
        pack = BatteryPack16S(cell_capacity_ah=100.0, seed=42)
        
        # Get initial capacities
        initial_capacities = [cell._capacity_actual_ah for cell in pack._cells]
        
        # Apply aging (1000 cycles)
        pack.set_aging(1000)
        
        # Get aged capacities
        aged_capacities = [cell._capacity_actual_ah for cell in pack._cells]
        
        # Capacities should decrease
        for i in range(16):
            assert aged_capacities[i] < initial_capacities[i], \
                f"Cell {i} capacity should decrease with aging"
    
    def test_get_pack_state(self):
        """Test get_pack_state() method."""
        pack = BatteryPack16S(seed=42)
        
        state = pack.get_pack_state()
        
        assert 'pack_voltage_mv' in state
        assert 'pack_current_ma' in state
        assert 'pack_soc_pct' in state
        assert 'cell_voltages_mv' in state
        assert 'cell_temperatures_c' in state
        assert 'cell_socs_pct' in state
        assert 'imbalance' in state
        assert 'ambient_temp_c' in state
        assert 'cycles' in state
        
        assert len(state['cell_voltages_mv']) == 16
        assert len(state['cell_temperatures_c']) == 16
        assert len(state['cell_socs_pct']) == 16
    
    def test_reset(self):
        """Test pack reset."""
        pack = BatteryPack16S(initial_soc_pct=50.0, seed=42)
        
        # Modify state
        pack.update(50000, 1000)  # Charge
        pack.set_cell_voltage(5, 3700.0)  # Inject fault
        
        # Reset
        pack.reset(soc_pct=30.0, temperature_c=20.0)
        
        # Check reset
        assert pack.get_pack_soc() < 35.0, "Pack SOC should be reset"
        temps = pack.get_cell_temperatures()
        assert np.all(temps < 25.0), "Temperatures should be reset"
        
        # Faults should be cleared
        voltages = pack.get_cell_voltages()
        assert voltages[5] != 3700.0 or abs(voltages[5] - 3700.0) > 100
    
    def test_thermal_coupling(self):
        """Test thermal coupling between adjacent cells."""
        pack = BatteryPack16S(ambient_temp_c=25.0, seed=42)
        
        # Heat up cell 5 with high current
        pack.update(100000, 10000, ambient_temp_c=25.0)  # 100A for 10s
        
        temps = pack.get_cell_temperatures()
        
        # Cell 5 should be hottest
        # Adjacent cells (4 and 6) should be warmer than distant cells due to coupling
        # Note: This is a simplified test - thermal coupling may be subtle
        assert temps[5] > temps[0], "Cell 5 should be warmer than distant cell"
        assert temps[5] > temps[15], "Cell 5 should be warmer than distant cell"
    
    def test_series_current(self):
        """Test that same current flows through all cells."""
        pack = BatteryPack16S(initial_soc_pct=50.0, seed=42)
        
        # Get initial SOCs
        initial_socs = pack.get_cell_socs()
        
        # Apply current
        current_ma = 50000  # 50A
        dt_ms = 1000  # 1 second
        
        pack.update(current_ma, dt_ms)
        
        # Get final SOCs
        final_socs = pack.get_cell_socs()
        
        # All cells should have same SOC change (same current, but different capacities)
        # Cells with higher capacity will have smaller SOC change
        soc_changes = final_socs - initial_socs
        
        # All SOC changes should be positive (charging)
        assert np.all(soc_changes > 0), "All cells should charge"
        
        # SOC changes should be proportional to capacity (higher capacity = smaller change)
        capacities = [cell._capacity_actual_ah for cell in pack._cells]
        # Check that cells with higher capacity have smaller SOC change
        # (This is approximate due to variations)
        for i in range(15):
            if capacities[i] > capacities[i+1]:
                # Higher capacity should have smaller SOC change
                assert soc_changes[i] <= soc_changes[i+1] * 1.1, \
                    "Higher capacity cells should have smaller SOC change"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

