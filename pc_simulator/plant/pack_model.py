"""
16S Battery Pack Model

This module implements a 16-cell series battery pack model using LiFePO₄ cells.
Includes cell-to-cell variations, thermal coupling, and fault injection capabilities.
"""

import numpy as np
from typing import Optional, Tuple
from plant.cell_model import LiFePO4Cell


class BatteryPack16S:
    """
    16S Battery Pack Model
    
    Features:
    - 16 cells in series connection
    - Cell-to-cell variations (capacity, SOC, resistance)
    - Thermal coupling between adjacent cells
    - Fault injection capabilities
    - Pack-level voltage and current calculation
    
    Parameters:
        cell_capacity_ah: Nominal capacity per cell in Ah (default: 100Ah)
        initial_soc_pct: Initial pack SOC in percent (default: 50%)
        ambient_temp_c: Ambient temperature in °C (default: 25.0)
        capacity_variation_sigma: Capacity mismatch standard deviation (default: 1.5%)
        soc_variation_sigma: Initial SOC variation standard deviation (default: 2%)
        resistance_variation: Internal resistance variation range (default: ±10%)
        thermal_coupling_coeff: Thermal coupling coefficient between cells (default: 0.1)
        soc_calculation_mode: 'average' or 'minimum' for pack SOC (default: 'minimum')
    """
    
    NUM_CELLS = 16
    
    def __init__(
        self,
        cell_capacity_ah: float = 100.0,
        initial_soc_pct: float = 50.0,
        ambient_temp_c: float = 25.0,
        capacity_variation_sigma: float = 0.4,  # Reduced from 1.5% to match real data (~6.8mV spread)
        soc_variation_sigma: float = 0.25,  # Reduced from 2.0% to match real data
        resistance_variation: float = 0.025,  # Reduced from 0.1 (10%) to 0.025 (2.5%) to match real data
        thermal_coupling_coeff: float = 0.1,
        soc_calculation_mode: str = 'minimum',
        seed: Optional[int] = None
    ):
        """
        Initialize 16S battery pack.
        
        Args:
            cell_capacity_ah: Nominal capacity per cell in Ah
            initial_soc_pct: Initial pack SOC in percent
            ambient_temp_c: Ambient temperature in °C
            capacity_variation_sigma: Capacity mismatch std dev in percent
            soc_variation_sigma: Initial SOC variation std dev in percent
            resistance_variation: Resistance variation range (±fraction)
            thermal_coupling_coeff: Thermal coupling coefficient (0-1)
            soc_calculation_mode: 'average' or 'minimum' for pack SOC
            seed: Random seed for reproducibility
        """
        self._cell_capacity_nominal_ah = cell_capacity_ah
        self._ambient_temp_c = ambient_temp_c
        self._thermal_coupling_coeff = thermal_coupling_coeff
        self._soc_calculation_mode = soc_calculation_mode
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Generate cell-to-cell variations
        # Reduced variations to match real data: real data shows ~6.8mV spread (vs 48.6mV in simulation)
        # Capacity mismatch: Gaussian distribution, σ = 0.4% (reduced from 1.5%)
        # Real well-balanced packs show <1% capacity variation
        capacity_variations = np.random.normal(1.0, capacity_variation_sigma / 100.0, self.NUM_CELLS)
        capacity_variations = np.clip(capacity_variations, 0.98, 1.02)  # Limit to ±2% (reduced from ±5%)
        
        # Initial SOC variation: Gaussian distribution, σ = 0.25% (reduced from 2%)
        # Real balanced packs have <0.5% SOC variation
        soc_variations = np.random.normal(0.0, soc_variation_sigma / 100.0, self.NUM_CELLS)
        initial_soc_per_cell = (initial_soc_pct / 100.0) + soc_variations
        initial_soc_per_cell = np.clip(initial_soc_per_cell, 0.0, 1.0)
        
        # Internal resistance variation: ±2.5% per cell (reduced from ±10%)
        # Real data shows very consistent cell behavior, suggesting low resistance variation
        resistance_variations = np.random.uniform(
            1.0 - resistance_variation,
            1.0 + resistance_variation,
            self.NUM_CELLS
        )
        
        # Create 16 cells with variations
        self._cells = []
        self._capacity_multipliers = capacity_variations
        self._resistance_multipliers = resistance_variations
        
        for i in range(self.NUM_CELLS):
            cell = LiFePO4Cell(
                capacity_ah=cell_capacity_ah * capacity_variations[i],
                initial_soc=initial_soc_per_cell[i],
                temperature_c=ambient_temp_c,
                resistance_multiplier=resistance_variations[i]
            )
            self._cells.append(cell)
        
        # Store original resistance multipliers for aging updates
        self._base_resistance_multipliers = resistance_variations.copy()
        
        # Fault injection state
        self._fault_voltages = np.full(self.NUM_CELLS, None, dtype=object)  # None or forced voltage in mV
        self._fault_temperatures = np.full(self.NUM_CELLS, None, dtype=object)  # None or forced temp in °C
        
        # Track pack current (same for all cells in series)
        self._pack_current_ma = 0.0
        
        # Statistics
        self._total_energy_ah = 0.0
        self._cycles = 0
    
    def update(
        self,
        current_ma: float,
        dt_ms: float,
        ambient_temp_c: Optional[float] = None
    ) -> None:
        """
        Update all cells in the pack.
        
        Args:
            current_ma: Pack current in mA (positive = charge, negative = discharge)
            dt_ms: Time step in milliseconds
            ambient_temp_c: Ambient temperature in °C (optional)
        """
        if ambient_temp_c is not None:
            self._ambient_temp_c = ambient_temp_c
        
        # Store pack current (same for all cells in series)
        self._pack_current_ma = current_ma
        
        # Get current cell temperatures (before update)
        current_temps = np.array([cell._temperature_c for cell in self._cells])
        
        # Update each cell
        for i, cell in enumerate(self._cells):
            # Check for fault injection
            forced_temp = self._fault_temperatures[i]
            temp_to_use = forced_temp if forced_temp is not None else None
            
            # Update cell
            voltage, soc = cell.update(
                current_ma=current_ma,
                dt_ms=dt_ms,
                temperature_c=temp_to_use,
                ambient_temp_c=self._ambient_temp_c
            )
            
            # Apply fault voltage if set
            if self._fault_voltages[i] is not None:
                # Override cell voltage (for fault injection)
                pass  # Voltage will be returned from get_cell_voltages()
        
        # Apply thermal coupling between adjacent cells
        self._apply_thermal_coupling(current_temps, dt_ms)
    
    def _apply_thermal_coupling(self, previous_temps: np.ndarray, dt_ms: float):
        """
        Apply thermal coupling between adjacent cells.
        
        Simplified model: heat flows between adjacent cells proportional to
        temperature difference.
        
        Args:
            previous_temps: Previous cell temperatures
            dt_ms: Time step in milliseconds
        """
        current_temps = np.array([cell._temperature_c for cell in self._cells])
        temp_diffs = current_temps - previous_temps
        
        # Thermal coupling: adjacent cells exchange heat
        # Heat flow: Q = k * (T_i - T_j) where k is coupling coefficient
        coupling_energy = np.zeros(self.NUM_CELLS)
        
        for i in range(self.NUM_CELLS):
            # Couple with left neighbor
            if i > 0:
                temp_diff = current_temps[i] - current_temps[i-1]
                coupling_energy[i] -= self._thermal_coupling_coeff * temp_diff
                coupling_energy[i-1] += self._thermal_coupling_coeff * temp_diff
            
            # Couple with right neighbor
            if i < self.NUM_CELLS - 1:
                temp_diff = current_temps[i] - current_temps[i+1]
                coupling_energy[i] -= self._thermal_coupling_coeff * temp_diff
                coupling_energy[i+1] += self._thermal_coupling_coeff * temp_diff
        
        # Apply thermal coupling (simplified: direct temperature adjustment)
        # Convert energy to temperature change: dT = Q / C_thermal
        # Using simplified model: adjust temperature directly
        thermal_mass = LiFePO4Cell.THERMAL_MASS
        dt_sec = dt_ms / 1000.0
        
        for i, cell in enumerate(self._cells):
            if self._fault_temperatures[i] is None:  # Don't override fault temperatures
                temp_change = (coupling_energy[i] * dt_sec) / thermal_mass
                cell._temperature_c += temp_change
                cell._temperature_c = np.clip(cell._temperature_c, -40.0, 85.0)
    
    def get_cell_voltages(self) -> np.ndarray:
        """
        Get cell voltages in mV.
        
        Returns:
            numpy array[16] of cell voltages in mV
        """
        voltages = np.zeros(self.NUM_CELLS)
        
        for i, cell in enumerate(self._cells):
            if self._fault_voltages[i] is not None:
                # Return fault-injected voltage
                voltages[i] = self._fault_voltages[i]
            else:
                # Get actual cell voltage
                state = cell.get_state()
                voltages[i] = state['voltage_mv']
        
        return voltages
    
    def get_cell_temperatures(self) -> np.ndarray:
        """
        Get cell temperatures in °C.
        
        Returns:
            numpy array[16] of cell temperatures in °C
        """
        temps = np.zeros(self.NUM_CELLS)
        
        for i, cell in enumerate(self._cells):
            if self._fault_temperatures[i] is not None:
                # Return fault-injected temperature
                temps[i] = self._fault_temperatures[i]
            else:
                temps[i] = cell._temperature_c
        
        return temps
    
    def get_cell_socs(self) -> np.ndarray:
        """
        Get cell SOCs in percent.
        
        Returns:
            numpy array[16] of cell SOCs in percent (0-100)
        """
        socs = np.array([cell._soc * 100.0 for cell in self._cells])
        return socs
    
    def get_pack_voltage(self) -> float:
        """
        Get total pack voltage in mV.
        
        Returns:
            Pack voltage in mV (sum of all cell voltages)
        """
        cell_voltages = self.get_cell_voltages()
        return np.sum(cell_voltages)
    
    def get_pack_current(self) -> float:
        """
        Get pack current in mA.
        
        Returns:
            Pack current in mA (same for all cells in series)
        """
        return self._pack_current_ma
    
    def get_pack_soc(self) -> float:
        """
        Get pack SOC in percent.
        
        Mode: 'average' or 'minimum' (configurable in __init__)
        
        Returns:
            Pack SOC in percent (0-100)
        """
        cell_socs = self.get_cell_socs()
        
        if self._soc_calculation_mode == 'minimum':
            return np.min(cell_socs)
        elif self._soc_calculation_mode == 'average':
            return np.mean(cell_socs)
        else:
            # Default to minimum
            return np.min(cell_socs)
    
    def set_cell_voltage(self, cell_index: int, voltage_mv: Optional[float]):
        """
        Inject fault: set cell voltage to a fixed value.
        
        Args:
            cell_index: Cell index (0-15)
            voltage_mv: Voltage in mV, or None to clear fault
        """
        if cell_index < 0 or cell_index >= self.NUM_CELLS:
            raise ValueError(f"Cell index must be 0-{self.NUM_CELLS-1}")
        
        self._fault_voltages[cell_index] = voltage_mv
    
    def set_cell_temperature(self, cell_index: int, temperature_c: Optional[float]):
        """
        Inject fault: set cell temperature to a fixed value.
        
        Args:
            cell_index: Cell index (0-15)
            temperature_c: Temperature in °C, or None to clear fault
        """
        if cell_index < 0 or cell_index >= self.NUM_CELLS:
            raise ValueError(f"Cell index must be 0-{self.NUM_CELLS-1}")
        
        self._fault_temperatures[cell_index] = temperature_c
    
    def clear_all_faults(self):
        """Clear all fault injections."""
        self._fault_voltages.fill(None)
        self._fault_temperatures.fill(None)
    
    def get_cell_imbalance(self) -> dict:
        """
        Get cell imbalance statistics.
        
        Returns:
            Dictionary with imbalance metrics:
            - min_voltage_mv: Minimum cell voltage
            - max_voltage_mv: Maximum cell voltage
            - voltage_delta_mv: Voltage difference (max - min)
            - min_soc_pct: Minimum cell SOC
            - max_soc_pct: Maximum cell SOC
            - soc_delta_pct: SOC difference (max - min)
        """
        voltages = self.get_cell_voltages()
        socs = self.get_cell_socs()
        
        return {
            'min_voltage_mv': np.min(voltages),
            'max_voltage_mv': np.max(voltages),
            'voltage_delta_mv': np.max(voltages) - np.min(voltages),
            'min_soc_pct': np.min(socs),
            'max_soc_pct': np.max(socs),
            'soc_delta_pct': np.max(socs) - np.min(socs),
            'voltage_std_mv': np.std(voltages),
            'soc_std_pct': np.std(socs)
        }
    
    def set_aging(self, cycles: int):
        """
        Apply aging to all cells.
        
        Args:
            cycles: Number of charge/discharge cycles
        """
        for cell in self._cells:
            cell.set_aging(cycles)
        self._cycles = cycles
    
    def get_pack_state(self) -> dict:
        """
        Get complete pack state.
        
        Returns:
            Dictionary with pack state information
        """
        cell_voltages = self.get_cell_voltages()
        cell_temps = self.get_cell_temperatures()
        cell_socs = self.get_cell_socs()
        imbalance = self.get_cell_imbalance()
        
        return {
            'pack_voltage_mv': self.get_pack_voltage(),
            'pack_current_ma': self.get_pack_current(),
            'pack_soc_pct': self.get_pack_soc(),
            'cell_voltages_mv': cell_voltages.tolist(),
            'cell_temperatures_c': cell_temps.tolist(),
            'cell_socs_pct': cell_socs.tolist(),
            'imbalance': imbalance,
            'ambient_temp_c': self._ambient_temp_c,
            'cycles': self._cycles
        }
    
    def reset(self, soc_pct: Optional[float] = None, temperature_c: Optional[float] = None):
        """
        Reset pack state (useful for testing).
        
        Args:
            soc_pct: New pack SOC in percent. If None, keep current.
            temperature_c: New temperature in °C. If None, keep current.
        """
        if soc_pct is not None:
            # Reset all cells to same SOC
            for cell in self._cells:
                cell.reset(soc_pct=soc_pct, temperature_c=temperature_c)
        else:
            for cell in self._cells:
                cell.reset(temperature_c=temperature_c)
        
        self.clear_all_faults()
        self._pack_current_ma = 0.0

