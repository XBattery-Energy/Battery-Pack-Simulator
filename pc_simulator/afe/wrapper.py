"""
AFE Measurement Wrapper

This module simulates MC33774 AFE measurement characteristics:
- ADC quantization
- Measurement noise
- Calibration errors (gain/offset)
- Fault injection (open-wire, stuck ADC, NTC faults, etc.)
- Fault scheduling (time-based)
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Union
from enum import Enum
import time


class FaultType(Enum):
    """Fault types for injection."""
    OPEN_WIRE = "open_wire"
    STUCK_ADC = "stuck_adc"
    NTC_OPEN = "ntc_open"
    NTC_SHORT = "ntc_short"
    CURRENT_SENSOR_FAULT = "current_sensor_fault"
    CRC_ERROR = "crc_error"


class AFEWrapper:
    """
    AFE Measurement Wrapper
    
    Simulates MC33774 AFE measurement characteristics:
    - 16-bit ADC quantization (voltage, current)
    - 12-bit ADC quantization (temperature)
    - Gaussian noise injection
    - Per-channel calibration errors (gain/offset)
    - Fault injection and scheduling
    
    Parameters:
        noise_config: Dictionary with noise parameters (optional)
        calibration_errors: Dictionary with calibration error parameters (optional)
        seed: Random seed for reproducibility (optional)
    """
    
    # ADC resolutions
    VOLTAGE_RESOLUTION_MV = 0.1  # 16-bit ADC, 0.1mV resolution
    TEMPERATURE_RESOLUTION_C = 0.1  # 12-bit ADC, 0.1°C resolution
    CURRENT_RESOLUTION_MA = 1.0  # 16-bit ADC, 1mA resolution
    
    # Default noise standard deviations
    DEFAULT_VOLTAGE_NOISE_MV = 2.0  # σ = 2mV
    DEFAULT_TEMP_NOISE_C = 0.5  # σ = 0.5°C
    DEFAULT_CURRENT_NOISE_MA = 50.0  # σ = 50mA
    
    # Default calibration error ranges
    DEFAULT_VOLTAGE_GAIN_ERROR = 0.001  # ±0.1%
    DEFAULT_VOLTAGE_OFFSET_MV = 5.0  # ±5mV
    DEFAULT_TEMP_OFFSET_C = 1.0  # ±1°C
    DEFAULT_CURRENT_GAIN_ERROR = 0.002  # ±0.2%
    DEFAULT_CURRENT_OFFSET_MA = 10.0  # ±10mA
    
    # Invalid values for fault injection
    INVALID_VOLTAGE_MV = 0.0  # Open wire = 0V
    INVALID_TEMP_C = -32768  # NTC fault = -32768 (int16 invalid)
    INVALID_CURRENT_MA = 0.0  # Current sensor fault = 0A
    
    NUM_CELLS = 16
    
    def __init__(
        self,
        noise_config: Optional[Dict] = None,
        calibration_errors: Optional[Dict] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize AFE wrapper.
        
        Args:
            noise_config: Dictionary with noise parameters:
                - voltage_noise_mv: Voltage noise std dev (default: 2.0mV)
                - temp_noise_c: Temperature noise std dev (default: 0.5°C)
                - current_noise_ma: Current noise std dev (default: 50.0mA)
            calibration_errors: Dictionary with calibration parameters:
                - voltage_gain_error: Voltage gain error range (default: 0.001)
                - voltage_offset_mv: Voltage offset range (default: 5.0mV)
                - temp_offset_c: Temperature offset range (default: 1.0°C)
                - current_gain_error: Current gain error range (default: 0.002)
                - current_offset_ma: Current offset range (default: 10.0mA)
            seed: Random seed for reproducibility
        """
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Noise configuration
        self._noise_config = noise_config or {}
        self._voltage_noise_mv = self._noise_config.get('voltage_noise_mv', self.DEFAULT_VOLTAGE_NOISE_MV)
        self._temp_noise_c = self._noise_config.get('temp_noise_c', self.DEFAULT_TEMP_NOISE_C)
        self._current_noise_ma = self._noise_config.get('current_noise_ma', self.DEFAULT_CURRENT_NOISE_MA)
        
        # Calibration errors (per-channel)
        self._calibration_errors = calibration_errors or {}
        voltage_gain_error = self._calibration_errors.get('voltage_gain_error', self.DEFAULT_VOLTAGE_GAIN_ERROR)
        voltage_offset_mv = self._calibration_errors.get('voltage_offset_mv', self.DEFAULT_VOLTAGE_OFFSET_MV)
        temp_offset_c = self._calibration_errors.get('temp_offset_c', self.DEFAULT_TEMP_OFFSET_C)
        current_gain_error = self._calibration_errors.get('current_gain_error', self.DEFAULT_CURRENT_GAIN_ERROR)
        current_offset_ma = self._calibration_errors.get('current_offset_ma', self.DEFAULT_CURRENT_OFFSET_MA)
        
        # Generate per-channel calibration errors
        # Voltage: gain and offset per cell
        self._voltage_gain_errors = np.random.uniform(
            1.0 - voltage_gain_error,
            1.0 + voltage_gain_error,
            self.NUM_CELLS
        )
        self._voltage_offsets_mv = np.random.uniform(
            -voltage_offset_mv,
            voltage_offset_mv,
            self.NUM_CELLS
        )
        
        # Temperature: offset per channel
        self._temp_offsets_c = np.random.uniform(
            -temp_offset_c,
            temp_offset_c,
            self.NUM_CELLS
        )
        
        # Current: gain and offset (single channel)
        self._current_gain_error = np.random.uniform(
            1.0 - current_gain_error,
            1.0 + current_gain_error
        )
        self._current_offset_ma = np.random.uniform(
            -current_offset_ma,
            current_offset_ma
        )
        
        # Fault injection state
        self._open_wire_mask = 0  # uint16 bitmask
        self._stuck_adc_mask = 0  # uint16 bitmask
        self._stuck_adc_values = np.zeros(self.NUM_CELLS)  # Last valid values
        self._ntc_fault_mask = 0  # uint16 bitmask
        self._current_sensor_fault = False
        self._crc_error_rate = 0.0  # Probability of CRC error
        
        # Fault scheduling
        self._fault_schedule: List[Dict] = []  # List of scheduled faults
        self._start_time_ms = None  # Simulation start time
        
        # Status flags
        self._status_flags = 0  # uint32
        
        # Statistics
        self._measurement_count = 0
        self._crc_error_count = 0
    
    def apply_measurement(
        self,
        true_voltages: np.ndarray,
        true_temps: np.ndarray,
        true_current: float
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Apply AFE measurement processing to true values.
        
        Args:
            true_voltages: True cell voltages in mV (array[16])
            true_temps: True cell temperatures in °C (array[16])
            true_current: True pack current in mA
        
        Returns:
            Tuple of (measured_voltages, measured_temps, measured_current, status_flags)
            - measured_voltages: array[16] in mV (float)
            - measured_temps: array[16] in centi-°C (int16, divide by 100 to get °C)
            - measured_current: float in mA
            - status_flags: uint32 (bit flags)
        """
        self._measurement_count += 1
        
        # Update fault schedule
        self._update_fault_schedule()
        
        # Process voltages
        measured_voltages = self._process_voltages(true_voltages)
        
        # Process temperatures
        measured_temps = self._process_temperatures(true_temps)
        
        # Process current
        measured_current = self._process_current(true_current)
        
        # Update status flags
        self._update_status_flags(measured_voltages, measured_temps, measured_current)
        
        # Apply CRC error (if enabled)
        if self._should_inject_crc_error():
            self._crc_error_count += 1
            # CRC error doesn't modify data, but sets flag
            self._status_flags |= (1 << 31)  # Set CRC error bit
        
        return measured_voltages, measured_temps, measured_current, self._status_flags
    
    def _process_voltages(self, true_voltages: np.ndarray) -> np.ndarray:
        """Process cell voltages with quantization, noise, calibration, and faults."""
        measured_voltages = true_voltages.copy()
        
        # Apply calibration errors (gain and offset)
        measured_voltages = measured_voltages * self._voltage_gain_errors + self._voltage_offsets_mv
        
        # Apply Gaussian noise
        noise = np.random.normal(0.0, self._voltage_noise_mv, self.NUM_CELLS)
        measured_voltages += noise
        
        # Apply fault injection
        for i in range(self.NUM_CELLS):
            cell_mask = 1 << i
            
            # Open wire fault
            if self._open_wire_mask & cell_mask:
                measured_voltages[i] = self.INVALID_VOLTAGE_MV
            # Stuck ADC fault
            elif self._stuck_adc_mask & cell_mask:
                # Keep last valid value (don't update)
                if self._stuck_adc_values[i] == 0.0:
                    # First time, store current value
                    self._stuck_adc_values[i] = measured_voltages[i]
                else:
                    # Use stored value
                    measured_voltages[i] = self._stuck_adc_values[i]
            else:
                # Update stored value (for stuck ADC)
                self._stuck_adc_values[i] = measured_voltages[i]
        
        # Quantization (16-bit ADC, 0.1mV resolution)
        measured_voltages = np.round(measured_voltages / self.VOLTAGE_RESOLUTION_MV) * self.VOLTAGE_RESOLUTION_MV
        
        # Clip to valid range (0-6553.5mV for 16-bit, but typical cell range is 2500-3650mV)
        measured_voltages = np.clip(measured_voltages, 0.0, 6553.5)
        
        return measured_voltages
    
    def _process_temperatures(self, true_temps: np.ndarray) -> np.ndarray:
        """
        Process temperatures with quantization, noise, calibration, and faults.
        
        Returns:
            Array of temperatures in centi-°C (int16 format)
        """
        measured_temps = true_temps.copy()
        
        # Apply calibration errors (offset only)
        measured_temps += self._temp_offsets_c
        
        # Apply Gaussian noise
        noise = np.random.normal(0.0, self._temp_noise_c, self.NUM_CELLS)
        measured_temps += noise
        
        # Apply fault injection
        for i in range(self.NUM_CELLS):
            cell_mask = 1 << i
            
            # NTC fault (open or short)
            if self._ntc_fault_mask & cell_mask:
                measured_temps[i] = self.INVALID_TEMP_C / 100.0  # Convert to °C for processing
        
        # Quantization (12-bit ADC, 0.1°C resolution)
        measured_temps = np.round(measured_temps / self.TEMPERATURE_RESOLUTION_C) * self.TEMPERATURE_RESOLUTION_C
        
        # Convert to centi-°C (int16 format: -32768 to 32767, representing -327.68°C to 327.67°C)
        measured_temps_centi = (measured_temps * 100.0).astype(np.int16)
        
        return measured_temps_centi  # Return in centi-°C (int16)
    
    def _process_current(self, true_current: float) -> float:
        """Process current with quantization, noise, calibration, and faults."""
        measured_current = true_current
        
        # Apply calibration errors (gain and offset)
        measured_current = measured_current * self._current_gain_error + self._current_offset_ma
        
        # Apply Gaussian noise
        noise = np.random.normal(0.0, self._current_noise_ma)
        measured_current += noise
        
        # Apply fault injection
        if self._current_sensor_fault:
            measured_current = self.INVALID_CURRENT_MA
        
        # Quantization (16-bit ADC, 1mA resolution)
        measured_current = round(measured_current / self.CURRENT_RESOLUTION_MA) * self.CURRENT_RESOLUTION_MA
        
        return measured_current
    
    def _update_status_flags(self, voltages: np.ndarray, temps: np.ndarray, current: float):
        """Update status flags based on measurements."""
        self._status_flags = 0
        
        # Open wire flags (bits 0-15)
        self._status_flags |= self._open_wire_mask
        
        # Stuck ADC flags (bits 16-31, but we'll use bits 16-31 for other faults)
        # For now, use bits 16-31 for various faults
        
        # NTC fault flags (we'll use a separate field or bits 16-31)
        # For simplicity, combine into status flags
        
        # Check for invalid voltages (open wire detection)
        for i in range(self.NUM_CELLS):
            if voltages[i] == self.INVALID_VOLTAGE_MV:
                self._status_flags |= (1 << i)  # Set open wire bit
        
        # Check for invalid temperatures (NTC fault)
        # temps are in centi-°C, so -32768 (0x8000) is invalid
        for i in range(self.NUM_CELLS):
            if temps[i] <= -32000:  # Invalid temperature threshold (centi-°C)
                self._status_flags |= (1 << (16 + i))  # Set NTC fault bit
        
        # Current sensor fault (bit 30)
        if self._current_sensor_fault or current == self.INVALID_CURRENT_MA:
            self._status_flags |= (1 << 30)
        
        # CRC error (bit 31)
        # Set in apply_measurement if CRC error injected
    
    def _should_inject_crc_error(self) -> bool:
        """Check if CRC error should be injected (based on error rate)."""
        if self._crc_error_rate <= 0.0:
            return False
        return np.random.random() < self._crc_error_rate
    
    def inject_fault(
        self,
        fault_type: Union[FaultType, str],
        cell_mask: Optional[int] = None,
        duration_ms: Optional[float] = None
    ):
        """
        Inject fault.
        
        Args:
            fault_type: Fault type (FaultType enum or string)
            cell_mask: Cell bitmask (for cell-specific faults) or None for current sensor
            duration_ms: Duration in milliseconds (None = permanent)
        """
        if isinstance(fault_type, str):
            try:
                fault_type = FaultType(fault_type.lower())
            except ValueError:
                raise ValueError(f"Unknown fault type: {fault_type}")
        
        if fault_type == FaultType.OPEN_WIRE:
            if cell_mask is None:
                raise ValueError("cell_mask required for open_wire fault")
            self._open_wire_mask |= cell_mask
        
        elif fault_type == FaultType.STUCK_ADC:
            if cell_mask is None:
                raise ValueError("cell_mask required for stuck_adc fault")
            self._stuck_adc_mask |= cell_mask
        
        elif fault_type == FaultType.NTC_OPEN or fault_type == FaultType.NTC_SHORT:
            if cell_mask is None:
                raise ValueError("cell_mask required for NTC fault")
            self._ntc_fault_mask |= cell_mask
        
        elif fault_type == FaultType.CURRENT_SENSOR_FAULT:
            self._current_sensor_fault = True
        
        elif fault_type == FaultType.CRC_ERROR:
            # CRC error is controlled by error rate, not mask
            pass
        
        # Schedule fault clearing if duration specified
        if duration_ms is not None:
            current_time = self._get_current_time_ms()
            clear_time = current_time + duration_ms
            self._fault_schedule.append({
                'fault_type': fault_type,
                'cell_mask': cell_mask,
                'clear_time_ms': clear_time
            })
    
    def clear_fault(
        self,
        fault_type: Union[FaultType, str],
        cell_mask: Optional[int] = None
    ):
        """
        Clear fault.
        
        Args:
            fault_type: Fault type (FaultType enum or string)
            cell_mask: Cell bitmask (for cell-specific faults) or None for current sensor
        """
        if isinstance(fault_type, str):
            try:
                fault_type = FaultType(fault_type.lower())
            except ValueError:
                raise ValueError(f"Unknown fault type: {fault_type}")
        
        if fault_type == FaultType.OPEN_WIRE:
            if cell_mask is None:
                self._open_wire_mask = 0
            else:
                self._open_wire_mask &= ~cell_mask
        
        elif fault_type == FaultType.STUCK_ADC:
            if cell_mask is None:
                self._stuck_adc_mask = 0
            else:
                self._stuck_adc_mask &= ~cell_mask
        
        elif fault_type == FaultType.NTC_OPEN or fault_type == FaultType.NTC_SHORT:
            if cell_mask is None:
                self._ntc_fault_mask = 0
            else:
                self._ntc_fault_mask &= ~cell_mask
        
        elif fault_type == FaultType.CURRENT_SENSOR_FAULT:
            self._current_sensor_fault = False
    
    def schedule_fault(
        self,
        fault_type: Union[FaultType, str],
        inject_time_ms: float,
        cell_mask: Optional[int] = None,
        duration_ms: Optional[float] = None
    ):
        """
        Schedule fault injection at specific time.
        
        Args:
            fault_type: Fault type
            inject_time_ms: Time to inject fault (milliseconds from start)
            cell_mask: Cell bitmask (for cell-specific faults)
            duration_ms: Duration in milliseconds (None = permanent)
        """
        self._fault_schedule.append({
            'fault_type': fault_type,
            'cell_mask': cell_mask,
            'inject_time_ms': inject_time_ms,
            'duration_ms': duration_ms,
            'injected': False
        })
    
    def set_crc_error_rate(self, error_rate: float):
        """
        Set CRC error injection rate.
        
        Args:
            error_rate: Probability of CRC error per measurement (0.0 to 1.0)
        """
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError("CRC error rate must be between 0.0 and 1.0")
        self._crc_error_rate = error_rate
    
    def start_simulation(self):
        """Start simulation (initialize start time for fault scheduling)."""
        self._start_time_ms = time.time() * 1000.0
    
    def _get_current_time_ms(self) -> float:
        """Get current simulation time in milliseconds."""
        if self._start_time_ms is None:
            return 0.0
        return (time.time() * 1000.0) - self._start_time_ms
    
    def _update_fault_schedule(self):
        """Update fault schedule (inject/clear faults based on time)."""
        current_time = self._get_current_time_ms()
        
        # Process scheduled faults
        remaining_schedule = []
        for fault_event in self._fault_schedule:
            if 'inject_time_ms' in fault_event and not fault_event.get('injected', False):
                # Check if it's time to inject
                if current_time >= fault_event['inject_time_ms']:
                    self.inject_fault(
                        fault_event['fault_type'],
                        fault_event.get('cell_mask'),
                        fault_event.get('duration_ms')
                    )
                    fault_event['injected'] = True
                    # If no duration, remove from schedule
                    if fault_event.get('duration_ms') is None:
                        continue
            
            if 'clear_time_ms' in fault_event:
                # Check if it's time to clear
                if current_time >= fault_event['clear_time_ms']:
                    self.clear_fault(
                        fault_event['fault_type'],
                        fault_event.get('cell_mask')
                    )
                    continue  # Remove from schedule
            
            remaining_schedule.append(fault_event)
        
        self._fault_schedule = remaining_schedule
    
    def get_status_flags(self) -> int:
        """
        Get current status flags.
        
        Returns:
            uint32 status flags:
            - bits 0-15: Open wire flags (one per cell)
            - bits 16-31: Other faults (NTC, current sensor, CRC, etc.)
        """
        return self._status_flags
    
    def get_statistics(self) -> Dict:
        """
        Get measurement statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'measurement_count': self._measurement_count,
            'crc_error_count': self._crc_error_count,
            'crc_error_rate': self._crc_error_count / max(self._measurement_count, 1),
            'open_wire_mask': self._open_wire_mask,
            'stuck_adc_mask': self._stuck_adc_mask,
            'ntc_fault_mask': self._ntc_fault_mask,
            'current_sensor_fault': self._current_sensor_fault
        }
    
    def reset(self):
        """Reset wrapper state (clear all faults, reset statistics)."""
        self._open_wire_mask = 0
        self._stuck_adc_mask = 0
        self._stuck_adc_values.fill(0.0)
        self._ntc_fault_mask = 0
        self._current_sensor_fault = False
        self._crc_error_rate = 0.0
        self._fault_schedule = []
        self._status_flags = 0
        self._measurement_count = 0
        self._crc_error_count = 0
        self._start_time_ms = None

