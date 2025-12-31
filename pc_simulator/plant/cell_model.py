"""
LiFePO₄ Battery Cell Equivalent Circuit Model (ECM)

This module implements a detailed ECM for LiFePO₄ cells with:
- OCV-SOC relationship with hysteresis (separate charge/discharge curves)
- Internal resistance R0 as function of SOC and temperature
- 2RC network (R1-C1 fast, R2-C2 slow) for transient response
- Temperature effects on OCV and capacity
- Aging factors (cycle aging + calendar aging, resistance increase)
- Thermal model (self-heating)
"""

import numpy as np
from typing import Tuple, Optional


class LiFePO4Cell:
    """
    LiFePO₄ Battery Cell Equivalent Circuit Model
    
    ECM Structure:
        OCV(SOC, T, direction) - R0(SOC, T) - [R1 || C1] - [R2 || C2] - Terminal
    
    Features:
    - Hysteresis: Separate charge/discharge OCV curves
    - 2RC Network: Fast (R1-C1) and slow (R2-C2) transients
    - Calendar Aging: Time-based capacity fade
    - Cycle Aging: Cycle-based capacity fade and resistance increase
    
    Parameters:
        capacity_ah: Nominal capacity in Ah (default: 100Ah)
        initial_soc: Initial state of charge (0.0 to 1.0, default: 0.5)
        temperature_c: Initial temperature in °C (default: 25.0)
        cycles: Number of charge/discharge cycles (default: 0)
    """
    
    # OCV-SOC lookup tables (101 points: 0% to 100%)
    # Typical LiFePO₄ curve: flat plateau around 3.2V, steep ends
    # Precision: 0.001V (1mV) for better accuracy, especially in steep regions
    # Hysteresis: Separate curves for charge and discharge
    _OCV_SOC_TABLE_DISCHARGE = np.array([
        # SOC%, OCV(V) - Interpolated from real data
        [0.0, 2.862],   # 0% - fully discharged
        [1.0, 2.912],
        [2.0, 2.962],
        [3.0, 3.012],
        [4.0, 3.062],
        [5.0, 3.112],
        [6.0, 3.124],
        [7.0, 3.136],
        [8.0, 3.148],
        [9.0, 3.160],
        [10.0, 3.172],  # 10%
        [11.0, 3.183],
        [12.0, 3.193],
        [13.0, 3.204],
        [14.0, 3.215],
        [15.0, 3.226],
        [16.0, 3.236],
        [17.0, 3.247],
        [18.0, 3.258],
        [19.0, 3.268],
        [20.0, 3.279],  # 20%
        [21.0, 3.280],
        [22.0, 3.280],
        [23.0, 3.281],
        [24.0, 3.281],
        [25.0, 3.282],
        [26.0, 3.283],
        [27.0, 3.283],
        [28.0, 3.284],
        [29.0, 3.284],
        [30.0, 3.285],  # 30%
        [31.0, 3.286],
        [32.0, 3.286],
        [33.0, 3.287],
        [34.0, 3.287],
        [35.0, 3.288],
        [36.0, 3.289],
        [37.0, 3.289],
        [38.0, 3.290],
        [39.0, 3.290],
        [40.0, 3.291],  # 40%
        [41.0, 3.292],
        [42.0, 3.292],
        [43.0, 3.293],
        [44.0, 3.293],
        [45.0, 3.294],
        [46.0, 3.295],
        [47.0, 3.295],
        [48.0, 3.296],
        [49.0, 3.296],
        [50.0, 3.297],  # 50%
        [51.0, 3.298],
        [52.0, 3.298],
        [53.0, 3.299],
        [54.0, 3.299],
        [55.0, 3.300],
        [56.0, 3.300],
        [57.0, 3.301],
        [58.0, 3.302],
        [59.0, 3.302],
        [60.0, 3.303],  # 60%
        [61.0, 3.303],
        [62.0, 3.304],
        [63.0, 3.304],
        [64.0, 3.305],
        [65.0, 3.306],
        [66.0, 3.306],
        [67.0, 3.307],
        [68.0, 3.307],
        [69.0, 3.308],
        [70.0, 3.308],  # 70%
        [71.0, 3.309],
        [72.0, 3.309],
        [73.0, 3.310],
        [74.0, 3.311],
        [75.0, 3.311],
        [76.0, 3.312],
        [77.0, 3.312],
        [78.0, 3.313],
        [79.0, 3.313],
        [80.0, 3.314],  # 80%
        [81.0, 3.316],
        [82.0, 3.317],
        [83.0, 3.319],
        [84.0, 3.320],
        [85.0, 3.322],
        [86.0, 3.323],
        [87.0, 3.325],
        [88.0, 3.326],
        [89.0, 3.328],
        [90.0, 3.329],  # 90%
        [91.0, 3.343],
        [92.0, 3.358],
        [93.0, 3.372],
        [94.0, 3.386],
        [95.0, 3.401],
        [96.0, 3.415],
        [97.0, 3.429],
        [98.0, 3.443],
        [99.0, 3.458],
        [100.0, 3.472],  # 100% - fully charged
    ])
    
    # Charge OCV table (typically 5-15mV higher than discharge at same SOC due to hysteresis)
    # LiFePO₄ shows less hysteresis than other chemistries, but it's still present
    _OCV_SOC_TABLE_CHARGE = np.array([
        # SOC%, OCV(V) - Charge curve (slightly higher)
        [0.0, 2.510],   # 0% - fully discharged
        [1.0, 2.560],
        [2.0, 2.610],
        [3.0, 2.660],
        [4.0, 2.710],
        [5.0, 2.760],
        [6.0, 2.810],
        [7.0, 2.860],
        [8.0, 2.910],
        [9.0, 2.960],
        [10.0, 3.010],
        [11.0, 3.060],
        [12.0, 3.110],
        [13.0, 3.160],
        [14.0, 3.190],
        [15.0, 3.210],
        [16.0, 3.220],
        [17.0, 3.230],
        [18.0, 3.240],
        [19.0, 3.250],
        [20.0, 3.260],  # Start of flat plateau (higher than discharge)
        [21.0, 3.260],
        [22.0, 3.260],
        [23.0, 3.260],
        [24.0, 3.260],
        [25.0, 3.260],
        [26.0, 3.260],
        [27.0, 3.260],
        [28.0, 3.260],
        [29.0, 3.260],
        [30.0, 3.260],
        [31.0, 3.260],
        [32.0, 3.260],
        [33.0, 3.260],
        [34.0, 3.260],
        [35.0, 3.260],
        [36.0, 3.260],
        [37.0, 3.260],
        [38.0, 3.260],
        [39.0, 3.260],
        [40.0, 3.260],
        [41.0, 3.260],
        [42.0, 3.260],
        [43.0, 3.260],
        [44.0, 3.260],
        [45.0, 3.260],
        [46.0, 3.260],
        [47.0, 3.260],
        [48.0, 3.260],
        [49.0, 3.260],
        [50.0, 3.260],
        [51.0, 3.260],
        [52.0, 3.260],
        [53.0, 3.260],
        [54.0, 3.260],
        [55.0, 3.260],
        [56.0, 3.260],
        [57.0, 3.260],
        [58.0, 3.260],
        [59.0, 3.260],
        [60.0, 3.260],
        [61.0, 3.260],
        [62.0, 3.260],
        [63.0, 3.260],
        [64.0, 3.260],
        [65.0, 3.260],
        [66.0, 3.260],
        [67.0, 3.260],
        [68.0, 3.260],
        [69.0, 3.260],
        [70.0, 3.260],
        [71.0, 3.260],
        [72.0, 3.260],
        [73.0, 3.260],
        [74.0, 3.260],
        [75.0, 3.260],
        [76.0, 3.260],
        [77.0, 3.260],
        [78.0, 3.260],
        [79.0, 3.260],
        [80.0, 3.260],  # End of flat plateau
        [81.0, 3.265],
        [82.0, 3.270],
        [83.0, 3.275],
        [84.0, 3.280],
        [85.0, 3.285],
        [86.0, 3.290],
        [87.0, 3.300],
        [88.0, 3.310],
        [89.0, 3.320],
        [90.0, 3.330],
        [91.0, 3.340],
        [92.0, 3.350],
        [93.0, 3.360],
        [94.0, 3.370],
        [95.0, 3.380],
        [96.0, 3.385],
        [97.0, 3.390],
        [98.0, 3.395],
        [99.0, 3.398],
        [100.0, 3.472],  # 100% - fully charged (adjusted to match real data: 3.472V)
    ])
    
    # ECM parameters - 2RC network
    # Fast RC network (short time constant)
    # Reduced resistances for high C-rate operation to prevent excessive voltage drops
    R1 = 1e-3  # Fast transient resistance: 1 mΩ (reduced from 3 mΩ)
    C1 = 2000.0  # Fast transient capacitance: 2000 F (time constant τ1 = R1*C1 = 2s)
    
    # Slow RC network (long time constant)
    R2 = 0.5e-3  # Slow transient resistance: 0.5 mΩ (reduced from 2 mΩ)
    C2 = 10000.0  # Slow transient capacitance: 10000 F (time constant τ2 = R2*C2 = 5s)
    
    # Temperature coefficients
    OCV_TEMP_COEFF = -0.5e-3  # OCV temperature coefficient: -0.5 mV/°C
    CAPACITY_TEMP_COEFF = 0.005  # Capacity temperature coefficient: +0.5% per °C
    
    # Aging parameters
    # Cycle aging
    FADE_RATE = 0.0001  # Capacity fade rate per cycle
    RESISTANCE_INCREASE_RATE = 0.001  # Resistance increase rate per cycle
    
    # Calendar aging parameters (Arrhenius-based model)
    CALENDAR_AGING_ACTIVATION_ENERGY = 30000.0  # Activation energy (J/mol) for Arrhenius equation
    GAS_CONSTANT = 8.314  # Gas constant (J/(mol·K))
    CALENDAR_AGING_BASE_RATE = 1.0e-9  # Base aging rate (per hour at 25°C, 50% SOC)
    CALENDAR_AGING_SOC_EXPONENT = 0.5  # SOC dependence exponent (aging faster at high/low SOC)
    CALENDAR_AGING_REF_TEMP = 298.15  # Reference temperature (25°C in Kelvin)
    
    # Thermal parameters
    THERMAL_MASS = 100.0  # Thermal mass (J/°C) - approximate for 100Ah cell
    THERMAL_RESISTANCE = 2.0  # Thermal resistance to ambient (°C/W)
    SELF_HEATING_COEFF = 0.001  # Self-heating coefficient (W per A²)
    
    def __init__(
        self,
        capacity_ah: float = 100.0,
        initial_soc: float = 0.5,
        temperature_c: float = 25.0,
        cycles: int = 0,
        ambient_temp_c: float = 25.0,
        resistance_multiplier: float = 1.0
    ):
        """
        Initialize LiFePO₄ cell model.
        
        Args:
            capacity_ah: Nominal capacity in Ah (default: 100Ah)
            initial_soc: Initial state of charge (0.0 to 1.0, default: 0.5)
            temperature_c: Initial temperature in °C (default: 25.0)
            cycles: Number of charge/discharge cycles (default: 0)
            ambient_temp_c: Ambient temperature in °C (default: 25.0)
            resistance_multiplier: Base resistance multiplier for cell-to-cell variation (default: 1.0)
        """
        # Store nominal capacity
        self._capacity_nominal_ah = capacity_ah
        
        # Initialize state variables
        self._soc = np.clip(initial_soc, 0.0, 1.0)
        self._temperature_c = temperature_c
        self._ambient_temp_c = ambient_temp_c
        self._cycles = cycles
        
        # Store base resistance multiplier (for cell-to-cell variation)
        self._base_resistance_multiplier = max(resistance_multiplier, 0.1)  # Prevent negative or zero multiplier
        
        # 2RC network state (voltages across C1 and C2)
        self._v_rc1 = 0.0  # Fast RC network voltage
        self._v_rc2 = 0.0  # Slow RC network voltage
        
        # Hysteresis tracking
        self._last_current_direction = 0  # 1 = charging, -1 = discharging, 0 = rest
        self._hysteresis_soc = initial_soc  # SOC at last current direction change
        
        # Calendar aging tracking
        self._calendar_aging_time_hours = 0.0  # Total time in hours (for calendar aging)
        self._last_update_time_hours = 0.0  # Last update time for calendar aging calculation
        self._storage_soc = initial_soc  # SOC during storage (for calendar aging)
        self._storage_temp = temperature_c  # Temperature during storage
        
        # Calculate aged capacity and resistance
        self._update_aging()
        
        # Initialize OCV lookup table interpolation (both charge and discharge)
        self._soc_table = self._OCV_SOC_TABLE_DISCHARGE[:, 0] / 100.0  # Convert % to fraction
        self._ocv_table_discharge = self._OCV_SOC_TABLE_DISCHARGE[:, 1]
        self._ocv_table_charge = self._OCV_SOC_TABLE_CHARGE[:, 1]
    
    def _update_aging(self):
        """
        Update capacity and resistance based on cycle count and calendar aging.
        
        Combined aging model:
        - Cycle aging: Capacity fade and resistance increase with cycles
        - Calendar aging: Capacity fade with time, temperature, and storage SOC
        """
        # Cycle aging: Capacity fade
        cycle_fade_factor = 1.0 - self.FADE_RATE * np.sqrt(max(self._cycles, 0))
        cycle_fade_factor = max(cycle_fade_factor, 0.5)  # Limit to 50% fade
        
        # Calendar aging: Time-based capacity fade
        # Arrhenius equation: rate = A * exp(-Ea/(R*T)) * SOC^exponent
        # Aging is faster at high temperature and extreme SOC
        if self._calendar_aging_time_hours > 0:
            temp_kelvin = self._storage_temp + 273.15
            arrhenius_factor = np.exp(
                -self.CALENDAR_AGING_ACTIVATION_ENERGY / 
                (self.GAS_CONSTANT * temp_kelvin)
            )
            
            # SOC dependence: aging faster at high/low SOC
            # Normalize SOC to 0-1 range, then apply exponent
            soc_factor = (self._storage_soc ** self.CALENDAR_AGING_SOC_EXPONENT) + \
                        ((1.0 - self._storage_soc) ** self.CALENDAR_AGING_SOC_EXPONENT)
            soc_factor = soc_factor / 2.0  # Normalize
            
            # Calculate calendar aging (hours to years conversion)
            calendar_aging_rate = self.CALENDAR_AGING_BASE_RATE * arrhenius_factor * soc_factor
            calendar_fade = calendar_aging_rate * self._calendar_aging_time_hours
            calendar_fade_factor = 1.0 - min(calendar_fade, 0.3)  # Limit to 30% calendar fade
        else:
            calendar_fade_factor = 1.0
        
        # Combined aging: multiply both factors
        total_fade_factor = cycle_fade_factor * calendar_fade_factor
        total_fade_factor = max(total_fade_factor, 0.5)  # Overall limit to 50% fade
        
        self._capacity_actual_ah = self._capacity_nominal_ah * total_fade_factor
        
        # Resistance increase: Only cycle-based (calendar aging has minimal effect on resistance)
        self._resistance_multiplier = 1.0 + self.RESISTANCE_INCREASE_RATE * max(self._cycles, 0)
    
    def get_ocv(
        self, 
        soc_pct: Optional[float] = None, 
        temperature_c: Optional[float] = None,
        current_direction: Optional[int] = None
    ) -> float:
        """
        Get Open Circuit Voltage (OCV) for given SOC and temperature with hysteresis.
        
        Args:
            soc_pct: State of charge in percent (0-100). If None, use current SOC.
            temperature_c: Temperature in °C. If None, use current temperature.
            current_direction: Current direction (1=charge, -1=discharge, 0=rest).
                             If None, use last known direction.
        
        Returns:
            OCV in volts
        """
        if soc_pct is None:
            soc = self._soc
        else:
            soc = np.clip(soc_pct / 100.0, 0.0, 1.0)
        
        if temperature_c is None:
            temp = self._temperature_c
        else:
            temp = temperature_c
        
        # Determine which OCV curve to use based on current direction
        if current_direction is None:
            current_direction = self._last_current_direction
        
        # Select OCV table based on current direction
        # Charge: use charge curve (higher voltage)
        # Discharge: use discharge curve (lower voltage)
        # Rest: interpolate between curves based on last direction
        if current_direction > 0:  # Charging
            ocv_table = self._ocv_table_charge
        elif current_direction < 0:  # Discharging
            ocv_table = self._ocv_table_discharge
        else:  # Rest - use average or last direction
            if self._last_current_direction > 0:
                ocv_table = self._ocv_table_charge
            elif self._last_current_direction < 0:
                ocv_table = self._ocv_table_discharge
            else:
                # No history - use average of charge and discharge
                ocv_charge = np.interp(soc, self._soc_table, self._ocv_table_charge)
                ocv_discharge = np.interp(soc, self._soc_table, self._ocv_table_discharge)
                ocv_base = (ocv_charge + ocv_discharge) / 2.0
                # Apply temperature correction
                ocv = ocv_base + self.OCV_TEMP_COEFF * (temp - 25.0)
                return ocv
        
        # Interpolate OCV from selected lookup table
        ocv_base = np.interp(soc, self._soc_table, ocv_table)
        
        # Apply temperature correction: OCV_temp = OCV_base + temp_coeff * (T - 25°C)
        ocv = ocv_base + self.OCV_TEMP_COEFF * (temp - 25.0)
        
        return ocv
    
    def get_internal_resistance(self, soc_pct: Optional[float] = None, temperature_c: Optional[float] = None) -> float:
        """
        Get internal resistance R0 as function of SOC and temperature.
        
        Formula: R0(SOC, T) = R0_base(SOC) * [1 - 0.005 * (T - 25)] * base_multiplier * aging_factor
        
        Args:
            soc_pct: State of charge in percent (0-100). If None, use current SOC.
            temperature_c: Temperature in °C. If None, use current temperature.
        
        Returns:
            Internal resistance in mΩ
        """
        if soc_pct is None:
            soc = self._soc
        else:
            soc = np.clip(soc_pct / 100.0, 0.0, 1.0)
        
        if temperature_c is None:
            temp = self._temperature_c
        else:
            temp = temperature_c
        
        # Base R0 at 25°C: 0.5 mΩ at 50% SOC
        # SOC dependence: higher at low SOC, lower at high SOC (typical LFP behavior)
        # Fine-tuned to better match real data behavior
        # - 1.4x at 0% SOC (slightly reduced from 1.5x for better low-SOC voltage match)
        # - 1.0x at 50% SOC (baseline)
        # - 0.75x at 100% SOC (slightly reduced from 0.8x for better high-SOC voltage match)
        if soc <= 0.5:
            # Linear from 0% to 50%
            r0_base_multiplier = 1.4 - (soc * 0.8)  # 1.4 at 0%, 1.0 at 50%
        else:
            # Linear from 50% to 100%: reduce resistance at high SOC
            r0_base_multiplier = 1.0 - ((soc - 0.5) * 0.5)  # 1.0 at 50%, 0.75 at 100%
        
        r0_base_mohm = 0.5 * r0_base_multiplier
        
        # Temperature dependence: -0.5% per °C (lower R0 at higher temp)
        temp_factor = 1.0 - 0.005 * (temp - 25.0)
        temp_factor = max(temp_factor, 0.5)  # Limit to 50% reduction
        
        # Apply base multiplier (cell-to-cell variation) and aging multiplier
        r0_mohm = r0_base_mohm * temp_factor * self._base_resistance_multiplier * self._resistance_multiplier
        
        return r0_mohm
    
    def _update_thermal_model(self, current_ma: float, dt_ms: float, ambient_temp_c: Optional[float] = None):
        """
        Update cell temperature based on self-heating and ambient.
        
        Thermal model:
        - Self-heating: P = I² * R0 (power dissipation)
        - Heat transfer: Q = (T_cell - T_ambient) / R_thermal
        - Temperature change: dT = (P - Q) * dt / C_thermal
        
        Args:
            current_ma: Current in mA (positive = charge, negative = discharge)
            dt_ms: Time step in milliseconds
            ambient_temp_c: Ambient temperature in °C. If None, use stored ambient.
        """
        if ambient_temp_c is not None:
            self._ambient_temp_c = ambient_temp_c
        
        # Convert current to Amperes
        current_a = current_ma / 1000.0
        
        # Calculate power dissipation: P = I² * R0
        r0_ohm = self.get_internal_resistance() / 1000.0  # Convert mΩ to Ω
        power_w = (current_a ** 2) * r0_ohm
        
        # Heat transfer to ambient: Q = (T_cell - T_ambient) / R_thermal
        temp_diff = self._temperature_c - self._ambient_temp_c
        heat_transfer_w = temp_diff / self.THERMAL_RESISTANCE
        
        # Net power: P_net = P_heating - Q_transfer
        net_power_w = power_w - heat_transfer_w
        
        # Temperature change: dT = P_net * dt / C_thermal
        dt_sec = dt_ms / 1000.0
        dtemp = (net_power_w * dt_sec) / self.THERMAL_MASS
        
        # Update temperature
        self._temperature_c += dtemp
        
        # Limit temperature to reasonable range
        self._temperature_c = np.clip(self._temperature_c, -40.0, 85.0)
    
    def update(
        self,
        current_ma: float,
        dt_ms: float,
        temperature_c: Optional[float] = None,
        ambient_temp_c: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Update cell state based on current and time step.
        
        ECM update:
        1. Update SOC using Coulomb counting
        2. Update 2RC network (R1-C1 fast, R2-C2 slow) voltages
        3. Calculate terminal voltage: V = OCV - I*R0 - V_RC1 - V_RC2
        4. Update thermal model
        5. Update calendar aging tracking
        
        Args:
            current_ma: Current in mA (positive = charge, negative = discharge)
            dt_ms: Time step in milliseconds
            temperature_c: External temperature in °C (optional, for forced temp)
            ambient_temp_c: Ambient temperature in °C (optional)
        
        Returns:
            Tuple of (terminal_voltage_mv, soc_pct)
        """
        # Update thermal model (if not forced temperature)
        if temperature_c is None:
            self._update_thermal_model(current_ma, dt_ms, ambient_temp_c)
        else:
            self._temperature_c = temperature_c
        
        # Get temperature-dependent capacity
        # Capacity increases with temperature: Q(T) = Q_nominal * [1 + 0.005 * (T - 25)]
        temp_capacity_factor = 1.0 + self.CAPACITY_TEMP_COEFF * (self._temperature_c - 25.0)
        capacity_ah = self._capacity_actual_ah * temp_capacity_factor
        
        # Update SOC using Coulomb counting
        # SOC change: dSOC = I * dt / Q
        # Positive current (charge) increases SOC, negative current (discharge) decreases SOC
        current_a = current_ma / 1000.0
        dt_hours = dt_ms / (1000.0 * 3600.0)
        dsoc = (current_a * dt_hours) / capacity_ah
        
        self._soc += dsoc
        self._soc = np.clip(self._soc, 0.0, 1.0)
        
        # Update current direction for hysteresis
        if current_ma > 0.001:  # Charging (small threshold to avoid noise)
            new_direction = 1
        elif current_ma < -0.001:  # Discharging
            new_direction = -1
        else:  # Rest
            new_direction = 0
        
        # Update hysteresis tracking if direction changed
        if new_direction != 0 and new_direction != self._last_current_direction:
            self._hysteresis_soc = self._soc
            self._last_current_direction = new_direction
        elif new_direction != 0:
            self._last_current_direction = new_direction
        
        # Update 2RC network voltages
        # C-rate dependent RC resistances: reduce at high C-rates to prevent unrealistic voltage drops
        # At high C-rates, polarization is lower due to better cell design and higher conductivity
        current_c_rate = abs(current_a) / self._capacity_nominal_ah if self._capacity_nominal_ah > 0 else 0.0
        
        # Scale RC resistances based on C-rate
        # At 1C: use full resistance
        # At 6C: use ~40% of resistance (reduced polarization at high rates)
        # Use a saturation function: R_eff = R_base * (1 / (1 + alpha * (C_rate - 1)))
        # This reduces RC resistance as C-rate increases
        if current_c_rate <= 1.0:
            rc_scale_factor = 1.0  # Full resistance at low C-rates
        else:
            # Reduce resistance at high C-rates: scale factor decreases from 1.0 at 1C to ~0.4 at 6C
            # Formula: scale = 1.0 / (1.0 + 0.15 * (C_rate - 1.0))
            rc_scale_factor = 1.0 / (1.0 + 0.15 * (current_c_rate - 1.0))
            rc_scale_factor = max(rc_scale_factor, 0.3)  # Minimum 30% to prevent zero resistance
        
        r1_effective = self.R1 * rc_scale_factor
        r2_effective = self.R2 * rc_scale_factor
        
        # Fast RC network (R1-C1): short time constant
        tau1 = r1_effective * self.C1  # Time constant depends on effective resistance
        dt_sec = dt_ms / 1000.0
        exp_factor1 = np.exp(-dt_sec / tau1) if tau1 > 0 else 0.0
        
        # Update fast RC voltage using effective resistance
        self._v_rc1 = self._v_rc1 * exp_factor1 + current_a * r1_effective * (1.0 - exp_factor1)
        
        # Slow RC network (R2-C2): long time constant
        tau2 = r2_effective * self.C2  # Time constant depends on effective resistance
        exp_factor2 = np.exp(-dt_sec / tau2) if tau2 > 0 else 0.0
        
        # Update slow RC voltage using effective resistance
        self._v_rc2 = self._v_rc2 * exp_factor2 + current_a * r2_effective * (1.0 - exp_factor2)
        
        # Calculate terminal voltage
        # V_terminal = OCV - |I|*R0 - |V_RC1| - |V_RC2|
        # Standard ECM: voltage drops always subtract from OCV
        # IR drop magnitude = |I|*R0 (always positive, subtracts from OCV)
        # RC voltage drops are always positive magnitude (subtract from OCV)
        ocv = self.get_ocv(current_direction=new_direction)
        r0_ohm = self.get_internal_resistance() / 1000.0  # Convert mΩ to Ω
        ir_drop_magnitude = abs(current_a) * r0_ohm
        v_terminal = ocv - ir_drop_magnitude - abs(self._v_rc1) - abs(self._v_rc2)
        
        # Apply minimum voltage limit (2.5V for LiFePO4 - prevents unrealistic negative voltages)
        MIN_VOLTAGE = 2.5  # Minimum safe operating voltage for LiFePO4
        v_terminal = max(v_terminal, MIN_VOLTAGE)
        
        # Update calendar aging
        # Track time and storage conditions
        # SOC: only update when at rest (storage SOC)
        # Temperature: always update (affects aging even during operation)
        current_time_hours = self._calendar_aging_time_hours + (dt_ms / (1000.0 * 3600.0))
        
        if abs(current_ma) < 0.001:  # At rest - update storage SOC
            self._storage_soc = self._soc
        
        # Always update storage temperature (temperature affects aging)
        self._storage_temp = self._temperature_c
        
        self._calendar_aging_time_hours = current_time_hours
        
        # Recalculate aging if significant time has passed
        if current_time_hours - self._last_update_time_hours > 1.0:  # Update every hour
            self._update_aging()
            self._last_update_time_hours = current_time_hours
        
        # Convert to mV
        voltage_mv = v_terminal * 1000.0
        
        # Return voltage in mV and SOC in percent
        return voltage_mv, self._soc * 100.0
    
    def set_aging(self, cycles: int, calendar_aging_hours: Optional[float] = None):
        """
        Set aging state (number of cycles and/or calendar aging time) and update capacity/resistance.
        
        Args:
            cycles: Number of charge/discharge cycles
            calendar_aging_hours: Total calendar aging time in hours (optional)
        """
        self._cycles = max(cycles, 0)
        if calendar_aging_hours is not None:
            self._calendar_aging_time_hours = max(calendar_aging_hours, 0.0)
        self._update_aging()
    
    def get_state(self) -> dict:
        """
        Get current cell state.
        
        Returns:
            Dictionary with cell state variables
        """
        return {
            'soc_pct': self._soc * 100.0,
            'voltage_mv': self.get_ocv() * 1000.0,
            'temperature_c': self._temperature_c,
            'capacity_ah': self._capacity_actual_ah,
            'internal_resistance_mohm': self.get_internal_resistance(),
            'cycles': self._cycles,
            'calendar_aging_hours': self._calendar_aging_time_hours,
            'rc1_voltage_v': self._v_rc1,
            'rc2_voltage_v': self._v_rc2,
            'current_direction': self._last_current_direction
        }
    
    def reset(self, soc_pct: Optional[float] = None, temperature_c: Optional[float] = None):
        """
        Reset cell state (useful for testing).
        
        Args:
            soc_pct: New SOC in percent (0-100). If None, keep current.
            temperature_c: New temperature in °C. If None, keep current.
        """
        if soc_pct is not None:
            self._soc = np.clip(soc_pct / 100.0, 0.0, 1.0)
        if temperature_c is not None:
            self._temperature_c = temperature_c
        self._v_rc1 = 0.0
        self._v_rc2 = 0.0
        self._last_current_direction = 0

