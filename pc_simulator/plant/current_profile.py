"""
Current Profile Generator for Battery Simulation

This module provides various current profile types for battery simulation:
- Constant current
- Pulse (square wave)
- YAML-based profiles with time segments
- Dynamic profiles (sinusoidal, etc.)
"""

import numpy as np
import yaml
from typing import Optional, List, Dict, Union, Callable
from enum import Enum


class ProfileType(Enum):
    """Current profile types."""
    CONSTANT = "constant"
    PULSE = "pulse"
    YAML = "yaml"
    DYNAMIC = "dynamic"


class CurrentProfile:
    """
    Current Profile Generator
    
    Supports multiple profile types:
    - Constant: fixed current value
    - Pulse: square wave current
    - YAML: time-based segments from YAML file
    - Dynamic: current as function of time (sinusoidal, etc.)
    
    Parameters:
        profile_type: Type of profile (ProfileType enum or string)
        **kwargs: Profile-specific parameters
    """
    
    def __init__(
        self,
        profile_type: Union[ProfileType, str],
        smooth_transitions: bool = False,
        transition_duration_sec: float = 1.0,
        **kwargs
    ):
        """
        Initialize current profile.
        
        Args:
            profile_type: Profile type ('constant', 'pulse', 'yaml', 'dynamic')
            smooth_transitions: Enable smooth transitions between segments (default: False)
            transition_duration_sec: Transition duration in seconds (default: 1.0)
            **kwargs: Profile-specific parameters:
                - constant: current_a (float)
                - pulse: current_high_a, current_low_a, period_sec, duty_cycle (0-1)
                - yaml: yaml_file (str) or yaml_data (dict)
                - dynamic: function (callable) or expression (str)
        """
        if isinstance(profile_type, str):
            try:
                profile_type = ProfileType(profile_type.lower())
            except ValueError:
                raise ValueError(f"Unknown profile type: {profile_type}")
        
        self._profile_type = profile_type
        self._smooth_transitions = smooth_transitions
        self._transition_duration_sec = transition_duration_sec
        
        # Initialize profile-specific data
        self._duration_sec = 0.0
        self._segments = []
        self._dynamic_function = None
        
        # Initialize based on profile type
        if profile_type == ProfileType.CONSTANT:
            self._init_constant(**kwargs)
        elif profile_type == ProfileType.PULSE:
            self._init_pulse(**kwargs)
        elif profile_type == ProfileType.YAML:
            self._init_yaml(**kwargs)
        elif profile_type == ProfileType.DYNAMIC:
            self._init_dynamic(**kwargs)
        else:
            raise ValueError(f"Unsupported profile type: {profile_type}")
    
    def _init_constant(self, current_a: float = 0.0, duration_sec: Optional[float] = None):
        """Initialize constant current profile."""
        self._current_a = current_a
        self._duration_sec = duration_sec if duration_sec is not None else float('inf')
        self._segments = [{'time_range': [0.0, self._duration_sec], 'current_a': current_a}]
    
    def _init_pulse(
        self,
        current_high_a: float,
        current_low_a: float,
        period_sec: float,
        duty_cycle: float = 0.5,
        duration_sec: Optional[float] = None,
        phase_sec: float = 0.0
    ):
        """
        Initialize pulse (square wave) profile.
        
        Args:
            current_high_a: High current value (A)
            current_low_a: Low current value (A)
            period_sec: Period in seconds
            duty_cycle: Duty cycle (0-1), fraction of period at high current
            duration_sec: Total duration (None = infinite)
            phase_sec: Phase offset in seconds
        """
        if not 0.0 <= duty_cycle <= 1.0:
            raise ValueError("Duty cycle must be between 0 and 1")
        
        self._current_high_a = current_high_a
        self._current_low_a = current_low_a
        self._period_sec = period_sec
        self._duty_cycle = duty_cycle
        self._phase_sec = phase_sec
        self._duration_sec = duration_sec if duration_sec is not None else float('inf')
        self._high_duration_sec = period_sec * duty_cycle
        self._low_duration_sec = period_sec * (1.0 - duty_cycle)
    
    def _init_yaml(self, yaml_file: Optional[str] = None, yaml_data: Optional[Dict] = None):
        """Initialize profile from YAML file or data."""
        if yaml_file is not None:
            with open(yaml_file, 'r') as f:
                yaml_data = yaml.safe_load(f)
        
        if yaml_data is None:
            raise ValueError("Either yaml_file or yaml_data must be provided")
        
        # Extract profile data
        self._profile_name = yaml_data.get('name', 'Unnamed Profile')
        self._duration_sec = yaml_data.get('duration_sec', 0.0)
        segments_data = yaml_data.get('segments', [])
        
        # Parse segments
        self._segments = []
        for seg in segments_data:
            time_range = seg.get('time_range', [0.0, 0.0])
            current_a = seg.get('current_a', 0.0)
            description = seg.get('description', '')
            
            self._segments.append({
                'time_range': time_range,
                'current_a': current_a,
                'description': description
            })
        
        # Sort segments by start time
        self._segments.sort(key=lambda x: x['time_range'][0])
        
        # Validate segments (no overlaps, continuous coverage)
        self._validate_segments()
    
    def _init_dynamic(
        self,
        function: Optional[Callable] = None,
        expression: Optional[str] = None,
        duration_sec: Optional[float] = None
    ):
        """
        Initialize dynamic profile (current as function of time).
        
        Args:
            function: Callable that takes time (seconds) and returns current (A)
            expression: String expression like "50 * sin(2*pi*t/3600)" (uses numpy)
            duration_sec: Total duration (None = infinite)
        """
        if function is not None:
            self._dynamic_function = function
        elif expression is not None:
            # Create function from expression
            # Expression can use: t (time), np (numpy), sin, cos, etc.
            def make_function(expr):
                def func(t):
                    # Create safe namespace for evaluation
                    namespace = {
                        't': t,
                        'np': np,
                        'sin': np.sin,
                        'cos': np.cos,
                        'tan': np.tan,
                        'exp': np.exp,
                        'log': np.log,
                        'sqrt': np.sqrt,
                        'abs': np.abs,
                        'pi': np.pi,
                        'e': np.e
                    }
                    try:
                        return float(eval(expr, namespace))
                    except Exception as e:
                        raise ValueError(f"Error evaluating expression '{expr}': {e}")
                return func
            
            self._dynamic_function = make_function(expression)
        else:
            raise ValueError("Either function or expression must be provided")
        
        self._duration_sec = duration_sec if duration_sec is not None else float('inf')
    
    def _validate_segments(self):
        """Validate YAML segments (no overlaps, continuous coverage)."""
        if not self._segments:
            return
        
        # Check for overlaps
        for i in range(len(self._segments) - 1):
            current_end = self._segments[i]['time_range'][1]
            next_start = self._segments[i + 1]['time_range'][0]
            
            if current_end > next_start:
                raise ValueError(f"Segment overlap: segment {i} ends at {current_end}s, "
                               f"segment {i+1} starts at {next_start}s")
    
    def get_current_at_time(self, t_sec: float) -> float:
        """
        Get current at specified time.
        
        Args:
            t_sec: Time in seconds
        
        Returns:
            Current in mA (milliamperes)
        """
        # Check duration limit
        if t_sec > self._duration_sec:
            return 0.0  # Return 0 after duration
        
        if self._profile_type == ProfileType.CONSTANT:
            return self._current_a * 1000.0  # Convert A to mA
        
        elif self._profile_type == ProfileType.PULSE:
            # Calculate position in period
            t_relative = (t_sec + self._phase_sec) % self._period_sec
            
            if t_relative < self._high_duration_sec:
                current_a = self._current_high_a
            else:
                current_a = self._current_low_a
            
            # Apply smooth transition if enabled
            if self._smooth_transitions:
                # Smooth transition at duty cycle boundary
                transition_width = min(self._transition_duration_sec, self._period_sec * 0.1)
                if abs(t_relative - self._high_duration_sec) < transition_width / 2:
                    # Linear interpolation during transition
                    transition_pos = (t_relative - (self._high_duration_sec - transition_width / 2)) / transition_width
                    current_a = self._current_high_a * (1.0 - transition_pos) + self._current_low_a * transition_pos
            
            return current_a * 1000.0  # Convert A to mA
        
        elif self._profile_type == ProfileType.YAML:
            # Find segment containing this time
            for i, seg in enumerate(self._segments):
                t_start, t_end = seg['time_range']
                
                if t_start <= t_sec < t_end:
                    current_a = seg['current_a']
                    
                    # Apply smooth transition if enabled and not first/last segment
                    if self._smooth_transitions and i > 0:
                        prev_seg = self._segments[i - 1]
                        prev_end = prev_seg['time_range'][1]
                        
                        # Check if we're in transition region
                        if t_sec < prev_end + self._transition_duration_sec:
                            # Linear ramp from previous to current
                            transition_pos = (t_sec - prev_end) / self._transition_duration_sec
                            transition_pos = np.clip(transition_pos, 0.0, 1.0)
                            prev_current_a = prev_seg['current_a']
                            current_a = prev_current_a * (1.0 - transition_pos) + current_a * transition_pos
                    
                    return current_a * 1000.0  # Convert A to mA
            
            # Time not in any segment
            return 0.0
        
        elif self._profile_type == ProfileType.DYNAMIC:
            # Call dynamic function
            current_a = self._dynamic_function(t_sec)
            return current_a * 1000.0  # Convert A to mA
        
        else:
            return 0.0
    
    def load_from_yaml(self, yaml_file: str):
        """
        Load profile from YAML file.
        
        Args:
            yaml_file: Path to YAML file
        """
        self._init_yaml(yaml_file=yaml_file)
    
    def get_duration(self) -> float:
        """
        Get total duration of profile.
        
        Returns:
            Duration in seconds (float('inf') for infinite profiles)
        """
        return self._duration_sec
    
    def get_profile_info(self) -> Dict:
        """
        Get profile information.
        
        Returns:
            Dictionary with profile metadata
        """
        info = {
            'profile_type': self._profile_type.value,
            'duration_sec': self._duration_sec if self._duration_sec != float('inf') else None,
            'smooth_transitions': self._smooth_transitions,
            'transition_duration_sec': self._transition_duration_sec
        }
        
        if self._profile_type == ProfileType.CONSTANT:
            info['current_a'] = self._current_a
        
        elif self._profile_type == ProfileType.PULSE:
            info.update({
                'current_high_a': self._current_high_a,
                'current_low_a': self._current_low_a,
                'period_sec': self._period_sec,
                'duty_cycle': self._duty_cycle
            })
        
        elif self._profile_type == ProfileType.YAML:
            info.update({
                'name': getattr(self, '_profile_name', 'Unnamed'),
                'num_segments': len(self._segments)
            })
        
        elif self._profile_type == ProfileType.DYNAMIC:
            info['has_function'] = self._dynamic_function is not None
        
        return info
    
    def generate_time_series(self, dt_sec: float = 1.0, t_start: float = 0.0) -> tuple:
        """
        Generate time series of current values.
        
        Args:
            dt_sec: Time step in seconds
            t_start: Start time in seconds
        
        Returns:
            Tuple of (time_array, current_array_mA)
        """
        if self._duration_sec == float('inf'):
            # For infinite profiles, generate up to a reasonable limit
            t_end = t_start + 3600.0  # 1 hour default
        else:
            t_end = min(self._duration_sec, t_start + 3600.0)
        
        time_array = np.arange(t_start, t_end, dt_sec)
        current_array = np.array([self.get_current_at_time(t) for t in time_array])
        
        return time_array, current_array

