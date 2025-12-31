"""
Unit tests for Current Profile Generator.
"""

import pytest
import numpy as np
import tempfile
import os
from sil_bms.pc_simulator.plant.current_profile import CurrentProfile, ProfileType


class TestCurrentProfile:
    """Test suite for CurrentProfile class."""
    
    def test_constant_profile(self):
        """Test constant current profile."""
        profile = CurrentProfile('constant', current_a=50.0, duration_sec=3600.0)
        
        # Test at various times
        assert profile.get_current_at_time(0.0) == 50000.0  # 50A = 50000mA
        assert profile.get_current_at_time(1000.0) == 50000.0
        assert profile.get_current_at_time(3600.0) == 50000.0
        
        # Test beyond duration
        assert profile.get_current_at_time(4000.0) == 0.0
        
        # Test duration
        assert profile.get_duration() == 3600.0
    
    def test_constant_profile_infinite(self):
        """Test constant profile with infinite duration."""
        profile = CurrentProfile('constant', current_a=-100.0)
        
        # Should work at any time
        assert profile.get_current_at_time(0.0) == -100000.0
        assert profile.get_current_at_time(10000.0) == -100000.0
        assert profile.get_duration() == float('inf')
    
    def test_pulse_profile(self):
        """Test pulse (square wave) profile."""
        profile = CurrentProfile(
            'pulse',
            current_high_a=100.0,
            current_low_a=-50.0,
            period_sec=60.0,
            duty_cycle=0.5,
            duration_sec=300.0
        )
        
        # At t=0, should be high (duty cycle starts with high)
        assert profile.get_current_at_time(0.0) == 100000.0  # 100A
        
        # At t=30s (half period, should be low)
        assert profile.get_current_at_time(30.0) == -50000.0  # -50A
        
        # At t=60s (one period, should be high again)
        assert profile.get_current_at_time(60.0) == 100000.0
        
        # At t=90s (1.5 periods, should be low)
        assert profile.get_current_at_time(90.0) == -50000.0
        
        # Test duration
        assert profile.get_duration() == 300.0
    
    def test_pulse_profile_duty_cycle(self):
        """Test pulse profile with different duty cycles."""
        # 25% duty cycle (high for 15s, low for 45s)
        profile = CurrentProfile(
            'pulse',
            current_high_a=100.0,
            current_low_a=0.0,
            period_sec=60.0,
            duty_cycle=0.25
        )
        
        # At t=0, should be high
        assert profile.get_current_at_time(0.0) == 100000.0
        
        # At t=15s (end of high period), should transition to low
        assert profile.get_current_at_time(15.0) == 0.0
        
        # At t=30s (in low period), should be low
        assert profile.get_current_at_time(30.0) == 0.0
    
    def test_pulse_profile_phase(self):
        """Test pulse profile with phase offset."""
        profile = CurrentProfile(
            'pulse',
            current_high_a=100.0,
            current_low_a=0.0,
            period_sec=60.0,
            duty_cycle=0.5,
            phase_sec=30.0  # 30s phase offset
        )
        
        # At t=0, with 30s phase, should be at 30s in period = low
        assert profile.get_current_at_time(0.0) == 0.0
        
        # At t=30s, should be at 60s in period = high (wrapped)
        assert profile.get_current_at_time(30.0) == 100000.0
    
    def test_yaml_profile(self):
        """Test YAML profile loading."""
        yaml_content = """
name: "Test_Profile"
duration_sec: 3600
segments:
  - time_range: [0, 1800]
    current_a: 50
    description: "Charge"
  - time_range: [1800, 3600]
    current_a: -100
    description: "Discharge"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_file = f.name
        
        try:
            profile = CurrentProfile('yaml', yaml_file=yaml_file)
            
            # Test first segment
            assert profile.get_current_at_time(0.0) == 50000.0  # 50A
            assert profile.get_current_at_time(900.0) == 50000.0
            assert profile.get_current_at_time(1799.0) == 50000.0
            
            # Test second segment
            assert profile.get_current_at_time(1800.0) == -100000.0  # -100A
            assert profile.get_current_at_time(2700.0) == -100000.0
            assert profile.get_current_at_time(3599.0) == -100000.0
            
            # Test beyond duration
            assert profile.get_current_at_time(4000.0) == 0.0
            
            # Test duration
            assert profile.get_duration() == 3600.0
            
        finally:
            os.unlink(yaml_file)
    
    def test_yaml_profile_smooth_transitions(self):
        """Test YAML profile with smooth transitions."""
        yaml_content = """
name: "Test_Profile"
duration_sec: 3600
segments:
  - time_range: [0, 1800]
    current_a: 50
  - time_range: [1800, 3600]
    current_a: -100
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_file = f.name
        
        try:
            profile = CurrentProfile(
                'yaml',
                yaml_file=yaml_file,
                smooth_transitions=True,
                transition_duration_sec=10.0
            )
            
            # Before transition
            assert abs(profile.get_current_at_time(1790.0) - 50000.0) < 1000
            
            # During transition (should interpolate)
            current_transition = profile.get_current_at_time(1805.0)
            assert -100000.0 < current_transition < 50000.0
            
            # After transition
            assert abs(profile.get_current_at_time(1810.0) - (-100000.0)) < 1000
            
        finally:
            os.unlink(yaml_file)
    
    def test_dynamic_profile_function(self):
        """Test dynamic profile with function."""
        def sin_current(t):
            return 50.0 * np.sin(2 * np.pi * t / 3600.0)
        
        profile = CurrentProfile('dynamic', function=sin_current, duration_sec=3600.0)
        
        # At t=0, sin(0) = 0
        assert abs(profile.get_current_at_time(0.0)) < 1.0
        
        # At t=900s (quarter period), sin(π/2) = 1, so 50A
        assert abs(profile.get_current_at_time(900.0) - 50000.0) < 1000.0
        
        # At t=1800s (half period), sin(π) = 0
        assert abs(profile.get_current_at_time(1800.0)) < 1.0
        
        # At t=2700s (three-quarter period), sin(3π/2) = -1, so -50A
        assert abs(profile.get_current_at_time(2700.0) - (-50000.0)) < 1000.0
    
    def test_dynamic_profile_expression(self):
        """Test dynamic profile with expression."""
        profile = CurrentProfile(
            'dynamic',
            expression="50 * sin(2 * pi * t / 3600)",
            duration_sec=3600.0
        )
        
        # At t=0, sin(0) = 0
        assert abs(profile.get_current_at_time(0.0)) < 1.0
        
        # At t=900s, should be ~50A
        assert abs(profile.get_current_at_time(900.0) - 50000.0) < 1000.0
    
    def test_dynamic_profile_complex_expression(self):
        """Test dynamic profile with complex expression."""
        profile = CurrentProfile(
            'dynamic',
            expression="50 + 30 * sin(2 * pi * t / 1800) - 20 * cos(4 * pi * t / 1800)",
            duration_sec=3600.0
        )
        
        # Should evaluate without error
        current = profile.get_current_at_time(100.0)
        assert isinstance(current, (int, float))
        assert -200000.0 < current < 200000.0  # Reasonable range
    
    def test_get_duration(self):
        """Test get_duration() method."""
        # Constant with duration
        profile1 = CurrentProfile('constant', current_a=50.0, duration_sec=3600.0)
        assert profile1.get_duration() == 3600.0
        
        # Constant infinite
        profile2 = CurrentProfile('constant', current_a=50.0)
        assert profile2.get_duration() == float('inf')
        
        # Pulse with duration
        profile3 = CurrentProfile('pulse', current_high_a=100.0, current_low_a=0.0,
                                 period_sec=60.0, duration_sec=300.0)
        assert profile3.get_duration() == 300.0
    
    def test_get_profile_info(self):
        """Test get_profile_info() method."""
        profile = CurrentProfile('constant', current_a=50.0, duration_sec=3600.0)
        
        info = profile.get_profile_info()
        
        assert info['profile_type'] == 'constant'
        assert info['duration_sec'] == 3600.0
        assert info['current_a'] == 50.0
        assert info['smooth_transitions'] == False
    
    def test_generate_time_series(self):
        """Test generate_time_series() method."""
        profile = CurrentProfile('constant', current_a=50.0, duration_sec=100.0)
        
        time_array, current_array = profile.generate_time_series(dt_sec=1.0, t_start=0.0)
        
        assert len(time_array) == len(current_array)
        assert len(time_array) == 100  # 100 seconds / 1 second step
        assert np.allclose(current_array, 50000.0)  # All should be 50A = 50000mA
    
    def test_pulse_smooth_transitions(self):
        """Test pulse profile with smooth transitions."""
        profile = CurrentProfile(
            'pulse',
            current_high_a=100.0,
            current_low_a=0.0,
            period_sec=60.0,
            duty_cycle=0.5,
            smooth_transitions=True,
            transition_duration_sec=5.0
        )
        
        # At transition point (30s), should be interpolated
        current = profile.get_current_at_time(30.0)
        assert 0.0 < current < 100000.0  # Should be between high and low
    
    def test_yaml_profile_multiple_segments(self):
        """Test YAML profile with multiple segments."""
        yaml_content = """
name: "Multi_Segment"
duration_sec: 1800
segments:
  - time_range: [0, 300]
    current_a: 25
  - time_range: [300, 600]
    current_a: 50
  - time_range: [600, 900]
    current_a: 75
  - time_range: [900, 1800]
    current_a: 100
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_file = f.name
        
        try:
            profile = CurrentProfile('yaml', yaml_file=yaml_file)
            
            # Test each segment
            assert profile.get_current_at_time(150.0) == 25000.0
            assert profile.get_current_at_time(450.0) == 50000.0
            assert profile.get_current_at_time(750.0) == 75000.0
            assert profile.get_current_at_time(1350.0) == 100000.0
            
        finally:
            os.unlink(yaml_file)
    
    def test_yaml_profile_validation_overlap(self):
        """Test YAML profile validation (overlapping segments)."""
        yaml_content = """
name: "Invalid_Profile"
duration_sec: 1800
segments:
  - time_range: [0, 600]
    current_a: 50
  - time_range: [500, 1200]  # Overlaps with previous
    current_a: -50
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_file = f.name
        
        try:
            with pytest.raises(ValueError, match="overlap"):
                profile = CurrentProfile('yaml', yaml_file=yaml_file)
        finally:
            os.unlink(yaml_file)
    
    def test_load_from_yaml(self):
        """Test load_from_yaml() method."""
        yaml_content = """
name: "Loaded_Profile"
duration_sec: 3600
segments:
  - time_range: [0, 3600]
    current_a: 50
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_file = f.name
        
        try:
            profile = CurrentProfile('constant', current_a=0.0)  # Initial profile
            profile.load_from_yaml(yaml_file)
            
            # Should now have YAML profile values
            assert profile.get_current_at_time(0.0) == 50000.0
            assert profile.get_duration() == 3600.0
            
        finally:
            os.unlink(yaml_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

