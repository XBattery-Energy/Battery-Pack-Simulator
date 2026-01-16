"""
BMS Simulator Web API Backend

Flask-based REST API with WebSocket support for real-time simulation data.
Provides endpoints for:
- Simulation control (start/stop/pause)
- Configuration management
- Real-time data streaming
- Fault injection
- Data analysis
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import time
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np
import traceback

# Import error logger
from logger import log_exception, log_error, get_log_file_path, get_recent_errors
from log_monitor import get_monitor

# Add parent directory to path for imports
try:
    backend_dir = Path(__file__).resolve().parent
except:
    # Fallback if __file__ is not available
    import os
    backend_dir = Path(os.getcwd())
    
# In Docker, backend is at /app/backend, so project_root should be /app
# In local dev, backend is at <project>/web_app/backend, so project_root should be <project>
if str(backend_dir).endswith('/app/backend') or str(backend_dir).endswith('\\app\\backend'):
    # Docker environment
    project_root = backend_dir.parent  # /app
else:
    # Local development
    project_root = backend_dir.parent.parent.resolve()  # <project>
pc_simulator_path = project_root / 'pc_simulator'

# Add paths to sys.path (use absolute paths)
project_root_str = str(project_root)
pc_simulator_path_str = str(pc_simulator_path)

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if pc_simulator_path_str not in sys.path:
    sys.path.insert(0, pc_simulator_path_str)

print(f"[INIT] Backend directory: {backend_dir}")
print(f"[INIT] Project root: {project_root}")
print(f"[INIT] PC Simulator path: {pc_simulator_path}")
print(f"[INIT] PC Simulator exists: {pc_simulator_path.exists()}")
print(f"[INIT] sys.path includes project_root: {project_root_str in sys.path}")
print(f"[INIT] sys.path includes pc_simulator: {pc_simulator_path_str in sys.path}")

try:
    from plant.pack_model import BatteryPack16S
    from afe.wrapper import AFEWrapper
    from communication.uart_tx import UARTTransmitter
    from communication.uart_tx_mcu import MCUCompatibleUARTTransmitter
    from communication.uart_tx_xbb import XBBUARTTransmitter
    SIMULATION_MODULES_AVAILABLE = True
    print("[INIT] OK Simulation modules imported successfully")
except ImportError as e:
    import traceback
    print(f"[INIT] ERROR Could not import simulation modules: {e}")
    print(f"[INIT] Traceback:")
    traceback.print_exc()
    print(f"[INIT] Current sys.path: {sys.path[:5]}...")  # Show first 5 entries
    SIMULATION_MODULES_AVAILABLE = False
    BatteryPack16S = None
    AFEWrapper = None

# Optional bidirectional support
try:
    from communication.uart_bidirectional import BidirectionalUART
    BIDIRECTIONAL_AVAILABLE = True
except ImportError:
    BIDIRECTIONAL_AVAILABLE = False

# Fault injection imports
try:
    from fault_injection.fault_scenarios import load_scenario, create_fault_injector_from_scenario
    FAULT_INJECTION_AVAILABLE = True
except ImportError:
    FAULT_INJECTION_AVAILABLE = False

app = Flask(__name__, static_folder='../frontend/build', static_url_path='')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Import database module
from database import BMSDatabase

# Global simulation state
simulation_state = {
    'running': False,
    'paused': False,
    'pack': None,
    'afe': None,
    'uart': None,
    'fault_injector': None,
    'simulation_thread': None,
    'current_config': None,
    'start_time': None,
    'frame_count': 0,
    'simulation_time_sec': 0.0,
    'session_id': None
}

# Initialize database (lazy - no blocking at startup)
bms_db = BMSDatabase()

# Thread lock for simulation state
simulation_lock = threading.Lock()


class SimulationManager:
    """Manages simulation execution and data streaming."""
    
    def __init__(self):
        self.running = False
        self.paused = False
        self.pack = None
        self.afe = None
        self.uart = None
        self.fault_injector = None
        self.config = None
        self.start_time = None
        self.frame_count = 0
        self.session_id = None
    
    def start_simulation(self, config: Dict[str, Any]):
        """Start simulation with given configuration."""
        with simulation_lock:
            if self.running:
                return {'success': False, 'error': 'Simulation already running'}
            
            if not SIMULATION_MODULES_AVAILABLE:
                error_msg = 'Simulation modules not available. Please install required dependencies: pip install pyserial pyyaml numpy'
                print(f"[SIM] ERROR: {error_msg}")
                return {'success': False, 'error': error_msg}
            
            try:
                print(f"[SIM] Step 1: Initializing battery pack...")
                # Initialize pack - ensure numeric types (frontend may send strings)
                try:
                    cell_capacity = float(config.get('cell_capacity_ah', 100.0))
                    initial_soc = float(config.get('initial_soc_pct', 50.0))
                    temperature = float(config.get('temperature_c', 32.0))
                    
                    self.pack = BatteryPack16S(
                        cell_capacity_ah=cell_capacity,
                        initial_soc_pct=initial_soc,
                        ambient_temp_c=temperature,
                        seed=int(config.get('seed', 42))
                    )
                    print(f"[SIM] Step 1: Battery pack initialized successfully")
                except Exception as e:
                    log_exception(type(e), e, e.__traceback__, context="SimulationManager: BatteryPack initialization")
                    raise
                
                print(f"[SIM] Step 2: Initializing AFE wrapper...")
                # Initialize AFE - ensure numeric types
                try:
                    self.afe = AFEWrapper(
                        noise_config={
                            'voltage_noise_mv': float(config.get('voltage_noise_mv', 2.0)),
                            'temp_noise_c': float(config.get('temp_noise_c', 0.5)),
                            'current_noise_ma': float(config.get('current_noise_ma', 50.0))
                        },
                        seed=42
                    )
                    self.afe.start_simulation()
                    print(f"[SIM] Step 2: AFE wrapper initialized successfully")
                except Exception as e:
                    log_exception(type(e), e, e.__traceback__, context="SimulationManager: AFE initialization")
                    raise
                
                # Initialize UART if port specified
                uart_port = config.get('uart_port')
                if uart_port and uart_port.strip():
                    protocol = config.get('protocol', 'mcu')
                    bidirectional = config.get('bidirectional', False)
                    
                    print(f"[SIM] Step 3: Initializing UART on port {uart_port}...")
                    try:
                        if bidirectional and BIDIRECTIONAL_AVAILABLE:
                            self.uart = BidirectionalUART(
                                port=uart_port,
                                baudrate=config.get('baudrate', 921600),
                                tx_rate_hz=config.get('frame_rate_hz', 50.0),
                                verbose=config.get('verbose', False)
                            )
                            if not self.uart.start():
                                print(f"[SIM] Warning: Failed to start bidirectional UART on {uart_port}")
                                self.uart = None
                        elif protocol == 'xbb':
                            self.uart = XBBUARTTransmitter(
                                port=uart_port,
                                baudrate=config.get('baudrate', 921600),
                                frame_rate_hz=config.get('frame_rate_hz', 1.0),
                                verbose=config.get('verbose', False)
                            )
                            if not self.uart.start():
                                print(f"[SIM] Warning: Failed to start XBB UART on {uart_port}")
                                self.uart = None
                        elif protocol == 'mcu':
                            self.uart = MCUCompatibleUARTTransmitter(
                                port=uart_port,
                                baudrate=config.get('baudrate', 921600),
                                frame_rate_hz=config.get('frame_rate_hz', 50.0),
                                verbose=config.get('verbose', False),
                                num_strings=1,
                                num_modules=1,
                                num_cells=16,
                                num_temp_sensors=16
                            )
                            if not self.uart.start():
                                print(f"[SIM] Warning: Failed to start MCU UART on {uart_port}")
                                self.uart = None
                        
                        if self.uart:
                            print(f"[SIM] Step 3: UART initialized successfully on {uart_port}")
                    except Exception as e:
                        log_exception(type(e), e, e.__traceback__, context="SimulationManager: UART initialization")
                        print(f"[SIM] Warning: UART initialization failed: {e}. Continuing without UART.")
                        self.uart = None
                
                # Load fault scenario if specified
                fault_scenario = config.get('fault_scenario')
                if fault_scenario and fault_scenario.strip() and FAULT_INJECTION_AVAILABLE:
                    try:
                        scenario_path = project_root / fault_scenario
                        if not scenario_path.exists():
                            log_error(f"Fault scenario file not found: {scenario_path}", context="SimulationManager: Fault Scenario")
                            print(f"[SIM] Warning: Fault scenario file not found: {scenario_path}")
                        else:
                            scenario = load_scenario(str(scenario_path))
                            self.fault_injector = create_fault_injector_from_scenario(scenario, seed=42)
                            print(f"[SIM] Loaded fault scenario: {fault_scenario}")
                    except Exception as e:
                        log_exception(type(e), e, e.__traceback__, context="SimulationManager: Fault Scenario Loading")
                        print(f"[SIM] Warning: Could not load fault scenario: {e}")
                        self.fault_injector = None
                
                print(f"[SIM] Step 4: Validating simulation mode...")
                # Validate simulation_mode is 'charge' or 'discharge' (mandatory)
                simulation_mode = config.get('simulation_mode', '').lower()
                if simulation_mode not in ['charge', 'discharge']:
                    error_msg = f"Invalid simulation_mode: '{simulation_mode}'. Must be 'charge' or 'discharge'."
                    print(f"[SIM] ERROR: {error_msg}")
                    return {'success': False, 'error': error_msg}
                
                print(f"[SIM] Step 5: Setting simulation state...")
                self.config = config
                self.running = True
                self.paused = False
                self.start_time = time.time()
                self.frame_count = 0
                
                print(f"[SIM] Step 6: Setting up BMS data callback...")
                # Set up BMS data callback if bidirectional UART is used
                if self.uart and hasattr(self.uart, 'set_bms_data_callback'):
                    try:
                        self.uart.set_bms_data_callback(lambda data: self._on_bms_data_received(data))
                        print(f"[SIM] Step 6: BMS data callback set")
                    except Exception as e:
                        log_exception(type(e), e, e.__traceback__, context="SimulationManager: BMS callback setup")
                        # Don't raise - callback setup failure shouldn't stop simulation
                
                print(f"[SIM] Step 7: Starting simulation thread...")
                # Start simulation thread
                try:
                    thread = threading.Thread(target=self._simulation_loop, daemon=True)
                    thread.start()
                    print(f"[SIM] Step 7: Simulation thread started")
                except Exception as e:
                    log_exception(type(e), e, e.__traceback__, context="SimulationManager: Thread start")
                    raise
                
                print(f"[SIM] Simulation started successfully!")
                result = {'success': True, 'message': 'Simulation started'}
            
            except Exception as e:
                # DIRECT FILE WRITE as backup - bypass logger completely
                log_file = get_log_file_path()
                try:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"EXCEPTION IN SimulationManager.start_simulation at {datetime.now().isoformat()}\n")
                        f.write(f"Exception type: {type(e).__name__}\n")
                        f.write(f"Exception message: {str(e)}\n")
                        f.write(f"Exception args: {e.args if hasattr(e, 'args') else 'N/A'}\n")
                        import traceback
                        f.write(f"Full Traceback:\n{traceback.format_exc()}\n")
                        f.write(f"{'='*80}\n")
                except Exception as file_err:
                    print(f"[CRITICAL] Failed to write to log file: {file_err}")
                
                # Try normal logging
                try:
                    log_exception(type(e), e, e.__traceback__, context="SimulationManager: start_simulation")
                except Exception as log_err:
                    print(f"[CRITICAL] Failed to log exception: {log_err}")
                
                import traceback
                error_trace = traceback.format_exc()
                print(f"[ERROR] Failed to start simulation: {e}")
                print(f"[ERROR] Traceback:\n{error_trace}")
                print(f"[ERROR] Exception type: {type(e).__name__}")
                print(f"[ERROR] Exception args: {e.args}")
                
                # Clean up on error
                self.running = False
                self.paused = False
                if self.uart:
                    try:
                        self.uart.stop()
                    except:
                        pass
                    self.uart = None
                self.pack = None
                self.afe = None
                self.fault_injector = None
                
                # Create detailed error message
                error_msg = f'Failed to start simulation: {str(e)}'
                if hasattr(e, '__traceback__') and e.__traceback__:
                    # Include first few lines of traceback in error message for debugging
                    tb_lines = traceback.format_tb(e.__traceback__)
                    if tb_lines:
                        error_msg += f'\nLocation: {tb_lines[-1].strip() if tb_lines else "Unknown"}'
                
                print(f"[SIM] Returning error: {error_msg}")
                result = {'success': False, 'error': error_msg}
            
            # Create database session AFTER releasing the lock to avoid blocking API response
            if result.get('success'):
                print(f"[SIM] Step 5: Creating database session (async)...")
                try:
                    session_name = config.get('session_name', f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    self.session_id = bms_db.create_session(session_name=session_name, config=config)
                    print(f"[SIM] Step 5: Database session created: {self.session_id}")
                except Exception as e:
                    log_exception(type(e), e, e.__traceback__, context="SimulationManager: Database session creation")
                    # Don't fail simulation if database session creation fails
                    print(f"[SIM] Warning: Database session creation failed: {e}")
                    self.session_id = None
            
            return result
    
    def stop_simulation(self):
        """Stop simulation."""
        with simulation_lock:
            self.running = False
            self.paused = False
            
            # End database session
            if self.session_id:
                bms_db.end_session(self.session_id, frame_count=self.frame_count)
                self.session_id = None
            
            if self.uart:
                self.uart.stop()
                self.uart = None
            
            self.pack = None
            self.afe = None
            self.fault_injector = None
            
            return {'success': True, 'message': 'Simulation stopped'}
    
    def pause_simulation(self):
        """Pause simulation."""
        with simulation_lock:
            if self.running:
                self.paused = True
                return {'success': True, 'message': 'Simulation paused'}
            return {'success': False, 'error': 'Simulation not running'}
    
    def resume_simulation(self):
        """Resume simulation."""
        with simulation_lock:
            if self.paused:
                self.paused = False
                return {'success': True, 'message': 'Simulation resumed'}
            return {'success': False, 'error': 'Simulation not paused'}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current simulation status."""
        with simulation_lock:
            if not self.running or not self.pack:
                return {
                    'running': False,
                    'paused': False,
                    'pack_soc': 0.0,
                    'pack_voltage': 0.0,
                    'frame_count': 0,
                    'elapsed_time': 0.0
                }
            
            return {
                'running': self.running,
                'paused': self.paused,
                'pack_soc': self.pack.get_pack_soc(),
                'pack_voltage': self.pack.get_pack_voltage() / 1000.0,
                'frame_count': self.frame_count,
                'elapsed_time': time.time() - self.start_time if self.start_time else 0.0,
                'bms_state': self.uart.get_bms_state() if self.uart and hasattr(self.uart, 'get_bms_state') else None,
                'session_id': self.session_id
            }
    
    def _on_bms_data_received(self, bms_data: Dict[str, Any]):
        """Callback for received BMS data - stores in database."""
        try:
            if self.session_id:
                bms_db.store_bms_frame(bms_data, session_id=self.session_id)
        except Exception as e:
            log_exception(type(e), e, e.__traceback__, context="SimulationManager: BMS Data Callback")
    
    def _simulation_loop(self):
        """Main simulation loop running in separate thread."""
        try:
            dt_ms = 1000.0 / self.config.get('frame_rate_hz', 50.0)
            
            # Handle simulation mode (charge or discharge - mandatory)
            simulation_mode = self.config.get('simulation_mode', 'charge')  # Default to charge if not set
            # Ensure current_amp is numeric
            current_amp = float(self.config.get('current_amp', 50.0))
            
            # Note: In cell_model, positive current = discharge (decreases SOC), negative current = charge (increases SOC)
            if simulation_mode == 'charge':
                current_ma = -abs(current_amp) * 1000.0  # Negative current for charge (increases SOC)
                print(f"[SIM] Charge mode: current_amp={current_amp}A -> current_ma={current_ma}mA (negative, SOC will INCREASE)")
            elif simulation_mode == 'discharge':
                current_ma = abs(current_amp) * 1000.0  # Positive current for discharge (decreases SOC)
                print(f"[SIM] Discharge mode: current_amp={current_amp}A -> current_ma={current_ma}mA (positive, SOC will DECREASE)")
            else:
                # Should not happen due to validation, but fallback to charge
                print(f"[SIM] WARNING: Unknown simulation_mode '{simulation_mode}', defaulting to charge")
                current_ma = -abs(current_amp) * 1000.0
            
            duration_sec = float(self.config.get('duration_sec', 0.0))
            target_soc_pct = self.config.get('target_soc_pct')
            if target_soc_pct is not None:
                target_soc_pct = float(target_soc_pct)
            use_target_soc = target_soc_pct is not None and duration_sec <= 0
            
            continuous_mode = duration_sec <= 0 and not use_target_soc
            
            step = 0
            simulation_time_ms = 0.0
            
            while self.running:
                # Check pause
                while self.paused and self.running:
                    time.sleep(0.1)
                
                if not self.running:
                    break
            
                # Check stopping conditions
                if use_target_soc:
                    # Stop at target SOC
                    current_soc = self.pack.get_pack_soc()
                    if simulation_mode == 'charge' and current_soc >= target_soc_pct:
                        break
                    elif simulation_mode == 'discharge' and current_soc <= target_soc_pct:
                        break
                elif not continuous_mode:
                    # Stop after duration
                    elapsed = time.time() - self.start_time
                    if elapsed >= duration_sec:
                        break
                
                # Update fault injector
                try:
                    if self.fault_injector:
                        pack_state = self.pack.get_pack_state()
                        self.fault_injector.update(
                            simulation_time_ms,
                            pack_state
                        )
                        self.fault_injector.apply_to_pack(self.pack)
                        for i, cell in enumerate(self.pack._cells):
                            self.fault_injector.apply_to_cell(cell, i)
                except Exception as e:
                    log_exception(type(e), e, e.__traceback__, context="Simulation Loop: Fault Injector Update")
            
                # Check current gating (bidirectional mode)
                # Note: current_ma > 0 = discharge, current_ma < 0 = charge
                effective_current_ma = current_ma
                try:
                    if self.uart and hasattr(self.uart, 'get_bms_state'):
                        bms_state = self.uart.get_bms_state()
                        # Positive current (discharge) requires discharge MOSFET
                        if current_ma > 0 and not bms_state.get('mosfet_discharge', True):
                            effective_current_ma = 0.0
                        # Negative current (charge) requires charge MOSFET
                        elif current_ma < 0 and not bms_state.get('mosfet_charge', True):
                            effective_current_ma = 0.0
                except Exception as e:
                    log_exception(type(e), e, e.__traceback__, context="Simulation Loop: BMS State Check")
                
                # Update pack
                try:
                    # Ensure temperature is numeric
                    ambient_temp = float(self.config.get('temperature_c', 32.0))
                    self.pack.update(
                        current_ma=effective_current_ma,
                        dt_ms=dt_ms,
                        ambient_temp_c=ambient_temp
                    )
                except Exception as e:
                    log_exception(type(e), e, e.__traceback__, context="Simulation Loop: Pack Update")
                    raise
                
                # Get measurements
                try:
                    true_v = self.pack.get_cell_voltages()
                    true_t = self.pack.get_cell_temperatures()
                    true_i = self.pack.get_pack_current()
                except Exception as e:
                    log_exception(type(e), e, e.__traceback__, context="Simulation Loop: Get Measurements")
                    raise
                
                # Apply AFE processing
                try:
                    measured_v, measured_t, measured_i, flags = self.afe.apply_measurement(
                        true_v, true_t, true_i
                    )
                except Exception as e:
                    log_exception(type(e), e, e.__traceback__, context="Simulation Loop: AFE Processing")
                    raise
                
                # Prepare frame data
                frame_data = {
                    'timestamp_ms': int(simulation_time_ms),
                    'time_s': simulation_time_ms / 1000.0,
                    'soc_percent': self.pack.get_pack_soc(),
                    'pack_voltage_mv': int(self.pack.get_pack_voltage()),
                    'pack_current_ma': int(effective_current_ma),
                    'cell_voltages_mv': true_v.tolist(),
                    'cell_temperatures_c': true_t.tolist(),
                    'measured_voltages_mv': measured_v.tolist(),
                    'measured_temperatures_c': (measured_t / 100.0).tolist(),
                    'status_flags': int(flags),
                    'bms_state': self.uart.get_bms_state() if self.uart and hasattr(self.uart, 'get_bms_state') else None
                }
                
                # Send via UART if available
                try:
                    if self.uart:
                        uart_frame = {
                            'timestamp_ms': int(simulation_time_ms),
                            'vcell_mv': measured_v.astype(np.uint16),
                            'tcell_cc': measured_t.astype(np.int16),
                            'pack_current_ma': int(effective_current_ma),
                            'pack_voltage_mv': int(self.pack.get_pack_voltage()),
                            'status_flags': int(flags)
                        }
                        self.uart.send_frame(uart_frame)
                except Exception as e:
                    log_exception(type(e), e, e.__traceback__, context="Simulation Loop: UART Send")
                    # Don't raise - continue simulation even if UART fails
                
                # Emit data via WebSocket
                try:
                    socketio.emit('simulation_data', frame_data)
                except Exception as e:
                    log_exception(type(e), e, e.__traceback__, context="Simulation Loop: WebSocket Emit")
                    # Don't raise - continue simulation even if WebSocket fails
                
                # Check for BMS data from RX queue (bidirectional mode)
                try:
                    if self.uart and hasattr(self.uart, 'receive_frame'):
                        bms_frame = self.uart.receive_frame(timeout=0.0)
                        if bms_frame and self.session_id:
                            bms_db.store_bms_frame(bms_frame, session_id=self.session_id)
                except Exception as e:
                    log_exception(type(e), e, e.__traceback__, context="Simulation Loop: BMS Frame Receive")
                    # Don't raise - continue simulation
                
                self.frame_count += 1
                step += 1
                simulation_time_ms += dt_ms
                
                # Rate limiting
                elapsed = time.time() - self.start_time
                expected_time = step * dt_ms / 1000.0
                sleep_time = expected_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Simulation ended normally
            self.running = False
            socketio.emit('simulation_stopped', {'message': 'Simulation completed'})
        except Exception as e:
            log_exception(type(e), e, e.__traceback__, context="Simulation Loop")
            self.running = False
            socketio.emit('simulation_stopped', {'message': f'Simulation error: {str(e)}'})


# Global simulation manager
sim_manager = SimulationManager()


# REST API Endpoints

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'bidirectional_available': BIDIRECTIONAL_AVAILABLE,
        'fault_injection_available': FAULT_INJECTION_AVAILABLE
    })


@app.route('/api/test-logging', methods=['GET'])
def test_logging():
    """Test endpoint to verify logging is working."""
    try:
        # Test logging
        test_error_msg = f"Test error at {datetime.now().isoformat()}"
        log_error(test_error_msg, context="API: /api/test-logging")
        
        # Test exception logging
        try:
            raise ValueError("Test exception for logging")
        except Exception as test_e:
            log_exception(type(test_e), test_e, test_e.__traceback__, context="API: /api/test-logging")
        
        log_file = get_log_file_path()
        return jsonify({
            'success': True,
            'message': 'Logging test completed',
            'log_file': str(log_file),
            'log_file_exists': log_file.exists(),
            'test_message': test_error_msg
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Logging test failed: {str(e)}',
            'exception_type': type(e).__name__
        }), 500


@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    """Start simulation with configuration."""
    # Direct file logging as backup
    log_file = get_log_file_path()
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[API] /api/simulation/start called at {datetime.now().isoformat()}\n")
    except:
        pass
    
    try:
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
        
        config = request.json
        if not config:
            return jsonify({'success': False, 'error': 'No configuration provided'}), 400
        
        print(f"[API] Starting simulation with config keys: {list(config.keys())}")
        print(f"[API] Config sample: {[(k, v) for k, v in list(config.items())[:5]]}")
        print(f"[API] Full config: {json.dumps(config, indent=2, default=str)}")
        
        # Validate and normalize required fields before calling start_simulation
        required_fields = ['cell_capacity_ah', 'initial_soc_pct', 'temperature_c', 'current_amp']
        missing_fields = []
        
        # Normalize numeric fields (convert strings to numbers)
        numeric_fields = ['cell_capacity_ah', 'initial_soc_pct', 'temperature_c', 'current_amp', 'duration_sec', 'frame_rate_hz', 'target_soc_pct']
        for field in numeric_fields:
            if field in config and config[field] is not None and config[field] != '':
                try:
                    config[field] = float(config[field])
                except (ValueError, TypeError) as e:
                    print(f"[API] Warning: Could not convert {field} to float: {config[field]} ({type(config[field])})")
                    if field in required_fields:
                        missing_fields.append(field)
            elif field in required_fields:
                missing_fields.append(field)
        
        if missing_fields:
            error_msg = f'Missing or invalid required fields: {missing_fields}. Config keys: {list(config.keys())}'
            print(f"[API] Validation error: {error_msg}")
            log_error(error_msg, context="API: /api/simulation/start")
            return jsonify({'success': False, 'error': error_msg}), 400
        
        print(f"[API] Calling sim_manager.start_simulation...")
        try:
            result = sim_manager.start_simulation(config)
            print(f"[API] Simulation start result: success={result.get('success')}, error={result.get('error', 'None')}")
        except Exception as e:
            # Catch any exception from start_simulation that wasn't caught internally
            try:
                log_exception(type(e), e, e.__traceback__, context="API: start_simulation call")
            except Exception as log_err:
                print(f"[CRITICAL] Failed to log exception: {log_err}")
            
            import traceback
            error_trace = traceback.format_exc()
            print(f"[API] ========== EXCEPTION IN start_simulation CALL ==========")
            print(f"[API] Exception type: {type(e).__name__}")
            print(f"[API] Exception message: {str(e)}")
            print(f"[API] Exception args: {e.args if hasattr(e, 'args') else 'N/A'}")
            print(f"[API] Full Traceback:\n{error_trace}")
            print(f"[API] =======================================================")
            result = {'success': False, 'error': f'Exception in start_simulation: {str(e)}'}
        
        # If simulation failed, return 500 with error details
        if not result.get('success'):
            error_msg = result.get('error', 'Unknown error')
            print(f"[API] Simulation start failed: {error_msg}")
            try:
                log_error(f"Simulation start failed: {error_msg}", context="API: /api/simulation/start")
            except Exception as log_err:
                print(f"[CRITICAL] Failed to log error: {log_err}")
            
            try:
                return jsonify(result), 500
            except Exception as json_error:
                # If jsonify fails, log it and return plain text
                try:
                    log_exception(type(json_error), json_error, json_error.__traceback__, context="API: JSON serialization in error response")
                except:
                    print(f"[CRITICAL] Failed to log JSON error: {json_error}")
                print(f"[API] JSON serialization error: {json_error}")
                return f'Simulation failed: {error_msg}', 500, {'Content-Type': 'text/plain'}
        
        try:
            return jsonify(result)
        except Exception as json_error:
            try:
                log_exception(type(json_error), json_error, json_error.__traceback__, context="API: JSON serialization in success response")
            except:
                print(f"[CRITICAL] Failed to log JSON error: {json_error}")
            print(f"[API] JSON serialization error for success response: {json_error}")
            return f'Simulation started but response serialization failed: {str(json_error)}', 500, {'Content-Type': 'text/plain'}
    except Exception as e:
        # Catch any exception in the route handler itself
        try:
            log_exception(type(e), e, e.__traceback__, context="API: /api/simulation/start route handler")
        except Exception as log_err:
            print(f"[CRITICAL] Failed to log exception: {log_err}")
        
        import traceback
        error_trace = traceback.format_exc()
        print(f"[API] ========== EXCEPTION IN ROUTE HANDLER ==========")
        print(f"[API] Exception type: {type(e).__name__}")
        print(f"[API] Exception message: {str(e)}")
        print(f"[API] Exception args: {e.args if hasattr(e, 'args') else 'N/A'}")
        print(f"[API] Full Traceback:\n{error_trace}")
        print(f"[API] ================================================")
        
        error_response = {'success': False, 'error': f'Server error: {str(e)}'}
        try:
            return jsonify(error_response), 500
        except Exception as json_error:
            # If even jsonify fails, return plain text
            try:
                log_exception(type(json_error), json_error, json_error.__traceback__, context="API: JSON serialization error")
            except:
                print(f"[CRITICAL] Failed to log JSON error: {json_error}")
            print(f"[API] JSON serialization failed, returning plain text")
            return f'Server error: {str(e)}', 500, {'Content-Type': 'text/plain'}


@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation():
    """Stop simulation."""
    result = sim_manager.stop_simulation()
    return jsonify(result)


@app.route('/api/simulation/pause', methods=['POST'])
def pause_simulation():
    """Pause simulation."""
    try:
        result = sim_manager.pause_simulation()
        return jsonify(result)
    except Exception as e:
        log_exception(type(e), e, e.__traceback__, context="API: /api/simulation/pause")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/simulation/resume', methods=['POST'])
def resume_simulation():
    """Resume simulation."""
    try:
        result = sim_manager.resume_simulation()
        return jsonify(result)
    except Exception as e:
        log_exception(type(e), e, e.__traceback__, context="API: /api/simulation/resume")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/simulation/status', methods=['GET'])
def get_simulation_status():
    """Get simulation status."""
    status = sim_manager.get_status()
    return jsonify(status)


@app.route('/api/config/master', methods=['GET'])
def get_master_settings():
    """Get master settings (global simulator configuration)."""
    master_settings_file = backend_dir / 'master_settings.json'
    if master_settings_file.exists():
        with open(master_settings_file, 'r') as f:
            return jsonify(json.load(f))
    else:
        # Return defaults
        return jsonify({
            'cell_capacity_ah': 100.0,
            'num_cells': 16,
            'temperature_c': 32.0,
            'protocol': 'mcu',
            'bidirectional': False,
            'uart_port': '',
            'baudrate': 921600,
            'frame_rate_hz': 50.0,
            'voltage_noise_mv': 2.0,
            'temp_noise_c': 0.5,
            'current_noise_ma': 50.0,
        })


@app.route('/api/config/master', methods=['POST'])
def save_master_settings():
    """Save master settings (global simulator configuration)."""
    master_settings_file = backend_dir / 'master_settings.json'
    try:
        settings = request.json
        with open(master_settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
        return jsonify({'success': True, 'message': 'Master settings saved'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/config/default', methods=['GET'])
def get_default_config():
    """Get default simulation configuration (deprecated - use master + session)."""
    return jsonify({
        'cell_capacity_ah': 100.0,
        'initial_soc_pct': 50.0,
        'temperature_c': 32.0,
        'current_amp': 50.0,
        'duration_sec': 3600.0,
        'frame_rate_hz': 50.0,
        'protocol': 'mcu',
        'bidirectional': False,
        'uart_port': None,
        'baudrate': 921600,
        'voltage_noise_mv': 2.0,
        'temp_noise_c': 0.5,
        'current_noise_ma': 50.0,
        'fault_scenario': None
    })


@app.route('/api/fault-scenarios', methods=['GET'])
def list_fault_scenarios():
    """List available fault scenarios."""
    scenarios_dir = project_root / 'scenarios' / 'deterministic'
    scenarios = []
    
    if scenarios_dir.exists():
        for yaml_file in scenarios_dir.glob('*.yaml'):
            scenarios.append({
                'name': yaml_file.stem,
                'path': str(yaml_file.relative_to(project_root))
            })
    
    return jsonify({'scenarios': scenarios})


@app.route('/api/fault-scenarios/<scenario_name>', methods=['GET'])
def get_fault_scenario(scenario_name: str):
    """Get fault scenario details."""
    try:
        scenarios_dir = project_root / 'scenarios' / 'deterministic'
        scenario_file = scenarios_dir / f'{scenario_name}.yaml'
        
        if not scenario_file.exists():
            return jsonify({'error': 'Scenario not found'}), 404
    except Exception as e:
        log_exception(type(e), e, e.__traceback__, context="API: /api/fault-scenarios/<name>")
        return jsonify({'error': str(e)}), 500
    
    try:
        scenario = load_scenario(str(scenario_file))
        return jsonify(scenario)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Database and Analysis Endpoints

@app.route('/api/sessions-test', methods=['GET'])
def sessions_test():
    """Simple test endpoint."""
    return jsonify({'test': 'ok', 'timestamp': datetime.now().isoformat()})


@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all simulation sessions."""
    limit = request.args.get('limit', 50, type=int)
    print(f"[API] /api/sessions called with limit={limit}")
    
    try:
        sessions = bms_db.get_sessions(limit=limit)
        print(f"[API] Found {len(sessions)} sessions")
        return jsonify({'sessions': sessions})
    except Exception as e:
        print(f"[API] Error: {e}")
        return jsonify({'sessions': [], 'error': str(e)}), 500


@app.route('/api/sessions/<int:session_id>', methods=['GET'])
def get_session(session_id: int):
    """Get session details."""
    sessions = bms_db.get_sessions(limit=1000)
    session = next((s for s in sessions if s['id'] == session_id), None)
    
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    
    return jsonify(session)


@app.route('/api/sessions/<int:session_id>/frames', methods=['GET'])
def get_session_frames(session_id: int):
    """Get BMS frames for a session."""
    start_time = request.args.get('start_time_ms', type=int)
    end_time = request.args.get('end_time_ms', type=int)
    limit = request.args.get('limit', 1000, type=int)
    
    frames = bms_db.get_frames(
        session_id=session_id,
        start_time_ms=start_time,
        end_time_ms=end_time,
        limit=limit
    )
    
    return jsonify({'frames': frames, 'count': len(frames)})


@app.route('/api/sessions/<int:session_id>/statistics', methods=['GET'])
def get_session_statistics(session_id: int):
    """Get statistics for a session."""
    stats = bms_db.get_statistics(session_id=session_id)
    return jsonify(stats)


@app.route('/api/statistics', methods=['GET'])
def get_all_statistics():
    """Get overall statistics."""
    stats = bms_db.get_statistics()
    return jsonify(stats)


@app.route('/api/sessions/<int:session_id>/export', methods=['GET'])
def export_session(session_id: int):
    """Export session data to CSV."""
    try:
        output_path = f"bms_export_session_{session_id}.csv"
        file_path = bms_db.export_to_csv(session_id=session_id, output_path=output_path)
        
        return send_from_directory(
            directory=Path(file_path).parent,
            path=Path(file_path).name,
            as_attachment=True,
            download_name=f"bms_data_session_{session_id}.csv"
        )
    except Exception as e:
        log_exception(type(e), e, e.__traceback__, context="API: /api/sessions/<id>/export")
        return jsonify({'error': str(e)}), 500


# Log Monitoring Endpoints

@app.route('/api/logs/errors', methods=['GET'])
def get_recent_log_errors():
    """Get recent error entries from log file."""
    try:
        limit = request.args.get('limit', 50, type=int)
        errors = get_recent_errors(limit=limit)
        return jsonify({
            'errors': errors,
            'count': len(errors),
            'log_file': str(get_log_file_path())
        })
    except Exception as e:
        log_exception(type(e), e, e.__traceback__, context="API: /api/logs/errors")
        return jsonify({'error': str(e)}), 500


@app.route('/api/logs/stats', methods=['GET'])
def get_log_stats():
    """Get log file statistics."""
    try:
        monitor = get_monitor()
        stats = monitor.get_log_stats()
        return jsonify(stats)
    except Exception as e:
        log_exception(type(e), e, e.__traceback__, context="API: /api/logs/stats")
        return jsonify({'error': str(e)}), 500


@app.route('/api/logs/new', methods=['GET'])
def get_new_log_errors():
    """Get new error entries since last check."""
    try:
        monitor = get_monitor()
        new_errors = monitor.get_new_errors()
        return jsonify({
            'errors': new_errors,
            'count': len(new_errors),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        log_exception(type(e), e, e.__traceback__, context="API: /api/logs/new")
        return jsonify({'error': str(e)}), 500


# WebSocket Events

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection."""
    emit('connected', {'message': 'Connected to BMS Simulator'})
    # Send current status
    status = sim_manager.get_status()
    emit('simulation_status', status)


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    pass


@socketio.on('request_status')
def handle_status_request():
    """Handle status request."""
    status = sim_manager.get_status()
    emit('simulation_status', status)


# Serve React app
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    """Serve React application."""
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


# Flask error handlers - MUST be registered before routes
@app.errorhandler(404)
def not_found(error):
    try:
        log_error(f"404 Not Found: {request.url}", context="Flask Error Handler")
    except:
        pass
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    # DIRECT FILE WRITE as backup - bypass logger completely
    log_file = get_log_file_path()
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"500 Internal Server Error at {datetime.now().isoformat()}\n")
            f.write(f"Error: {str(error)}\n")
            f.write(f"Type: {type(error).__name__}\n")
            import traceback
            f.write(f"Traceback:\n{traceback.format_exc()}\n")
            f.write(f"{'='*80}\n")
    except Exception as log_err:
        print(f"[CRITICAL] Failed to write to log file: {log_err}")
    
    try:
        log_error(f"500 Internal Server Error: {str(error)}", context="Flask Error Handler")
    except Exception as log_error_exc:
        print(f"[CRITICAL] Failed to log error: {log_error_exc}")
    
    print(f"[ERROR] 500 Internal Server Error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    # DIRECT FILE WRITE as backup - bypass logger completely
    log_file = get_log_file_path()
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Unhandled Exception at {datetime.now().isoformat()}\n")
            f.write(f"Exception: {str(e)}\n")
            f.write(f"Type: {type(e).__name__}\n")
            import traceback
            f.write(f"Traceback:\n{traceback.format_exc()}\n")
            f.write(f"{'='*80}\n")
    except Exception as log_err:
        print(f"[CRITICAL] Failed to write to log file: {log_err}")
    
    try:
        log_exception(type(e), e, e.__traceback__, context="Flask Unhandled Exception")
    except Exception as log_error_exc:
        print(f"[CRITICAL] Failed to log exception: {log_error_exc}")
    
    print(f"[ERROR] Unhandled Exception: {e}")
    import traceback
    traceback.print_exc()
    return jsonify({'error': f'Unhandled exception: {str(e)}'}), 500


if __name__ == '__main__':
    log_file_path = get_log_file_path()
    print(f"[SERVER] Error log file: {log_file_path}")
    print(f"[SERVER] All runtime errors will be logged to: {log_file_path}")
    print(f"[SERVER] Monitor log file at: {log_file_path}")
    print(f"[SERVER] Or use API endpoints: /api/logs/errors, /api/logs/stats, /api/logs/new")
    print(f"[SERVER] Or run: python watch_logs.py")
    
    # Initialize log monitor
    monitor = get_monitor()
    print(f"[SERVER] Log monitor initialized")
    
    socketio.run(app, host='0.0.0.0', port=5050, debug=False, allow_unsafe_werkzeug=True)
