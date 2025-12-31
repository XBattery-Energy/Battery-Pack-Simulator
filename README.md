# Battery Pack Simulator

A **Software-in-the-Loop (SIL) Battery Management System (BMS) Simulator** for 16-cell series (16S) LiFePO₄ battery packs. This simulator provides realistic battery behavior modeling, AFE (Analog Front End) measurement simulation, and UART communication protocols for testing BMS hardware and software.

**Repository:** [XBattery-Energy/Battery-Pack-Simulator](https://github.com/XBattery-Energy/Battery-Pack-Simulator)

---

## 🚀 Features

### Battery Modeling
- **16S LiFePO₄ Pack Model** with cell-to-cell variations
- **Detailed ECM (Equivalent Circuit Model)** with 2RC network
- **Hysteresis modeling** (separate charge/discharge OCV curves)
- **Temperature effects** on OCV, capacity, and resistance
- **Aging models** (cycle aging + calendar aging)
- **Thermal coupling** between adjacent cells
- **Fault injection** capabilities

### AFE Simulation
- **MC33774 AFE measurement simulation**
- **ADC quantization** (16-bit voltage/current, 12-bit temperature)
- **Gaussian noise injection** (configurable)
- **Per-channel calibration errors** (gain/offset)
- **Fault injection** (open wire, stuck ADC, NTC faults, etc.)
- **Time-based fault scheduling**

### Communication Protocols
- **XBB Protocol** - Custom protocol with CRC8 checksum
- **MCU Protocol** - MCU-compatible format
- **Legacy Protocol** - Backward compatibility
- **UART transmission** with configurable baud rates
- **Thread-safe frame queue** with rate limiting

### Current Profiles
- **Constant current**
- **Pulse (square wave)**
- **YAML-based profiles** with time segments
- **Dynamic profiles** (user-defined functions)

---

## 📋 Requirements

- Python 3.8 or higher
- See `requirements.txt` for dependencies

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/XBattery-Energy/Battery-Pack-Simulator.git
cd Battery-Pack-Simulator
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🎯 Quick Start

### Two Main Scripts

The project has two main entry points:

1. **`main.py`** - Full SIL simulator with AFE wrapper and UART communication (for BMS testing)
2. **`run_cell_simulation.py`** - Standalone pack simulation for analysis (saves CSV/plots, no AFE/UART)

### Print-Only Mode (No UART)

Run simulation and print frame data to console:

```bash
python pc_simulator/main.py --current 50.0 --duration 10.0
```

### With UART Transmission

Connect to physical BMS hardware via serial port:

```bash
# Windows
python pc_simulator/main.py --port COM3 --current 50.0 --rate 1.0 --protocol xbb

# Linux/Mac
python pc_simulator/main.py --port /dev/ttyUSB0 --current 50.0 --rate 1.0 --protocol xbb
```

### Continuous Mode

Run infinite simulation (press Ctrl+C to stop):

```bash
python pc_simulator/main.py --port COM3 --current 50.0 --duration 0
```

### Standalone Pack Simulation (Analysis Mode)

For analysis and testing without AFE/UART:

```bash
python pc_simulator/plant/run_cell_simulation.py --mode discharge --current 1.0 --duration 60
```

This script:
- Runs pack simulation only (no AFE wrapper, no UART)
- Saves CSV data to `pc_simulator/plant/output/`
- Optionally generates plots
- Useful for model validation and analysis

**When to use which script:**
- **`main.py`**: Use for BMS testing, HIL testing, or when you need AFE simulation and UART communication
- **`run_cell_simulation.py`**: Use for analysis, model validation, or when you just need pack simulation data

---

## 📖 Usage

### Command-Line Arguments

```bash
python pc_simulator/main.py [OPTIONS]
```

**Options:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--port` | str | None | Serial port (e.g., COM3, /dev/ttyUSB0). If not specified, print-only mode |
| `--baudrate` | int | 921600 | Baud rate for UART communication |
| `--rate` | float | 1.0 | Frame rate in Hz |
| `--duration` | float | 10.0 | Simulation duration in seconds (0 = infinite) |
| `--current` | float | 50.0 | Pack current in Amperes (positive = charge, negative = discharge) |
| `--soc` | float | 50.0 | Initial SOC in percent |
| `--protocol` | str | xbb | Protocol type: `xbb`, `mcu`, or `legacy` |
| `--verbose` | flag | False | Enable verbose logging |
| `--no-print` | flag | False | Disable frame printing |

### Examples

#### Example 1: Charge Simulation
```bash
python pc_simulator/main.py --current 50.0 --duration 60.0 --soc 20.0
```
- Charge at 50A for 60 seconds
- Start from 20% SOC

#### Example 2: Discharge Simulation
```bash
python pc_simulator/main.py --current -100.0 --duration 120.0 --soc 80.0
```
- Discharge at 100A for 120 seconds
- Start from 80% SOC

#### Example 3: XBB Protocol with UART
```bash
python pc_simulator/main.py --port COM3 --protocol xbb --rate 10.0 --current 50.0
```
- Transmit via COM3 at 10 Hz
- Use XBB protocol
- Charge at 50A

#### Example 4: MCU Protocol
```bash
python pc_simulator/main.py --port COM3 --protocol mcu --rate 1.0 --current 25.0
```
- Use MCU-compatible protocol
- Transmit at 1 Hz

#### Example 5: Standalone Pack Simulation (Analysis Mode)
```bash
python pc_simulator/plant/run_cell_simulation.py --mode discharge --current 1.0 --duration 60
```
- Run pack simulation without AFE/UART
- Saves CSV data to `pc_simulator/plant/output/`
- Useful for analysis and testing

**Additional options for `run_cell_simulation.py`:**
```bash
# Charge simulation
python pc_simulator/plant/run_cell_simulation.py --mode charge --current 2.0 --duration 120

# Discharge until target SOC
python pc_simulator/plant/run_cell_simulation.py --mode discharge --current 1.0 --target-soc 50

# With plot generation
python pc_simulator/plant/run_cell_simulation.py --mode discharge --current 1.0 --duration 60 --plot

# Custom initial SOC
python pc_simulator/plant/run_cell_simulation.py --mode discharge --current 1.0 --duration 60 --initial-soc 80
```

---

## 📁 Project Structure

```
Battery-Pack-Simulator/
├── pc_simulator/
│   ├── main.py                 # Main integration script
│   ├── afe/
│   │   └── wrapper.py          # AFE measurement simulation
│   ├── communication/
│   │   ├── protocol_xbb.py    # XBB protocol encoder
│   │   ├── protocol_mcu.py    # MCU protocol encoder
│   │   ├── protocol.py         # Legacy protocol
│   │   ├── uart_tx_xbb.py     # XBB UART transmitter
│   │   ├── uart_tx_mcu.py     # MCU UART transmitter
│   │   └── uart_tx.py          # Base UART transmitter
│   └── plant/
│       ├── cell_model.py          # LiFePO₄ cell ECM model
│       ├── pack_model.py           # 16S pack model
│       ├── current_profile.py      # Current profile generator
│       └── run_cell_simulation.py # Standalone pack simulation (CSV/plot output)
├── scenarios/
│   ├── charge_discharge_cycle.yaml
│   ├── charge_profile.yaml
│   ├── discharge_profile.yaml
│   ├── mixed_profile.yaml
│   └── pulse_profile.yaml
├── tests_legacy/
│   ├── test_cell_model.py
│   ├── test_pack_model.py
│   ├── test_afe_wrapper.py
│   ├── test_current_profile.py
│   └── test_uart_tx.py
├── requirements.txt
└── README.md
```

---

## 🔬 Testing

Run unit tests:

```bash
# Run all tests
pytest tests_legacy/

# Run specific test file
pytest tests_legacy/test_cell_model.py

# Run with verbose output
pytest tests_legacy/ -v
```

Run integration test:

```bash
python pc_simulator/test_integration.py
```

---

## 📡 Protocol Details

### XBB Protocol

**Frame Format:**
```
[0xA5] [0x33] [SubIndex: 0x0000] [DataLen: 80] [Data: 80 bytes] [0xB5] [CRC8]
```

**Data Structure (20 int32 values, big-endian, 80 bytes):**
- `pack_current_A`: 4 bytes (int32, milli_A)
- `pack_voltage_V`: 4 bytes (int32, milli_V)
- `temp_cell_C`: 4 bytes (int32, milli_degC)
- `temp_pcb_C`: 4 bytes (int32, milli_degC)
- `cell_1_V` through `cell_16_V`: 64 bytes (16 × int32, milli_V)

**Total Frame Size:** 88 bytes

**CRC8:** Calculated over all bytes from 0xA5 through 0xB5 (excluding CRC8 byte)

### MCU Protocol

MCU-compatible protocol with support for:
- Multiple strings/modules
- Configurable cell/temperature sensor counts
- Legacy format compatibility

---

## ⚙️ Configuration

### Battery Pack Parameters

Default configuration in `pack_model.py`:
- **Cell Capacity:** 100 Ah per cell
- **Initial SOC:** 50%
- **Ambient Temperature:** 25°C
- **Cell Variations:**
  - Capacity mismatch: σ = 0.4%
  - SOC variation: σ = 0.25%
  - Resistance variation: ±2.5%

### AFE Parameters

Default configuration in `afe/wrapper.py`:
- **Voltage Noise:** σ = 2.0 mV
- **Temperature Noise:** σ = 0.5°C
- **Current Noise:** σ = 50 mA
- **Calibration Errors:**
  - Voltage: ±0.1% gain, ±5mV offset
  - Temperature: ±1°C offset
  - Current: ±0.2% gain, ±10mA offset

### Current Profiles

YAML-based profiles in `scenarios/` directory:

```yaml
name: "Charge_Discharge_Cycle"
description: "Complete charge-discharge cycle"
duration_sec: 7200
segments:
  - time_range: [0, 3600]
    current_a: 50
    description: "Charge at 0.5C"
  - time_range: [3600, 7200]
    current_a: -100
    description: "Discharge at 1C"
```

---

## 🛠️ Development

### Code Structure

- **Plant Model:** Battery physics simulation (`pc_simulator/plant/`)
- **AFE Wrapper:** Measurement simulation (`pc_simulator/afe/`)
- **Communication:** UART protocols (`pc_simulator/communication/`)
- **Main Integration:** Orchestrates components (`pc_simulator/main.py`)

### Key Classes

- `LiFePO4Cell`: Cell ECM model with 2RC network
- `BatteryPack16S`: 16-cell series pack model
- `AFEWrapper`: AFE measurement simulation
- `XBBFrameEncoder`: XBB protocol encoder
- `CurrentProfile`: Current profile generator

---

## 📊 Model Details

### Cell Model (LiFePO₄)

**ECM Structure:**
```
OCV(SOC, T, direction) - R0(SOC, T) - [R1 || C1] - [R2 || C2] - Terminal
```

**Parameters:**
- R0: 0.5 mΩ (at 50% SOC, 25°C)
- R1: 1 mΩ, C1: 2000 F (fast transient, τ=2s)
- R2: 0.5 mΩ, C2: 10000 F (slow transient, τ=5s)

**OCV-SOC Characteristics:**
- Flat plateau: ~3.26-3.30V (20-80% SOC)
- Steep ends: 2.86V (0%) to 3.47V (100%)
- Hysteresis: 5-15mV difference between charge/discharge

**Aging Models:**
- **Cycle Aging:** Capacity fade and resistance increase with cycles
- **Calendar Aging:** Arrhenius-based time-dependent capacity fade

---

## 🐛 Troubleshooting

### Serial Port Issues

**Windows:**
- Check COM port in Device Manager
- Ensure port is not in use by another application
- Try different baud rates (115200, 921600)

**Linux/Mac:**
- Check permissions: `sudo chmod 666 /dev/ttyUSB0`
- List available ports: `ls /dev/tty*`
- Check if port exists: `dmesg | grep tty`

### Import Errors

If you encounter import errors:
```bash
# Ensure you're in the project root directory
cd Battery-Pack-Simulator

# Verify Python path
python -c "import sys; print(sys.path)"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## 📝 License

This project is part of XBattery Energy's BMS development tools.

---

## 👥 Contributing

For issues, feature requests, or contributions, please contact the XBattery Energy development team.

**Repository:** [XBattery-Energy/Battery-Pack-Simulator](https://github.com/XBattery-Energy/Battery-Pack-Simulator)

---

## 📚 Additional Resources

- **Codebase Analysis:** See `CODEBASE_ANALYSIS.md` for detailed technical analysis
- **Test Files:** See `tests_legacy/` directory for usage examples
- **Scenarios:** See `scenarios/` directory for YAML profile examples

---

## 🔗 Related Projects

- XBattery Energy BMS Firmware
- XBattery Energy Hardware-in-the-Loop (HIL) Test System

---

**Last Updated:** 2025-12-31

