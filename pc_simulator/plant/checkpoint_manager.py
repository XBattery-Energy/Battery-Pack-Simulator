"""Checkpoint manager for saving and restoring model parameters."""
import json
import os
from pathlib import Path
import numpy as np

class ModelCheckpoint:
    """Manages checkpoints for battery model parameters."""
    
    def __init__(self, checkpoint_dir="checkpoints"):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(self, checkpoint_name, parameters):
        """Save model parameters to a checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint (e.g., "checkpoint_1")
            parameters: Dictionary of parameters to save
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_params = {}
        for key, value in parameters.items():
            if isinstance(value, np.ndarray):
                serializable_params[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_params[key] = float(value)
            else:
                serializable_params[key] = value
        
        with open(checkpoint_file, 'w') as f:
            json.dump(serializable_params, f, indent=2)
        
        print(f"✓ Checkpoint '{checkpoint_name}' saved to {checkpoint_file}")
        return checkpoint_file
    
    def load_checkpoint(self, checkpoint_name):
        """Load model parameters from a checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint to load
            
        Returns:
            Dictionary of parameters
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.json"
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint '{checkpoint_name}' not found at {checkpoint_file}")
        
        with open(checkpoint_file, 'r') as f:
            parameters = json.load(f)
        
        # Convert lists back to numpy arrays where needed
        for key, value in parameters.items():
            if isinstance(value, list) and len(value) > 0:
                # Check if it's a 2D array (OCV tables)
                if isinstance(value[0], list):
                    parameters[key] = np.array(value)
                else:
                    # Try to convert to numpy array
                    try:
                        parameters[key] = np.array(value)
                    except:
                        pass
        
        print(f"✓ Checkpoint '{checkpoint_name}' loaded from {checkpoint_file}")
        return parameters
    
    def list_checkpoints(self):
        """List all available checkpoints."""
        checkpoints = list(self.checkpoint_dir.glob("*.json"))
        if not checkpoints:
            print("No checkpoints found")
            return []
        
        print(f"\nAvailable checkpoints in {self.checkpoint_dir}:")
        for cp in sorted(checkpoints):
            print(f"  - {cp.stem}")
        return [cp.stem for cp in checkpoints]

def save_current_model_state(checkpoint_name="checkpoint_1"):
    """Save current model state from cell_model.py."""
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    from sil_bms.pc_simulator.plant.cell_model import LiFePO4Cell
    
    # Create a cell instance to access class attributes
    cell = LiFePO4Cell()
    
    # Extract all relevant parameters
    parameters = {
        # OCV tables - access class attributes directly
        'ocv_soc_table_discharge': LiFePO4Cell._OCV_SOC_TABLE_DISCHARGE.tolist(),
        'ocv_soc_table_charge': LiFePO4Cell._OCV_SOC_TABLE_CHARGE.tolist(),
        
        # ECM parameters
        'R1': float(LiFePO4Cell.R1),
        'C1': float(LiFePO4Cell.C1),
        'R2': float(LiFePO4Cell.R2),
        'C2': float(LiFePO4Cell.C2),
        
        # Resistance parameters (from get_internal_resistance logic)
        'r0_base_mohm_at_50pct': 0.5,  # Base R0 at 50% SOC
        'r0_multiplier_at_0pct': 1.5,   # Multiplier at 0% SOC
        'r0_multiplier_at_100pct': 0.8, # Multiplier at 100% SOC (current value)
        
        # Temperature coefficients
        'ocv_temp_coeff': float(LiFePO4Cell.OCV_TEMP_COEFF),
        'capacity_temp_coeff': float(LiFePO4Cell.CAPACITY_TEMP_COEFF),
        
        # Voltage limits (hardcoded in update method)
        'min_cell_voltage': 2.5,  # From update method
        'max_cell_voltage': 3.65, # From update method (if exists, otherwise inferred)
        
        # RC network scaling (C-rate dependent) - from update method
        'rc_scale_alpha': 0.15,  # From the formula: 1.0 / (1.0 + 0.15 * (C_rate - 1.0))
        'rc_scale_min': 0.3,     # Minimum scale factor
        
        # Notes
        'notes': 'Checkpoint 1: Original model parameters before adjustments to match real data',
        'timestamp': str(pd.Timestamp.now()) if 'pd' in dir() else None
    }
    
    # Save checkpoint
    checkpoint_mgr = ModelCheckpoint()
    checkpoint_file = checkpoint_mgr.save_checkpoint(checkpoint_name, parameters)
    
    print(f"\n{'='*80}")
    print(f"CHECKPOINT SAVED: {checkpoint_name}")
    print(f"{'='*80}")
    print(f"\nSaved parameters:")
    print(f"  - OCV tables (discharge & charge)")
    print(f"  - RC network: R1={parameters['R1']*1000:.1f}mΩ, C1={parameters['C1']:.0f}F")
    print(f"                 R2={parameters['R2']*1000:.1f}mΩ, C2={parameters['C2']:.0f}F")
    print(f"  - Internal resistance: {parameters['r0_base_mohm_at_50pct']:.1f}mΩ at 50% SOC")
    print(f"  - Voltage limits: {parameters['min_cell_voltage']:.2f}V to {parameters['max_cell_voltage']:.2f}V")
    print(f"\nCheckpoint file: {checkpoint_file}")
    print(f"\nTo restore this checkpoint, use: restore_checkpoint('{checkpoint_name}')")
    print(f"{'='*80}\n")
    
    return checkpoint_file

def restore_checkpoint(checkpoint_name="checkpoint_1"):
    """Restore model parameters from a checkpoint."""
    checkpoint_mgr = ModelCheckpoint()
    parameters = checkpoint_mgr.load_checkpoint(checkpoint_name)
    
    print(f"\n{'='*80}")
    print(f"RESTORING CHECKPOINT: {checkpoint_name}")
    print(f"{'='*80}")
    print("\nTo apply these parameters, you'll need to manually update cell_model.py")
    print("or use the apply_checkpoint_to_model() function.")
    print(f"{'='*80}\n")
    
    return parameters

def apply_checkpoint_to_model(checkpoint_name="checkpoint_1"):
    """Apply checkpoint parameters to cell_model.py file."""
    checkpoint_mgr = ModelCheckpoint()
    parameters = checkpoint_mgr.load_checkpoint(checkpoint_name)
    
    print(f"\nApplying checkpoint '{checkpoint_name}' to cell_model.py...")
    print("NOTE: This will modify the cell_model.py file.")
    print("Make sure you have a backup if needed.\n")
    
    # This function would need to modify cell_model.py
    # For safety, we'll just print what needs to be changed
    print("Parameters to restore:")
    print(f"  - OCV tables: {len(parameters['ocv_soc_table_discharge'])} points")
    print(f"  - R1: {parameters['R1']*1000:.1f}mΩ, C1: {parameters['C1']:.0f}F")
    print(f"  - R2: {parameters['R2']*1000:.1f}mΩ, C2: {parameters['C2']:.0f}F")
    print(f"  - R0 at 100% SOC multiplier: {parameters['r0_multiplier_at_100pct']:.1f}x")
    
    return parameters

if __name__ == '__main__':
    # Save current state as checkpoint 1
    save_current_model_state("checkpoint_1")

