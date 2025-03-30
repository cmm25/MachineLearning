import os
from pathlib import Path

class PathConfig:    
    def __init__(self):
        # Get the project root directory
        self.root_dir = Path(__file__).parent.parent.absolute()
        
        # Model directories
        self.models_dir = os.path.join(self.root_dir, "models")
        self.east_model_path = os.path.join(self.models_dir, "frozen_east_text_detection.pb")
        
        # Data directories
        self.data_dir = os.path.join(self.root_dir, "data")
        self.input_dir = os.path.join(self.data_dir, "input")
        self.output_dir = os.path.join(self.data_dir, "output")
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create directories if they don't exist"""
        for directory in [self.models_dir, self.data_dir, self.input_dir, self.output_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def get_input_path(self, filename):
        """Get absolute path to input file"""
        return os.path.join(self.input_dir, filename)
    
    def get_output_path(self, filename):
        """Get absolute path to output file"""
        return os.path.join(self.output_dir, filename)
    
    def get_model_path(self, model_name):
        """Get absolute path to model file"""
        return os.path.join(self.models_dir, model_name)