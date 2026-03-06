"""
OGPR File Parser

Handles reading and parsing of .ogpr format GPR data files.
"""

import json
import struct
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional


class OGPRParser:
    """
    Parser for OGPR format Ground Penetrating Radar data.
    
    OGPR file structure:
    - Header: signature (4 bytes) + UUID (32 bytes) + offset (8 bytes)
    - JSON descriptor with metadata
    - Binary data blocks (radar volume + geolocations)
    """
    
    def __init__(self, filepath: str):
        """
        Initialize parser with OGPR file path.
        
        Args:
            filepath: Path to .ogpr file
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        self.descriptor = None
        self.radar_data = None
        self.geolocations = None
        self._file_size = self.filepath.stat().st_size
        
    def parse_header(self) -> Dict:
        """
        Parse OGPR file header and JSON descriptor.
        
        Returns:
            Dictionary containing parsed metadata
        """
        with open(self.filepath, 'rb') as f:
            # Read signature (should be 'ogpr')
            signature = f.read(4).decode('ascii')
            if signature != 'ogpr':
                raise ValueError(f"Invalid OGPR signature: {signature}")
            
            # Read UUID (32 hex characters)
            uuid = f.read(32).decode('ascii')
            
            # Read data offset (8 hex characters)
            offset_hex = f.read(8).decode('ascii')
            data_offset = int(offset_hex, 16)
            
            # Read JSON descriptor (from current position to data_offset)
            json_size = data_offset - f.tell()
            json_data = f.read(json_size).decode('utf-8').strip()
            
            # Parse JSON
            descriptor = json.loads(json_data)
            
        self.descriptor = descriptor
        return descriptor
    
    def get_metadata(self) -> Dict:
        """
        Get main metadata from descriptor.
        
        Returns:
            Dictionary with key parameters
        """
        if self.descriptor is None:
            self.parse_header()
        
        main = self.descriptor['mainDescriptor']
        radar_block = self.descriptor['dataBlockDescriptors'][0]
        
        return {
            'samples_count': main['samplesCount'],
            'channels_count': main['channelsCount'],
            'slices_count': main['slicesCount'],
            'sampling_step_m': radar_block['radar']['samplingStep_m'],
            'sampling_time_ns': radar_block['radar']['samplingTime_ns'],
            'frequency_mhz': radar_block['radar'].get('fequency_MHz', 
                                                       radar_block['radar'].get('frequency_MHz', 0)),
            'polarization': radar_block['radar']['polarization'],
            'swath_name': main.get('metadata', {}).get('swathName', 'Unknown'),
        }
    
    def load_radar_volume(self, lazy: bool = False) -> np.ndarray:
        """
        Load radar data volume from file.
        
        Args:
            lazy: If True, return memmap instead of loading into RAM
        
        Returns:
            3D numpy array of shape (samples, channels, slices)
        """
        if self.descriptor is None:
            self.parse_header()
        
        # Get radar data block info
        radar_block = self.descriptor['dataBlockDescriptors'][0]
        byte_offset = radar_block['byteOffset']
        byte_size = radar_block['byteSize']
        
        # Get dimensions
        main = self.descriptor['mainDescriptor']
        samples = main['samplesCount']
        channels = main['channelsCount']
        slices = main['slicesCount']
        
        # Calculate expected size
        expected_size = samples * channels * slices * 4  # float32 = 4 bytes
        if byte_size != expected_size:
            print(f"Warning: Expected {expected_size} bytes, got {byte_size}")
        
        if lazy:
            # Memory-mapped array (doesn't load into RAM)
            data = np.memmap(
                self.filepath,
                dtype=np.float32,
                mode='r',
                offset=byte_offset,
                shape=(samples, channels, slices)
            )
        else:
            # Load into RAM
            with open(self.filepath, 'rb') as f:
                f.seek(byte_offset)
                data = np.frombuffer(
                    f.read(byte_size),
                    dtype=np.float32
                ).reshape((samples, channels, slices))
        
        self.radar_data = data
        return data
    
    def load_geolocations(self) -> Optional[np.ndarray]:
        """
        Load sample geographic locations if available.
        
        Returns:
            Array of geographic coordinates or None if not available
        """
        if self.descriptor is None:
            self.parse_header()
        
        # Check if geolocations block exists
        geo_blocks = [
            block for block in self.descriptor['dataBlockDescriptors']
            if block['type'] == 'Sample Geolocations'
        ]
        
        if not geo_blocks:
            return None
        
        geo_block = geo_blocks[0]
        byte_offset = geo_block['byteOffset']
        byte_size = geo_block['byteSize']
        
        # Geolocations are typically stored as double precision (x, y, z)
        # Shape: (slices, channels, samples, 3)
        with open(self.filepath, 'rb') as f:
            f.seek(byte_offset)
            data = np.frombuffer(
                f.read(byte_size),
                dtype=np.float64
            )
        
        # Reshape based on dimensions
        main = self.descriptor['mainDescriptor']
        samples = main['samplesCount']
        channels = main['channelsCount']
        slices = main['slicesCount']
        
        # Try to reshape (format may vary)
        try:
            geolocations = data.reshape((slices, channels, samples, 3))
        except ValueError:
            # Alternative format: (slices, samples, 3)
            try:
                geolocations = data.reshape((slices, samples, 3))
            except ValueError:
                print(f"Warning: Could not reshape geolocations. Size: {data.size}")
                return data
        
        self.geolocations = geolocations
        return geolocations
    
    def load_data(self, lazy: bool = False) -> Dict:
        """
        Load all data from OGPR file.
        
        Args:
            lazy: Use memory mapping for large files
        
        Returns:
            Dictionary with radar data, metadata, and geolocations
        """
        metadata = self.get_metadata()
        radar_volume = self.load_radar_volume(lazy=lazy)
        geolocations = self.load_geolocations()
        
        return {
            'radar_volume': radar_volume,
            'metadata': metadata,
            'geolocations': geolocations,
            'descriptor': self.descriptor
        }
    
    def get_bscan(self, channel: int = 0, slice_idx: Optional[int] = None) -> np.ndarray:
        """
        Extract B-scan (2D radargram) from data.
        
        Args:
            channel: Channel index to extract
            slice_idx: Specific slice index, or None for all slices
        
        Returns:
            2D array of shape (samples, slices) or (samples, 1)
        """
        if self.radar_data is None:
            self.load_radar_volume()
        
        if slice_idx is not None:
            return self.radar_data[:, channel, slice_idx:slice_idx+1]
        else:
            return self.radar_data[:, channel, :]
    
    def __repr__(self):
        if self.descriptor is None:
            return f"OGPRParser('{self.filepath}', not loaded)"
        
        meta = self.get_metadata()
        return (
            f"OGPRParser('{self.filepath}')\n"
            f"  Samples: {meta['samples_count']}\n"
            f"  Channels: {meta['channels_count']}\n"
            f"  Slices: {meta['slices_count']}\n"
            f"  Frequency: {meta['frequency_mhz']} MHz\n"
            f"  File size: {self._file_size / 1024 / 1024:.1f} MB"
        )