"""
OGPR File Parser

Handles reading and parsing of .ogpr format GPR data files.
Supports OGPR format versions 1.x and 2.x.
Supports IDS Stream UP (float32) and IDS Stream DP (int16) antenna types.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional


class OGPRParser:
    """
    Parser for OGPR format Ground Penetrating Radar data.

    Automatically detects dtype from JSON descriptor:
      - 'valueType': 'float'  => np.float32  (IDS Stream UP)
      - 'valueType': 'int'    => np.int16    (IDS Stream DP)
      - absent (v1)           => np.float32  (default)
    """

    # dtype map from 'valueType' field in JSON
    _DTYPE_MAP = {
        'float': np.float32,
        'int':   np.int16,
    }
    _DTYPE_BYTES = {
        np.float32: 4,
        np.int16:   2,
    }

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        self.descriptor: Optional[Dict] = None
        self.radar_data: Optional[np.ndarray] = None
        self.geolocations: Optional[np.ndarray] = None
        self._file_size = self.filepath.stat().st_size

    # ------------------------------------------------------------------
    # Header / JSON
    # ------------------------------------------------------------------

    def parse_header(self) -> Dict:
        """Parse OGPR file header and JSON descriptor."""
        with open(self.filepath, 'rb') as f:
            signature = f.read(4).decode('ascii')
            if signature != 'ogpr':
                raise ValueError(f"Invalid OGPR signature: '{signature}'")

            uuid = f.read(32).decode('ascii')  # noqa: F841

            offset_hex = f.read(8).decode('ascii')
            data_offset = int(offset_hex, 16)

            json_size = data_offset - f.tell()
            json_raw = f.read(json_size).decode('utf-8').strip()

        self.descriptor = json.loads(json_raw)
        return self.descriptor

    # ------------------------------------------------------------------
    # dtype detection
    # ------------------------------------------------------------------

    def _detect_dtype(self) -> type:
        """
        Detect numpy dtype for the Radar Volume block.

        Priority:
          1. 'valueType' field inside the Radar Volume block descriptor
          2. Default: float32
        """
        for block in self.descriptor['dataBlockDescriptors']:
            if block['type'] == 'Radar Volume':
                vtype = block.get('valueType', 'float').lower()
                return self._DTYPE_MAP.get(vtype, np.float32)
        return np.float32

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self) -> Dict:
        """Return a flat metadata dict with the most useful fields."""
        if self.descriptor is None:
            self.parse_header()

        main = self.descriptor['mainDescriptor']
        radar_block = next(
            b for b in self.descriptor['dataBlockDescriptors']
            if b['type'] == 'Radar Volume'
        )
        r = radar_block['radar']
        dtype = self._detect_dtype()

        return {
            'samples_count':   main['samplesCount'],
            'channels_count':  main['channelsCount'],
            'slices_count':    main['slicesCount'],
            'sampling_step_m': r['samplingStep_m'],
            'sampling_time_ns': r['samplingTime_ns'],
            'frequency_mhz':   r.get('fequency_MHz', r.get('frequency_MHz', 0)),
            'polarization':    r['polarization'],
            'swath_name':      main.get('metadata', {}).get('swathName', 'Unknown'),
            'array_id':        main.get('metadata', {}).get('arrayId', 0),
            'version':         self.descriptor.get('version', {'major': 1, 'minor': 0}),
            'dtype':           dtype,
            'dtype_name':      'float32' if dtype == np.float32 else 'int16',
        }

    # ------------------------------------------------------------------
    # Radar Volume
    # ------------------------------------------------------------------

    def load_radar_volume(self, lazy: bool = False) -> np.ndarray:
        """
        Load the Radar Volume binary block.

        Args:
            lazy: If True, return a memory-mapped array (useful for large files).

        Returns:
            3-D ndarray  shape=(samples, channels, slices), dtype=float32 or int16.
        """
        if self.descriptor is None:
            self.parse_header()

        radar_block = next(
            b for b in self.descriptor['dataBlockDescriptors']
            if b['type'] == 'Radar Volume'
        )
        byte_offset = radar_block['byteOffset']
        byte_size   = radar_block['byteSize']

        main     = self.descriptor['mainDescriptor']
        samples  = main['samplesCount']
        channels = main['channelsCount']
        slices   = main['slicesCount']
        dtype    = self._detect_dtype()
        itemsize = self._DTYPE_BYTES[dtype]

        expected = samples * channels * slices * itemsize
        if byte_size != expected:
            print(
                f"[OGPR] Warning: expected {expected} bytes for "
                f"{dtype.__name__} radar volume, got {byte_size}. "
                f"Will attempt reshape anyway."
            )

        shape = (samples, channels, slices)

        if lazy:
            data = np.memmap(
                self.filepath, dtype=dtype, mode='r',
                offset=byte_offset, shape=shape
            )
        else:
            with open(self.filepath, 'rb') as f:
                f.seek(byte_offset)
                raw = f.read(byte_size)
            data = np.frombuffer(raw, dtype=dtype).reshape(shape)

        # Always work in float32 internally for processing
        if dtype == np.int16:
            data = data.astype(np.float32)

        self.radar_data = data
        return data

    # ------------------------------------------------------------------
    # Geolocations
    # ------------------------------------------------------------------

    def load_geolocations(self) -> Optional[np.ndarray]:
        """Load sample geographic locations (float64, XYZ in EPSG units)."""
        if self.descriptor is None:
            self.parse_header()

        geo_blocks = [
            b for b in self.descriptor['dataBlockDescriptors']
            if b['type'] == 'Sample Geolocations'
        ]
        if not geo_blocks:
            return None

        geo_block   = geo_blocks[0]
        byte_offset = geo_block['byteOffset']
        byte_size   = geo_block['byteSize']

        with open(self.filepath, 'rb') as f:
            f.seek(byte_offset)
            raw = f.read(byte_size)

        data = np.frombuffer(raw, dtype=np.float64)

        main     = self.descriptor['mainDescriptor']
        samples  = main['samplesCount']
        channels = main['channelsCount']
        slices   = main['slicesCount']

        for shape in [
            (slices, channels, samples, 3),
            (slices, samples, 3),
            (slices, 3),
        ]:
            expected_n = 1
            for d in shape:
                expected_n *= d
            if data.size == expected_n:
                self.geolocations = data.reshape(shape)
                return self.geolocations

        print(f"[OGPR] Warning: geolocations size {data.size} doesn't match known shapes.")
        self.geolocations = data
        return data

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def load_data(self, lazy: bool = False) -> Dict:
        """Load all data (radar volume + geolocations + metadata)."""
        metadata      = self.get_metadata()
        radar_volume  = self.load_radar_volume(lazy=lazy)
        geolocations  = self.load_geolocations()

        return {
            'radar_volume': radar_volume,
            'metadata':     metadata,
            'geolocations': geolocations,
            'descriptor':   self.descriptor,
            'filepath':     str(self.filepath),
        }

    def get_bscan(self, channel: int = 0, slice_start: int = 0,
                  slice_end: Optional[int] = None) -> np.ndarray:
        """Extract a 2-D B-scan (samples × slices) for the given channel."""
        if self.radar_data is None:
            self.load_radar_volume()
        s = slice_end if slice_end is not None else self.radar_data.shape[2]
        return self.radar_data[:, channel, slice_start:s]

    def __repr__(self):
        if self.descriptor is None:
            return f"OGPRParser('{self.filepath}', not loaded)"
        m = self.get_metadata()
        v = m['version']
        return (
            f"OGPRParser('{self.filepath.name}')\n"
            f"  Version  : {v['major']}.{v['minor']}\n"
            f"  Dtype    : {m['dtype_name']}\n"
            f"  Samples  : {m['samples_count']}\n"
            f"  Channels : {m['channels_count']}\n"
            f"  Slices   : {m['slices_count']}\n"
            f"  Frequency: {m['frequency_mhz']} MHz\n"
            f"  Swath    : {m['swath_name']}\n"
            f"  File size: {self._file_size / 1024 / 1024:.1f} MB"
        )
