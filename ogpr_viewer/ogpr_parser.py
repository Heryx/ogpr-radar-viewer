"""
OGPR File Parser

Handles reading and parsing of .ogpr format GPR data files.
Supports OGPR format versions 1.x and 2.x.
Supports IDS Stream UP (float32) and IDS Stream DP (int16) antenna types.

Header format (fields separated by newlines):
  Line 0: "ogpr"
  Line 1: UUID  (32 hex chars)
  Line 2: data offset  (8 hex chars)
  Remaining bytes up to data_offset: JSON descriptor
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

    @staticmethod
    def _read_line(f) -> str:
        """Read one newline-terminated text line from a binary file handle."""
        buf = b''
        while True:
            ch = f.read(1)
            if ch in (b'\n', b''):
                break
            buf += ch
        return buf.decode('ascii').strip()

    def parse_header(self) -> Dict:
        """
        Parse OGPR file header and JSON descriptor.

        The header consists of three newline-separated text lines:
          1. signature  -> must be 'ogpr'
          2. UUID       -> 32 hex characters
          3. data offset -> 8 hex characters (big-endian uint32 as hex string)

        Everything from after those three lines up to byte <data_offset>
        is the UTF-8 JSON descriptor.
        """
        with open(self.filepath, 'rb') as f:
            # --- line 1: signature ---
            signature = self._read_line(f)
            if signature != 'ogpr':
                raise ValueError(
                    f"Invalid OGPR signature: '{signature}' "
                    f"(expected 'ogpr')"
                )

            # --- line 2: UUID ---
            uuid = self._read_line(f)  # noqa: F841  (stored for future use)

            # --- line 3: data offset (hex) ---
            offset_hex = self._read_line(f).strip()
            try:
                data_offset = int(offset_hex, 16)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot parse data offset '{offset_hex}' as hex: {exc}"
                ) from exc

            # --- JSON descriptor ---
            current_pos = f.tell()
            json_size   = data_offset - current_pos
            if json_size <= 0:
                raise ValueError(
                    f"data_offset ({data_offset}) <= current position "
                    f"({current_pos}); file may be corrupted."
                )
            json_raw = f.read(json_size).decode('utf-8').strip()

        self.descriptor = json.loads(json_raw)
        return self.descriptor

    # ------------------------------------------------------------------
    # dtype detection
    # ------------------------------------------------------------------

    def _detect_dtype(self) -> type:
        for block in self.descriptor['dataBlockDescriptors']:
            if block['type'] == 'Radar Volume':
                vtype = block.get('valueType', 'float').lower()
                return self._DTYPE_MAP.get(vtype, np.float32)
        return np.float32

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self) -> Dict:
        if self.descriptor is None:
            self.parse_header()

        main = self.descriptor['mainDescriptor']
        radar_block = next(
            b for b in self.descriptor['dataBlockDescriptors']
            if b['type'] == 'Radar Volume'
        )
        r     = radar_block['radar']
        dtype = self._detect_dtype()

        return {
            'samples_count':    main['samplesCount'],
            'channels_count':   main['channelsCount'],
            'slices_count':     main['slicesCount'],
            'sampling_step_m':  r['samplingStep_m'],
            'sampling_time_ns': r['samplingTime_ns'],
            'frequency_mhz':    r.get('fequency_MHz', r.get('frequency_MHz', 0)),
            'polarization':     r['polarization'],
            'swath_name':       main.get('metadata', {}).get('swathName', 'Unknown'),
            'array_id':         main.get('metadata', {}).get('arrayId', 0),
            'version':          self.descriptor.get('version', {'major': 1, 'minor': 0}),
            'dtype':            dtype,
            'dtype_name':       'float32' if dtype == np.float32 else 'int16',
        }

    # ------------------------------------------------------------------
    # Radar Volume
    # ------------------------------------------------------------------

    def load_radar_volume(self, lazy: bool = False) -> np.ndarray:
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
                f"{dtype.__name__} volume, got {byte_size}."
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

        # Always process as float32
        if dtype == np.int16:
            data = data.astype(np.float32)

        self.radar_data = data
        return data

    # ------------------------------------------------------------------
    # Geolocations
    # ------------------------------------------------------------------

    def load_geolocations(self) -> Optional[np.ndarray]:
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
            if data.size == int(np.prod(shape)):
                self.geolocations = data.reshape(shape)
                return self.geolocations

        print(f"[OGPR] Warning: geolocations size {data.size} doesn't match known shapes.")
        self.geolocations = data
        return data

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def load_data(self, lazy: bool = False) -> Dict:
        metadata     = self.get_metadata()
        radar_volume = self.load_radar_volume(lazy=lazy)
        geolocations = self.load_geolocations()

        return {
            'radar_volume': radar_volume,
            'metadata':     metadata,
            'geolocations': geolocations,
            'descriptor':   self.descriptor,
            'filepath':     str(self.filepath),
        }

    def get_bscan(self, channel: int = 0, slice_start: int = 0,
                  slice_end: Optional[int] = None) -> np.ndarray:
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
