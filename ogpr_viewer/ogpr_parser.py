"""
OGPR File Parser

Handles reading and parsing of .ogpr format GPR data files.
Supports OGPR format versions 1.x and 2.x.
Supports IDS Stream UP (float32) and IDS Stream DP (int16) antenna types.

Header format (newline-separated text lines):
  Line 1: "ogpr"
  Line 2: UUID  (32 hex chars)
  Line 3: data offset  (8 hex chars, used only as cross-check)
  Remaining text: JSON descriptor  <- read by brace-balancing, NOT by byte offset
  After JSON: binary data blocks

On-disk memory layout (IDS format):
  Data is written trace-by-trace: for each slice, all channels, all samples.
  Binary order: (slices, channels, samples)  <- C-order on disk
  After reshape + transpose -> (samples, channels, slices)  <- viewer convention
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional

LOG = logging.getLogger('ogpr_viewer')


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
    # Low-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_line(f) -> str:
        """Read one newline-terminated ASCII line from a binary file handle."""
        buf = b''
        while True:
            ch = f.read(1)
            if ch in (b'\n', b''):
                break
            buf += ch
        return buf.decode('ascii').strip()

    @staticmethod
    def _read_json_by_braces(f) -> str:
        """
        Read exactly one complete JSON object from the current file position
        by counting opening/closing braces.
        """
        buf         = []
        depth       = 0
        in_string   = False
        escape_next = False
        started     = False

        while True:
            byte = f.read(1)
            if not byte:
                break
            ch = byte.decode('latin-1')

            if escape_next:
                buf.append(ch)
                escape_next = False
                continue

            if in_string:
                buf.append(ch)
                if ch == '\\':
                    escape_next = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                buf.append(ch)
            elif ch == '{':
                depth += 1
                started = True
                buf.append(ch)
            elif ch == '}':
                depth -= 1
                buf.append(ch)
                if started and depth == 0:
                    break
            else:
                if started:
                    buf.append(ch)

        return ''.join(buf)

    # ------------------------------------------------------------------
    # Header / JSON
    # ------------------------------------------------------------------

    def parse_header(self) -> Dict:
        with open(self.filepath, 'rb') as f:
            signature = self._read_line(f)
            if signature != 'ogpr':
                raise ValueError(
                    f"Invalid OGPR signature: '{signature}' (expected 'ogpr')"
                )
            _uuid      = self._read_line(f)
            offset_hex = self._read_line(f).strip()
            try:
                self._data_offset = int(offset_hex, 16)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot parse data offset '{offset_hex}' as hex: {exc}"
                ) from exc
            json_text = self._read_json_by_braces(f)

        if not json_text:
            raise ValueError("No JSON descriptor found in file.")
        try:
            self.descriptor = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON descriptor: {exc}") from exc

        return self.descriptor

    # ------------------------------------------------------------------
    # dtype detection
    # ------------------------------------------------------------------

    def _detect_dtype(self) -> type:
        for block in self.descriptor['dataBlockDescriptors']:
            if block['type'] == 'Radar Volume':
                vtype = block.get('valueType', 'float').lower()
                dtype = self._DTYPE_MAP.get(vtype, np.float32)
                LOG.debug(f'dtype detection: valueType="{vtype}" -> {dtype.__name__}')
                return dtype
        LOG.warning('No Radar Volume block found in descriptor, defaulting to float32')
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
            'dtype':            dtype.__name__,
            'dtype_name':       dtype.__name__,
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

        LOG.debug(
            f'Radar Volume block: byteOffset={byte_offset}  byteSize={byte_size}  '
            f'expected={expected}  dtype={dtype.__name__}  '
            f'shape=({samples},{channels},{slices})'
        )

        if byte_size != expected:
            other_dtype    = np.int16   if dtype == np.float32 else np.float32
            other_bytes    = self._DTYPE_BYTES[other_dtype]
            other_expected = samples * channels * slices * other_bytes

            if byte_size == other_expected:
                LOG.warning(
                    f'byte_size mismatch: descriptor says {dtype.__name__} '
                    f'(expected {expected} bytes) but byte_size={byte_size} '
                    f'matches {other_dtype.__name__}. Switching dtype.'
                )
                dtype    = other_dtype
                itemsize = self._DTYPE_BYTES[dtype]
            else:
                LOG.warning(
                    f'byte_size mismatch: expected {expected} bytes for '
                    f'{dtype.__name__}, got {byte_size}. Data may be corrupt.'
                )

        # IDS on-disk layout: (slices, channels, samples) — C-order
        # We read with the correct on-disk shape, then transpose to
        # viewer convention: (samples, channels, slices)
        disk_shape   = (slices, channels, samples)
        viewer_shape = (samples, channels, slices)  # after transpose

        if lazy:
            # memmap with disk shape, then transpose to a view
            raw = np.memmap(
                self.filepath, dtype=dtype, mode='r',
                offset=byte_offset, shape=disk_shape
            )
            # np.transpose returns a view — no copy, memory-efficient
            data = np.transpose(raw, (2, 1, 0))
            LOG.debug(
                f'memmap loaded: disk_shape={disk_shape} '
                f'-> transposed viewer_shape={data.shape}'
            )
        else:
            with open(self.filepath, 'rb') as f:
                f.seek(byte_offset)
                raw = f.read(byte_size)
            data = np.frombuffer(raw, dtype=dtype).reshape(disk_shape)
            data = np.ascontiguousarray(np.transpose(data, (2, 1, 0)))
            LOG.debug(
                f'loaded: disk_shape={disk_shape} '
                f'-> transposed viewer_shape={data.shape}'
            )

        if dtype == np.int16:
            LOG.debug('Converting int16 -> float32')
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

        LOG.warning(f'Geolocations size {data.size} does not match any known shape')
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
