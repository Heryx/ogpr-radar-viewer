# OGPR File Format Specification

Detailed technical specification of the OGPR format used for Ground Penetrating Radar data.

## Overview

OGPR is a binary format designed for efficient storage of GPR data with embedded metadata. The format consists of:

1. Text header with JSON descriptor
2. Binary data blocks containing radar measurements
3. Optional geolocation data

## File Structure

```
[OGPR File Structure]

+---------------------------+
| ASCII Header              |
|  - Signature (4 bytes)    |  "ogpr"
|  - UUID (32 bytes)        |  Unique identifier
|  - Offset (8 bytes)       |  Data start position (hex)
+---------------------------+
| JSON Descriptor           |
|  - Version info           |
|  - Main descriptor        |
|  - Data block descriptors |
+---------------------------+
| Binary Data Block 1       |
|  - Radar Volume           |  3D float32 array
+---------------------------+
| Binary Data Block 2       |
|  - Geolocations (optional)|  3D float64 array
+---------------------------+
```

## Header Format

### Signature (4 bytes)

```
Bytes 0-3: "ogpr" (ASCII)
```

Identifies file as OGPR format.

### UUID (32 bytes)

```
Bytes 4-35: 32 hexadecimal characters
```

Unique identifier for this dataset.

Example: `10230449e9f244059331c935735f3647`

### Data Offset (8 bytes)

```
Bytes 36-43: 8 hexadecimal characters
```

Byte offset from file start to first data block.

Example: `00000871` (2161 bytes in decimal)

## JSON Descriptor

### Structure

```json
{
  "version": {
    "major": 1,
    "minor": 0
  },
  "mainDescriptor": {
    "samplesCount": <int>,
    "channelsCount": <int>,
    "slicesCount": <int>,
    "metadata": {
      "swathName": <string>,
      "swathId": <string>,
      "arrayId": <int>
    }
  },
  "dataBlockDescriptors": [
    {
      "type": <string>,
      "name": <string>,
      "byteSize": <int>,
      "byteOffset": <int>,
      ...
    }
  ]
}
```

### Version Object

```json
"version": {
  "major": 1,
  "minor": 0
}
```

- `major`: Major version number (breaking changes)
- `minor`: Minor version number (compatible changes)

### Main Descriptor

```json
"mainDescriptor": {
  "samplesCount": 1024,
  "channelsCount": 11,
  "slicesCount": 560,
  "metadata": {
    "swathName": "Swath003",
    "swathId": "ca298b5c0aba198e85699e26de5f9cc6",
    "arrayId": 2
  }
}
```

**Fields:**

- `samplesCount`: Number of time samples per trace
- `channelsCount`: Number of radar channels (antennas)
- `slicesCount`: Number of spatial positions (traces)
- `metadata`:
  - `swathName`: Human-readable survey identifier
  - `swathId`: Unique swath identifier (UUID)
  - `arrayId`: Antenna array identifier

### Data Block Descriptors

Array of objects describing binary data blocks.

#### Radar Volume Block

```json
{
  "type": "Radar Volume",
  "name": "Swath003 Array02 Radar Data Volume",
  "byteSize": 12615680,
  "byteOffset": 918,
  "radar": {
    "samplingStep_m": 0.039725000970065594,
    "samplingTime_ns": 0.125,
    "propagationVelocity_mPerSec": 100000000.0,
    "fequency_MHz": 600,
    "polarization": "horizontal"
  },
  "metadata": {
    "processing": null
  }
}
```

**Fields:**

- `type`: "Radar Volume"
- `name`: Descriptive name
- `byteSize`: Size of data block in bytes
- `byteOffset`: Offset from file start to this block
- `radar`:
  - `samplingStep_m`: Spatial sampling interval (meters)
  - `samplingTime_ns`: Time sampling interval (nanoseconds)
  - `propagationVelocity_mPerSec`: EM wave velocity (m/s)
  - `fequency_MHz` or `frequency_MHz`: Antenna center frequency (MHz)
  - `polarization`: Antenna polarization ("horizontal", "vertical")
- `metadata`:
  - `processing`: Processing history (if any)

#### Geolocations Block

```json
{
  "type": "Sample Geolocations",
  "name": "Swath003 Array02 Sample Geographic Locations",
  "byteSize": 398720,
  "byteOffset": 12616598,
  "srs": {
    "type": "EPSG",
    "value": 32633
  }
}
```

**Fields:**

- `type`: "Sample Geolocations"
- `name`: Descriptive name
- `byteSize`: Size of data block in bytes
- `byteOffset`: Offset from file start to this block
- `srs`: Spatial Reference System
  - `type`: "EPSG" (coordinate system type)
  - `value`: EPSG code (e.g., 32633 = WGS 84 / UTM zone 33N)

## Binary Data Blocks

### Radar Volume Data

**Format:** 32-bit floating point (float32)  
**Byte Order:** Little-endian (typically)  
**Dimensions:** (samplesCount, channelsCount, slicesCount)  
**Size:** samplesCount × channelsCount × slicesCount × 4 bytes

**Array Layout:**

```python
radar_volume[sample, channel, slice] -> float32
```

- `sample`: Time sample index (0 to samplesCount-1)
- `channel`: Channel index (0 to channelsCount-1)
- `slice`: Spatial position index (0 to slicesCount-1)

**Value Range:** Typically -1.0 to +1.0 (normalized amplitude)

**Example:**

```python
import numpy as np

# Read radar volume
with open('data.ogpr', 'rb') as f:
    f.seek(byteOffset)  # Jump to data
    data = np.frombuffer(
        f.read(byteSize),
        dtype=np.float32
    ).reshape((samplesCount, channelsCount, slicesCount))
```

### Geolocation Data

**Format:** 64-bit floating point (float64)  
**Byte Order:** Little-endian (typically)  
**Dimensions:** Variable (depends on format variant)

**Common Layouts:**

1. **Per-sample geolocations:**
   ```python
   geolocations[slice, channel, sample, coordinate] -> float64
   ```
   - `coordinate`: 0=X (easting), 1=Y (northing), 2=Z (elevation)

2. **Per-trace geolocations:**
   ```python
   geolocations[slice, sample, coordinate] -> float64
   ```

3. **Per-slice geolocations:**
   ```python
   geolocations[slice, coordinate] -> float64
   ```

**Coordinate System:** Defined by `srs` in descriptor

**Example:**

```python
import numpy as np

# Read geolocations
with open('data.ogpr', 'rb') as f:
    f.seek(geo_byteOffset)
    geo_data = np.frombuffer(
        f.read(geo_byteSize),
        dtype=np.float64
    )

# Reshape based on expected dimensions
# (needs parsing logic to determine correct shape)
```

## Size Calculations

### Radar Volume Size

```
byteSize = samplesCount × channelsCount × slicesCount × 4
```

Example:
```
1024 samples × 11 channels × 560 slices × 4 bytes = 25,231,360 bytes (~24 MB)
```

### Geolocation Size (per-sample)

```
byteSize = slicesCount × channelsCount × samplesCount × 3 × 8
```

Example:
```
560 slices × 11 channels × 1024 samples × 3 coords × 8 bytes = 150,994,944 bytes (~144 MB)
```

### Total File Size

```
file_size ≈ header_size + radar_volume_size + geolocation_size
```

## Coordinate Systems

### Common EPSG Codes

| EPSG | Description |
|------|-------------|
| 4326 | WGS 84 (latitude/longitude) |
| 3857 | Web Mercator (used by web maps) |
| 32633 | WGS 84 / UTM zone 33N |
| 32634 | WGS 84 / UTM zone 34N |

### Coordinate Interpretation

**UTM Coordinates (e.g., EPSG:32633):**
- X: Easting (meters east of zone origin)
- Y: Northing (meters north of equator)
- Z: Elevation (meters above sea level or datum)

**Geographic Coordinates (EPSG:4326):**
- X: Longitude (degrees, -180 to 180)
- Y: Latitude (degrees, -90 to 90)
- Z: Elevation (meters)

## Parsing Example

### Complete Parser

```python
import json
import numpy as np

class OGPRReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.descriptor = None
    
    def read_header(self):
        with open(self.filepath, 'rb') as f:
            # Read signature
            signature = f.read(4).decode('ascii')
            assert signature == 'ogpr', f"Invalid signature: {signature}"
            
            # Read UUID
            uuid = f.read(32).decode('ascii')
            
            # Read offset
            offset_hex = f.read(8).decode('ascii')
            data_offset = int(offset_hex, 16)
            
            # Read JSON
            json_size = data_offset - f.tell()
            json_str = f.read(json_size).decode('utf-8')
            self.descriptor = json.loads(json_str)
        
        return self.descriptor
    
    def read_radar_volume(self):
        desc = self.descriptor
        radar_block = desc['dataBlockDescriptors'][0]
        
        # Get dimensions
        shape = (
            desc['mainDescriptor']['samplesCount'],
            desc['mainDescriptor']['channelsCount'],
            desc['mainDescriptor']['slicesCount']
        )
        
        # Read binary data
        with open(self.filepath, 'rb') as f:
            f.seek(radar_block['byteOffset'])
            data = np.frombuffer(
                f.read(radar_block['byteSize']),
                dtype=np.float32
            ).reshape(shape)
        
        return data

# Usage
reader = OGPRReader('survey.ogpr')
reader.read_header()
data = reader.read_radar_volume()
```

## Validation

### File Integrity Checks

1. **Signature validation:**
   ```python
   assert signature == 'ogpr'
   ```

2. **Size consistency:**
   ```python
   expected_size = samples * channels * slices * 4
   assert radar_block['byteSize'] == expected_size
   ```

3. **Offset validity:**
   ```python
   assert radar_block['byteOffset'] < file_size
   assert radar_block['byteOffset'] + radar_block['byteSize'] <= file_size
   ```

4. **JSON validity:**
   ```python
   try:
       descriptor = json.loads(json_str)
   except json.JSONDecodeError:
       raise ValueError("Invalid JSON descriptor")
   ```

## Extensions

### Future Versions

The format is designed to be extensible:

- Additional data blocks can be added
- New metadata fields in JSON descriptor
- Backward compatibility through version checking

### Custom Data Blocks

New block types can include:

- Processed data (filtered, migrated)
- Attribute volumes (amplitude, frequency, phase)
- Annotation data
- Auxiliary sensor data (GPS, IMU)

---

**Version:** 1.0.0  
**Last Updated:** March 2026