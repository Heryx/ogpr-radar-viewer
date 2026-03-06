# OGPR Radar Viewer - User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Loading Data](#loading-data)
3. [Navigation](#navigation)
4. [Signal Processing](#signal-processing)
5. [Display Options](#display-options)
6. [Exporting Results](#exporting-results)
7. [Tips and Best Practices](#tips-and-best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Heryx/ogpr-radar-viewer.git
   cd ogpr-radar-viewer
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application:**
   ```bash
   python -m ogpr_viewer.main
   ```

### First Run

When you first launch the application, you'll see:
- Empty visualization area (left panel)
- Control panel with disabled controls (right panel)
- Menu bar with File and Help options
- Status bar at the bottom

---

## Loading Data

### Opening OGPR Files

1. **Using Menu:**
   - Click `File` → `Open OGPR File...`
   - Or press `Ctrl+O`

2. **Select File:**
   - Navigate to your `.ogpr` file
   - Click `Open`

3. **Loading Process:**
   - Status bar shows "Loading [filename]..."
   - For large files (>100 MB), lazy loading is used automatically
   - Once loaded, file information appears in the control panel

### File Information Panel

After loading, you'll see:
- **File name**
- **Swath name**: Survey identifier
- **Samples**: Number of time samples per trace
- **Channels**: Number of radar channels
- **Slices**: Number of spatial positions
- **Frequency**: Antenna center frequency (MHz)
- **Sampling**: Time sampling interval (ns)

---

## Navigation

### Channel Selection

**Channel Selector:**
- Spin box to select active channel (0 to N-1)
- Use arrow buttons or type directly
- Press Enter to apply
- View updates automatically when changed

### Slice Range

**Slice Range Selectors:**
- **Start Slice**: First slice to display
- **End Slice**: Last slice to display
- Allows viewing subset of data
- Useful for focusing on specific areas

**Update View Button:**
- Click to apply slice range changes
- Reprocesses data for new range

### Keyboard Shortcuts

- `Ctrl+O`: Open file
- `Ctrl+E`: Export image
- `←/→`: Navigate slices (future feature)
- `↑/↓`: Navigate channels (future feature)

---

## Signal Processing

All processing is applied in real-time as you adjust parameters.

### Background Removal

**Purpose:** Remove static background and horizontal banding

**Options:**
- ☑ Enable/disable checkbox
- **Method:**
  - `mean`: Subtract average trace (faster)
  - `median`: Subtract median trace (more robust)

**When to use:**
- Always recommended as first step
- Removes antenna coupling and air wave
- Reduces horizontal striping

### Dewow Filter

**Purpose:** Remove low-frequency drift ("wow" effect)

**Parameters:**
- ☑ Enable/disable checkbox
- **Window**: Moving average window size (10-200 samples)
  - Smaller window: removes higher frequencies
  - Larger window: removes only very low frequencies

**When to use:**
- When data shows slow baseline drift
- After background removal
- Common in unshielded antennas

### Bandpass Filter

**Purpose:** Isolate radar signal frequency range

**Parameters:**
- ☑ Enable/disable checkbox
- **Low Freq**: Lower cutoff frequency (MHz)
- **High Freq**: Upper cutoff frequency (MHz)

**Recommended Settings:**
- For 600 MHz antenna: 100-800 MHz
- For 400 MHz antenna: 50-600 MHz
- For 200 MHz antenna: 25-400 MHz

**Tips:**
- Set range around antenna center frequency
- Low freq removes drift and low-frequency noise
- High freq removes high-frequency electronic noise

### Gain

**Purpose:** Amplify signals with depth (compensate for attenuation)

**Options:**
- ☑ Enable/disable checkbox
- **Type:**
  - `exp`: Exponential gain (most common)
  - `linear`: Linear increase with time
  - `agc`: Automatic Gain Control (adaptive)
- **Factor**: Gain strength (0.1-10.0)
  - Lower values: subtle enhancement
  - Higher values: strong amplification

**When to use:**
- Almost always recommended
- Essential for deep targets
- Adjust factor to balance shallow/deep signals

**Caution:**
- Too much gain amplifies noise at depth
- Apply after filtering for best results

### Hilbert Transform

**Purpose:** Extract signal envelope (instantaneous amplitude)

**Parameters:**
- ☑ Enable/disable checkbox
- No additional parameters

**When to use:**
- To emphasize reflections
- When phase information not needed
- For easier visual interpretation
- Before attribute analysis

**Effect:**
- Removes phase oscillations
- Shows only amplitude variations
- Creates smoother appearance

### Time-Zero Correction

**Purpose:** Align traces to common time-zero (first break)

**Parameters:**
- ☑ Enable/disable checkbox
- Automatic detection

**When to use:**
- When ground surface appears uneven
- To flatten surface reflection
- Before migration or velocity analysis

**Note:**
- Works best on clear surface reflection
- May fail on very noisy data

### Normalize

**Purpose:** Scale data to common range

**Options:**
- ☑ Enable/disable checkbox
- **Method:**
  - `minmax`: Scale to [0, 1] (simple)
  - `zscore`: Zero mean, unit variance (statistical)
  - `robust`: Use percentiles (resistant to outliers)

**When to use:**
- Final step before visualization
- Comparing multiple profiles
- Before export

---

## Display Options

### Colormap Selection

Choose visualization colormap:

- **gray**: Classic grayscale (recommended for most cases)
- **seismic**: Blue-white-red (emphasizes polarity)
- **RdBu_r**: Red-blue reversed (alternative polarity)
- **viridis**: Perceptually uniform (good for color-blind)
- **jet**: Rainbow (not recommended - can be misleading)
- **hot**: Black-red-yellow-white (high contrast)

**Best Practices:**
- Use `gray` for publication-quality figures
- Use `seismic` to highlight polarity changes
- Avoid `jet` for scientific work

---

## Exporting Results

### Export Image

1. **Menu Method:**
   - Click `File` → `Export Image...`
   - Or press `Ctrl+E`

2. **Button Method:**
   - Click `Export Image` button in control panel

3. **Choose Format:**
   - PNG: Raster image (high quality)
   - PDF: Vector format (scalable)

4. **Settings:**
   - Default DPI: 300 (high resolution)
   - Includes colorbar and labels

### Export Processed Data (Programmatic)

For batch processing, use Python scripts:

```python
from ogpr_viewer import OGPRParser, SignalProcessor
import numpy as np

# Load and process
parser = OGPRParser('data.ogpr')
data = parser.load_data()
bscan = parser.get_bscan(channel=0)

processor = SignalProcessor(bscan, data['metadata']['sampling_time_ns'])
processor.remove_background()
processor.apply_bandpass(100, 800)

# Save
processed = processor.get_processed_data()
np.save('processed.npy', processed)
```

---

## Tips and Best Practices

### Processing Workflow

**Recommended order:**

1. **Background Removal** (always first)
2. **Dewow** (if needed)
3. **Bandpass Filter** (essential)
4. **Gain** (after filtering)
5. **Time-Zero Correction** (if needed)
6. **Hilbert Transform** (optional)
7. **Normalize** (last step)

### Performance Tips

**For Large Files:**
- Files > 100 MB use automatic lazy loading
- Process limited slice ranges
- Export processed data to avoid reprocessing
- Close other applications to free RAM

**For Multiple Channels:**
- Process one channel at a time
- Use batch scripts for automation
- Compare channels after applying same processing

### Interpretation Tips

**Signal Characteristics:**
- **Hyperbolas**: Point targets (pipes, rocks)
- **Horizontal reflections**: Layer boundaries
- **Vertical features**: Faults, utilities
- **Random noise**: Increase filtering

**Depth Estimation:**
- Time (ns) × Velocity (m/ns) ÷ 2
- Typical velocities:
  - Air: 0.3 m/ns
  - Dry soil: 0.12-0.15 m/ns
  - Wet soil: 0.06-0.08 m/ns
  - Concrete: 0.10-0.13 m/ns
  - Asphalt: 0.12 m/ns

---

## Troubleshooting

### File Loading Issues

**Problem:** "Invalid OGPR signature"
- **Cause:** File is corrupted or not OGPR format
- **Solution:** Verify file integrity, check file format

**Problem:** "File not found"
- **Cause:** Incorrect file path
- **Solution:** Check file location and permissions

**Problem:** Slow loading
- **Cause:** Very large file
- **Solution:** Automatic lazy loading activates for files > 100 MB

### Processing Issues

**Problem:** No visible improvement after processing
- **Cause:** Inappropriate filter settings
- **Solution:**
  - Check frequency range matches antenna
  - Increase gain factor
  - Verify background removal is enabled

**Problem:** Over-amplified noise
- **Cause:** Too much gain
- **Solution:**
  - Reduce gain factor
  - Apply bandpass filter first
  - Use AGC instead of exponential gain

**Problem:** "Processing error"
- **Cause:** Invalid parameter combination
- **Solution:**
  - Reset all filters
  - Apply one filter at a time
  - Check bandpass frequencies (low < high)

### Display Issues

**Problem:** Image appears too bright/dark
- **Cause:** Auto-scaling limits
- **Solution:**
  - Apply normalization
  - Change colormap
  - Adjust gain settings

**Problem:** No detail visible
- **Cause:** Insufficient processing
- **Solution:**
  - Enable background removal
  - Apply gain
  - Use seismic colormap to see polarity

### Memory Issues

**Problem:** Application crashes with large files
- **Cause:** Insufficient RAM
- **Solution:**
  - Process smaller slice ranges
  - Use lazy loading (automatic)
  - Close other applications
  - Process channels separately

---

## Getting Help

**Documentation:**
- README: Overview and installation
- User Guide: This document
- Examples: `examples/` directory

**Support:**
- GitHub Issues: Report bugs or request features
- Email: [your-email@example.com]

**Contributing:**
- Fork the repository
- Submit pull requests
- Share processing workflows

---

**Version:** 1.0.0  
**Last Updated:** March 2026  
**Author:** Heryx