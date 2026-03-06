# Quick Start Guide

## Installation (5 minutes)

### Step 1: Clone Repository

```bash
git clone https://github.com/Heryx/ogpr-radar-viewer.git
cd ogpr-radar-viewer
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- NumPy, SciPy, Matplotlib, PyQt6

### Step 3: Launch Application

```bash
python -m ogpr_viewer.main
```

---

## First Use (2 minutes)

### 1. Open Your OGPR File

- Click **File** → **Open OGPR File...**
- Or press `Ctrl+O`
- Select your `.ogpr` file

### 2. View Your Data

- Data loads automatically
- Channel 0 is displayed by default
- File information appears in right panel

### 3. Apply Basic Processing

**Recommended first steps:**

1. ☑ **Background Removal** (always enable)
2. ☑ **Bandpass Filter**
   - Low Freq: 100 MHz
   - High Freq: 800 MHz (adjust for your antenna)
3. ☑ **Apply Gain**
   - Type: exp
   - Factor: 2.0
4. ☑ **Normalize**
   - Method: robust

### 4. Explore

- Change **Channel** to view different channels
- Try different **Colormaps** (gray, seismic, hot)
- Adjust **Slice Range** to focus on specific areas

### 5. Export

- Click **Export Image** button
- Or **File** → **Export Image...**
- Choose PNG or PDF format
- Save your processed radargram!

---

## Processing Presets

### Preset 1: Standard Processing

```
☑ Background Removal (mean)
☑ Bandpass Filter (100-800 MHz)
☑ Apply Gain (exp, factor 2.0)
☑ Normalize (robust)
```

**Use for:** Most GPR data, general purpose

### Preset 2: High Noise Environment

```
☑ Background Removal (median)
☑ Dewow (window 50)
☑ Bandpass Filter (150-700 MHz)
☑ Apply Gain (agc)
☑ Normalize (robust)
```

**Use for:** Noisy urban environments, electromagnetic interference

### Preset 3: Deep Targets

```
☑ Background Removal (mean)
☑ Bandpass Filter (100-800 MHz)
☑ Apply Gain (exp, factor 3.0)
☑ Hilbert Transform
☑ Normalize (minmax)
```

**Use for:** Archaeological surveys, deep utilities

### Preset 4: Shallow High-Resolution

```
☑ Background Removal (mean)
☑ Bandpass Filter (200-1000 MHz)
☑ Apply Gain (linear, factor 1.5)
☑ Time-Zero Correction
☑ Normalize (robust)
```

**Use for:** Shallow concrete scanning, rebar detection

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open file |
| `Ctrl+E` | Export image |
| `Ctrl+Q` | Quit application |
| `R` | Reset all filters (future) |
| `H` | Toggle Hilbert (future) |

---

## Common Issues

### "Invalid OGPR signature"

**Problem:** File is not valid OGPR format  
**Solution:** Verify file is actually .ogpr format, not renamed

### Application is slow

**Problem:** Large file size  
**Solution:** Process smaller slice ranges, use lazy loading (automatic for files >100 MB)

### No visible improvement after processing

**Problem:** Wrong filter settings  
**Solution:**
1. Enable Background Removal first
2. Check bandpass frequencies match your antenna
3. Increase gain factor
4. Try seismic colormap

### Can't see deep reflections

**Problem:** Insufficient gain  
**Solution:**
1. Increase gain factor (try 3.0-5.0)
2. Use exponential gain type
3. Apply after bandpass filter

---

## Programmatic Usage

### Minimal Example

```python
from ogpr_viewer import OGPRParser, SignalProcessor

# Load OGPR file
parser = OGPRParser('your_file.ogpr')
data_dict = parser.load_data()

# Get B-scan
bscan = parser.get_bscan(channel=0)

# Process
processor = SignalProcessor(bscan, 
                            data_dict['metadata']['sampling_time_ns'])
processor.remove_background()
processor.apply_bandpass(100, 800)
processor.apply_gain('exp', 2.0)

# Get result
processed = processor.get_processed_data()

# Plot
from ogpr_viewer.visualization import plot_bscan_standalone
plot_bscan_standalone(processed, cmap='gray')
```

### Batch Processing Multiple Files

```python
from pathlib import Path
from ogpr_viewer import OGPRParser, SignalProcessor
import numpy as np

# Process all OGPR files in directory
data_dir = Path('data/')
output_dir = Path('processed/')
output_dir.mkdir(exist_ok=True)

for ogpr_file in data_dir.glob('*.ogpr'):
    print(f"Processing {ogpr_file.name}...")
    
    # Load
    parser = OGPRParser(ogpr_file)
    data_dict = parser.load_data()
    bscan = parser.get_bscan(channel=0)
    
    # Process
    processor = SignalProcessor(bscan, 
                               data_dict['metadata']['sampling_time_ns'])
    processor.remove_background()
    processor.apply_bandpass(100, 800)
    processor.apply_gain('exp', 2.0)
    processor.normalize()
    
    # Save
    processed = processor.get_processed_data()
    output_file = output_dir / f"{ogpr_file.stem}_processed.npy"
    np.save(output_file, processed)
    
    print(f"  Saved to {output_file}")
```

---

## Next Steps

1. **Read User Guide:** See `docs/user_guide.md` for detailed documentation
2. **Try Examples:** Run scripts in `examples/` directory
3. **Experiment:** Try different processing combinations
4. **Contribute:** Share your workflows or improvements

---

## Support

- **Documentation:** `docs/user_guide.md`
- **Examples:** `examples/` directory
- **Issues:** [GitHub Issues](https://github.com/Heryx/ogpr-radar-viewer/issues)
- **Email:** [your-email@example.com]

---

**Happy processing! 🛰️**