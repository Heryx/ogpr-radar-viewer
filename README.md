# OGPR Radar Viewer

A professional Python application for visualizing and processing Ground Penetrating Radar (GPR) data in OGPR format.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

### Data Handling
- ✅ **OGPR Format Support**: Complete parser for .ogpr file format
- ✅ **Efficient Memory Management**: Lazy loading for large datasets
- ✅ **Multi-channel Support**: Handle multiple radar channels
- ✅ **Georeferencing**: EPSG coordinate system support

### Signal Processing
- 🔧 **Background Removal**: DC offset and horizontal banding removal
- 🔧 **Bandpass Filter**: Butterworth filter with adjustable frequency range
- 🔧 **Gain Control**: Time-varying gain (exponential, linear, AGC)
- 🔧 **Hilbert Transform**: Envelope detection
- 🔧 **Time-Zero Correction**: First break picking and correction
- 🔧 **Migration**: (Coming soon)

### Visualization
- 📊 **B-Scan Display**: 2D radargram visualization
- 📊 **Interactive Navigation**: Browse through channels and slices
- 📊 **Multiple Colormaps**: Gray, seismic, viridis, etc.
- 📊 **Real-time Processing**: Apply filters interactively
- 📊 **Export**: Save processed images (PNG, PDF)

## Installation

### Requirements
- Python 3.8 or higher
- Windows (optimized for Windows, but works on Linux/macOS)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/Heryx/ogpr-radar-viewer.git
cd ogpr-radar-viewer

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m ogpr_viewer.main
```

### Development Install

```bash
# Install in editable mode
pip install -e .

# Run tests (coming soon)
pytest tests/
```

## Usage

### Basic Usage

```python
from ogpr_viewer import OGPRParser, SignalProcessor

# Load OGPR file
parser = OGPRParser('data/survey.ogpr')
data = parser.load_data()

# Apply processing
processor = SignalProcessor(data)
processed = processor.remove_background()
processed = processor.apply_bandpass(low_freq=100, high_freq=800)
processed = processor.apply_gain(gain_type='exp', factor=2.0)

# Visualize
processor.plot_bscan(channel=0, cmap='gray')
```

### GUI Application

1. Launch the application:
   ```bash
   python -m ogpr_viewer.main
   ```

2. **Load Data**: File → Open → Select your .ogpr file

3. **Navigate**: Use channel/slice selectors to browse data

4. **Apply Filters**: Adjust sliders in the Processing panel
   - Background Removal: Remove DC and horizontal banding
   - Bandpass: Set frequency range (MHz)
   - Gain: Adjust time-varying amplification
   - Hilbert: Toggle envelope detection

5. **Export**: File → Export Image

### Keyboard Shortcuts

- `Ctrl+O`: Open file
- `Ctrl+S`: Save processed data
- `Ctrl+E`: Export image
- `←/→`: Previous/Next slice
- `↑/↓`: Previous/Next channel
- `R`: Reset all filters
- `H`: Toggle Hilbert transform

## OGPR Format Specification

The .ogpr format structure:

```
[Header]
- Signature: "ogpr" (4 bytes)
- UUID: 32 hex characters
- Data offset: 8 hex characters
- JSON descriptor: metadata and data block info

[Data Blocks]
- Radar Volume: 3D array (samples × channels × slices) in float32
- Geolocations: Geographic coordinates for each sample
```

Example JSON descriptor:
```json
{
  "version": {"major": 1, "minor": 0},
  "mainDescriptor": {
    "samplesCount": 1024,
    "channelsCount": 11,
    "slicesCount": 560
  },
  "dataBlockDescriptors": [
    {
      "type": "Radar Volume",
      "byteSize": 12615680,
      "byteOffset": 918,
      "radar": {
        "samplingStep_m": 0.0397,
        "samplingTime_ns": 0.125,
        "frequency_MHz": 600,
        "polarization": "horizontal"
      }
    }
  ]
}
```

## Signal Processing Details

### Background Removal

Removes static background and horizontal banding:
- **DC Removal**: Subtract mean trace
- **Horizontal Banding**: Subtract median of each time sample

### Bandpass Filter

Butterworth filter to isolate radar frequency range:
- **Low frequency**: Remove low-frequency noise
- **High frequency**: Remove high-frequency noise
- **Order**: Filter steepness (default: 4)

### Gain Functions

1. **Exponential Gain**: `gain = exp(factor * time)`
2. **Linear Gain**: `gain = 1 + factor * time`
3. **AGC (Automatic Gain Control)**: Adaptive normalization

### Hilbert Transform

Computes instantaneous amplitude (envelope):
- Useful for detecting reflections
- Reduces phase effects

## Performance Optimization

### For Large Datasets

- **Lazy Loading**: Only load visible data
- **Caching**: Store processed results
- **Parallel Processing**: Multi-threaded filtering (coming soon)
- **Memory Mapping**: Direct file access without full load

### Tips

- Process individual channels/slices for very large files
- Use lower resolution preview mode
- Export processed data to avoid reprocessing

## Project Structure

```
ogpr-radar-viewer/
├── ogpr_viewer/
│   ├── __init__.py           # Package initialization
│   ├── main.py               # GUI main window
│   ├── ogpr_parser.py        # OGPR file parser
│   ├── signal_processing.py  # GPR signal processing
│   ├── visualization.py      # Plotting utilities
│   └── widgets/              # Custom Qt widgets
│       ├── __init__.py
│       ├── viewer_widget.py  # Main viewer panel
│       └── control_panel.py  # Processing controls
├── examples/
│   ├── basic_usage.py        # Simple examples
│   └── advanced_processing.py
├── tests/
│   ├── test_parser.py
│   └── test_processing.py
├── docs/
│   └── user_guide.md
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Roadmap

- [ ] 3D visualization support
- [ ] Migration algorithms
- [ ] Attribute analysis (instantaneous frequency, phase)
- [ ] Multi-file batch processing
- [ ] Plugin system for custom processing
- [ ] GPU acceleration for large datasets
- [ ] Machine learning-based anomaly detection

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{ogpr_viewer,
  title = {OGPR Radar Viewer},
  author = {Heryx},
  year = {2026},
  url = {https://github.com/Heryx/ogpr-radar-viewer}
}
```

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: [your-email@example.com]

## Acknowledgments

- GPR signal processing algorithms based on geophysical literature
- GUI framework: PyQt6
- Scientific computing: NumPy, SciPy, Matplotlib

---

**Made with ❤️ for the GPR community**