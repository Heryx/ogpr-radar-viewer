"""
Basic usage examples for OGPR Radar Viewer.

Demonstrates how to use the library programmatically.
"""

import numpy as np
from ogpr_viewer import OGPRParser, SignalProcessor
from ogpr_viewer.visualization import plot_bscan_standalone


def example_1_load_and_display():
    """
    Example 1: Load OGPR file and display raw data.
    """
    print("Example 1: Loading and displaying OGPR data")
    print("=" * 50)
    
    # Load OGPR file
    parser = OGPRParser('path/to/your/file.ogpr')
    
    # Print file info
    print(parser)
    print()
    
    # Load data
    data_dict = parser.load_data()
    
    # Get metadata
    meta = data_dict['metadata']
    print(f"Swath: {meta['swath_name']}")
    print(f"Dimensions: {meta['samples_count']} x {meta['channels_count']} x {meta['slices_count']}")
    print(f"Frequency: {meta['frequency_mhz']} MHz")
    print()
    
    # Extract B-scan (channel 0)
    bscan = parser.get_bscan(channel=0)
    print(f"B-scan shape: {bscan.shape}")
    
    # Plot
    plot_bscan_standalone(
        bscan,
        title="Raw GPR Data - Channel 0",
        cmap='gray'
    )


def example_2_basic_processing():
    """
    Example 2: Apply basic processing pipeline.
    """
    print("Example 2: Basic processing pipeline")
    print("=" * 50)
    
    # Load data
    parser = OGPRParser('path/to/your/file.ogpr')
    data_dict = parser.load_data()
    
    # Get B-scan
    bscan = parser.get_bscan(channel=0)
    
    # Create processor
    sampling_time = data_dict['metadata']['sampling_time_ns']
    processor = SignalProcessor(bscan, sampling_time)
    
    # Processing pipeline
    print("1. Removing background...")
    processor.remove_background(method='mean')
    
    print("2. Applying bandpass filter (100-800 MHz)...")
    processor.apply_bandpass(low_freq=100, high_freq=800)
    
    print("3. Applying exponential gain...")
    processor.apply_gain(gain_type='exp', factor=2.0)
    
    print("4. Normalizing...")
    processor.normalize(method='robust')
    
    # Get processed data
    processed = processor.get_processed_data()
    
    # Plot comparison
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Raw data
    ax1.imshow(bscan, cmap='gray', aspect='auto')
    ax1.set_title('Raw Data')
    ax1.set_xlabel('Trace')
    ax1.set_ylabel('Sample')
    
    # Processed data
    ax2.imshow(processed, cmap='gray', aspect='auto')
    ax2.set_title('Processed Data')
    ax2.set_xlabel('Trace')
    ax2.set_ylabel('Sample')
    
    plt.tight_layout()
    plt.show()


def example_3_advanced_processing():
    """
    Example 3: Advanced processing with Hilbert transform.
    """
    print("Example 3: Advanced processing")
    print("=" * 50)
    
    # Load data
    parser = OGPRParser('path/to/your/file.ogpr')
    data_dict = parser.load_data()
    bscan = parser.get_bscan(channel=0)
    
    sampling_time = data_dict['metadata']['sampling_time_ns']
    processor = SignalProcessor(bscan, sampling_time)
    
    # Full processing chain
    processor.remove_background(method='median')
    processor.dewow(window_size=50)
    processor.apply_bandpass(low_freq=100, high_freq=800, order=4)
    processor.apply_gain(gain_type='agc', window_ns=50)
    processor.apply_hilbert()  # Get envelope
    processor.normalize(method='minmax')
    
    # Get time and depth axes
    time_axis = processor.get_time_axis()
    depth_axis = processor.get_depth_axis(velocity=0.1)  # 0.1 m/ns for soil
    
    # Plot with depth axis
    processed = processor.get_processed_data()
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(
        processed,
        cmap='hot',
        aspect='auto',
        extent=[0, processed.shape[1], depth_axis[-1], depth_axis[0]]
    )
    
    ax.set_xlabel('Trace Number', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title('Processed GPR Data (Hilbert Envelope)', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Normalized Amplitude')
    plt.tight_layout()
    plt.show()


def example_4_multiple_channels():
    """
    Example 4: Compare multiple channels.
    """
    print("Example 4: Multi-channel comparison")
    print("=" * 50)
    
    # Load data
    parser = OGPRParser('path/to/your/file.ogpr')
    data_dict = parser.load_data()
    
    n_channels = data_dict['metadata']['channels_count']
    print(f"Processing {n_channels} channels...")
    
    import matplotlib.pyplot as plt
    
    # Create subplot grid
    n_cols = min(3, n_channels)
    n_rows = (n_channels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_channels > 1 else [axes]
    
    for ch in range(n_channels):
        # Get data for this channel
        bscan = parser.get_bscan(channel=ch)
        
        # Quick processing
        sampling_time = data_dict['metadata']['sampling_time_ns']
        processor = SignalProcessor(bscan, sampling_time)
        processor.remove_background()
        processor.apply_bandpass(100, 800)
        processor.normalize()
        
        processed = processor.get_processed_data()
        
        # Plot
        axes[ch].imshow(processed, cmap='gray', aspect='auto')
        axes[ch].set_title(f'Channel {ch}')
        axes[ch].set_xlabel('Trace')
        axes[ch].set_ylabel('Sample')
    
    # Hide extra subplots
    for i in range(n_channels, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def example_5_export_processed_data():
    """
    Example 5: Process and export data.
    """
    print("Example 5: Export processed data")
    print("=" * 50)
    
    # Load and process
    parser = OGPRParser('path/to/your/file.ogpr')
    data_dict = parser.load_data()
    bscan = parser.get_bscan(channel=0)
    
    sampling_time = data_dict['metadata']['sampling_time_ns']
    processor = SignalProcessor(bscan, sampling_time)
    
    # Apply processing
    processor.remove_background()
    processor.apply_bandpass(100, 800)
    processor.apply_gain('exp', 2.0)
    processor.normalize()
    
    processed = processor.get_processed_data()
    
    # Save as numpy array
    np.save('processed_data.npy', processed)
    print("Saved processed data to 'processed_data.npy'")
    
    # Save as image
    plot_bscan_standalone(
        processed,
        title="Processed GPR Data",
        cmap='seismic',
        save_path='processed_radargram.png'
    )
    print("Saved image to 'processed_radargram.png'")
    
    # Save metadata
    import json
    meta_export = {
        'original_file': str(parser.filepath),
        'channel': 0,
        'processing_steps': [
            'background_removal',
            'bandpass_100-800MHz',
            'exponential_gain_2.0',
            'robust_normalization'
        ],
        'shape': processed.shape,
        'metadata': data_dict['metadata']
    }
    
    with open('metadata.json', 'w') as f:
        json.dump(meta_export, f, indent=2)
    print("Saved metadata to 'metadata.json'")


if __name__ == '__main__':
    print("\nOGPR Radar Viewer - Usage Examples")
    print("=" * 50)
    print("\nNote: Replace 'path/to/your/file.ogpr' with actual file path")
    print("\nAvailable examples:")
    print("  1. Load and display raw data")
    print("  2. Basic processing pipeline")
    print("  3. Advanced processing with Hilbert transform")
    print("  4. Multi-channel comparison")
    print("  5. Export processed data")
    print("\nUncomment the example you want to run.")
    print()
    
    # Uncomment to run examples:
    # example_1_load_and_display()
    # example_2_basic_processing()
    # example_3_advanced_processing()
    # example_4_multiple_channels()
    # example_5_export_processed_data()