"""
GPR Project Management System

Manages project structure for incremental GPR data processing.
Each processing step is saved in a separate folder to allow
rollback and comparison.

Project Structure:
    ProjectName/
    ├── project.json              # Project metadata
    ├── raw/                      # Original data
    │   ├── Swath001_ch0.ogpr
    │   └── Swath001_ch0_meta.json
    ├── 01_time_zero/            # After Time-Zero correction
    ├── 02_dewow/                # After Dewow filter
    ├── 03_background/           # After Background Removal
    ├── 04_bandpass/             # After Bandpass filter
    ├── 05_gain/                 # After Gain application
    ├── 06_hilbert/              # After Hilbert transform
    └── 07_migration/            # After Migration
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np

LOG = logging.getLogger('ogpr_viewer')


# ---------------------------------------------------------------------------
# Processing Steps Definition
# ---------------------------------------------------------------------------

PROCESSING_STEPS = [
    'raw',
    '01_time_zero',
    '02_dewow', 
    '03_background',
    '04_bandpass',
    '05_gain',
    '06_hilbert',
    '07_migration',
]

STEP_DESCRIPTIONS = {
    'raw': 'Original unprocessed data',
    '01_time_zero': 'Time-zero correction applied',
    '02_dewow': 'Dewow (DC drift removal) applied',
    '03_background': 'Background removal applied',
    '04_bandpass': 'Bandpass filter applied',
    '05_gain': 'Gain correction applied',
    '06_hilbert': 'Hilbert envelope applied',
    '07_migration': 'Kirchhoff migration applied',
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ProcessingHistory:
    """Record of a single processing step."""
    step: str
    timestamp: str
    parameters: Dict[str, Any]
    description: str = ''


@dataclass
class AntennaInfo:
    """Antenna configuration."""
    model: str = 'Unknown'
    frequency_mhz: float = 400.0
    frequency_range: List[float] = field(default_factory=lambda: [100.0, 800.0])
    spacing_m: float = 0.05


@dataclass
class GPSInfo:
    """GPS/Navigation information."""
    enabled: bool = False
    coordinate_system: str = 'WGS84'
    traces_with_gps: List[int] = field(default_factory=list)
    gps_data: Dict[int, Dict[str, float]] = field(default_factory=dict)


@dataclass
class ProjectMetadata:
    """Complete project metadata."""
    name: str
    created: str
    modified: str
    description: str = ''
    current_step: str = 'raw'
    antenna: AntennaInfo = field(default_factory=AntennaInfo)
    gps: GPSInfo = field(default_factory=GPSInfo)
    sampling_time_ns: float = 0.117
    velocity_m_ns: float = 0.10
    history: List[ProcessingHistory] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GPRProject Class
# ---------------------------------------------------------------------------

class GPRProject:
    """
    Manages a GPR processing project with incremental steps.
    
    Usage:
        # Create new project
        project = GPRProject.create_new('/path/to/MyProject')
        
        # Import raw data
        project.import_raw_data('/path/to/Swath001_ch0.ogpr', metadata)
        
        # Process and save to next step
        data = project.load_data('raw', 'Swath001_ch0')
        processed = apply_time_zero(data)
        project.save_step('01_time_zero', 'Swath001_ch0', processed, params)
        
        # Load from specific step
        data = project.load_data('01_time_zero', 'Swath001_ch0')
    """

    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.metadata_file = self.project_path / 'project.json'
        self.metadata: Optional[ProjectMetadata] = None
        
        if self.metadata_file.exists():
            self.load_metadata()
        else:
            LOG.warning(f'Project metadata not found: {self.metadata_file}')

    # ----------------------------------------------------------------------
    # Creation and Loading
    # ----------------------------------------------------------------------

    @classmethod
    def create_new(
        cls,
        project_path: str | Path,
        name: Optional[str] = None,
        description: str = '',
        antenna: Optional[AntennaInfo] = None,
    ) -> GPRProject:
        """
        Create a new GPR project with folder structure.
        
        Args:
            project_path: Path where to create the project
            name: Project name (defaults to folder name)
            description: Project description
            antenna: Antenna configuration
            
        Returns:
            GPRProject instance
        """
        project_path = Path(project_path)
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Create folder structure
        for step in PROCESSING_STEPS:
            step_dir = project_path / step
            step_dir.mkdir(exist_ok=True)
            LOG.info(f'Created folder: {step_dir}')
        
        # Initialize metadata
        now = datetime.now().isoformat()
        metadata = ProjectMetadata(
            name=name or project_path.name,
            created=now,
            modified=now,
            description=description,
            current_step='raw',
            antenna=antenna or AntennaInfo(),
        )
        
        # Save metadata
        project = cls(project_path)
        project.metadata = metadata
        project.save_metadata()
        
        LOG.info(f'Created new project: {project_path}')
        return project

    @classmethod
    def load_existing(cls, project_path: str | Path) -> GPRProject:
        """Load an existing project."""
        project_path = Path(project_path)
        if not project_path.exists():
            raise FileNotFoundError(f'Project not found: {project_path}')
        
        project = cls(project_path)
        if project.metadata is None:
            raise ValueError(f'Invalid project: {project_path}')
        
        LOG.info(f'Loaded project: {project.metadata.name}')
        return project

    # ----------------------------------------------------------------------
    # Metadata Management
    # ----------------------------------------------------------------------

    def load_metadata(self) -> ProjectMetadata:
        """Load project metadata from JSON."""
        with open(self.metadata_file, 'r') as f:
            data = json.load(f)
        
        # Reconstruct nested dataclasses
        antenna = AntennaInfo(**data.get('antenna', {}))
        gps = GPSInfo(**data.get('gps', {}))
        history = [
            ProcessingHistory(**h) for h in data.get('history', [])
        ]
        
        self.metadata = ProjectMetadata(
            name=data['name'],
            created=data['created'],
            modified=data['modified'],
            description=data.get('description', ''),
            current_step=data.get('current_step', 'raw'),
            antenna=antenna,
            gps=gps,
            sampling_time_ns=data.get('sampling_time_ns', 0.117),
            velocity_m_ns=data.get('velocity_m_ns', 0.10),
            history=history,
        )
        return self.metadata

    def save_metadata(self):
        """Save project metadata to JSON."""
        if self.metadata is None:
            raise ValueError('No metadata to save')
        
        self.metadata.modified = datetime.now().isoformat()
        
        # Convert to dict (handles nested dataclasses)
        data = asdict(self.metadata)
        
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        LOG.debug(f'Saved metadata: {self.metadata_file}')

    # ----------------------------------------------------------------------
    # Data Import/Export
    # ----------------------------------------------------------------------

    def import_raw_data(
        self,
        source_path: str | Path,
        file_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Import raw data file into project.
        
        Args:
            source_path: Path to source .ogpr file
            file_metadata: Optional metadata dict to save alongside
            
        Returns:
            Path to imported file in raw/ folder
        """
        source_path = Path(source_path)
        dest_path = self.project_path / 'raw' / source_path.name
        
        # Copy data file
        shutil.copy2(source_path, dest_path)
        LOG.info(f'Imported: {source_path.name} -> raw/')
        
        # Save metadata if provided
        if file_metadata:
            meta_path = dest_path.with_suffix('.json')
            with open(meta_path, 'w') as f:
                json.dump(file_metadata, f, indent=2)
            LOG.debug(f'Saved file metadata: {meta_path.name}')
        
        return dest_path

    def list_files(self, step: str = 'raw') -> List[Path]:
        """
        List all .ogpr files in a specific step folder.
        
        Args:
            step: Processing step name (e.g. 'raw', '01_time_zero')
            
        Returns:
            List of Path objects for .ogpr files
        """
        step_dir = self.project_path / step
        if not step_dir.exists():
            LOG.warning(f'Step folder not found: {step}')
            return []
        
        files = sorted(step_dir.glob('*.ogpr'))
        return files

    def load_data(
        self,
        step: str,
        filename: str,
    ) -> np.ndarray:
        """
        Load data array from specific processing step.
        
        Args:
            step: Processing step (e.g. 'raw', '01_time_zero')
            filename: Base filename without extension
            
        Returns:
            Numpy array with radar data
        """
        file_path = self.project_path / step / f'{filename}.ogpr'
        
        if not file_path.exists():
            raise FileNotFoundError(f'File not found: {file_path}')
        
        # Load binary data (assuming float32 format after processing)
        # For raw data, use ogpr_parser.py to parse properly
        data = np.load(file_path, allow_pickle=False)
        
        LOG.debug(f'Loaded: {step}/{filename}.ogpr  shape={data.shape}')
        return data

    def save_step(
        self,
        step: str,
        filename: str,
        data: np.ndarray,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Save processed data to a specific step folder.
        
        Args:
            step: Target processing step
            filename: Base filename without extension
            data: Numpy array to save
            parameters: Processing parameters for this step
        """
        if step not in PROCESSING_STEPS:
            raise ValueError(f'Invalid step: {step}. Valid: {PROCESSING_STEPS}')
        
        step_dir = self.project_path / step
        step_dir.mkdir(exist_ok=True)
        
        file_path = step_dir / f'{filename}.ogpr'
        
        # Save as numpy array
        np.save(file_path, data.astype(np.float32))
        
        LOG.info(f'Saved: {step}/{filename}.ogpr  shape={data.shape}')
        
        # Update project metadata
        if self.metadata:
            self.metadata.current_step = step
            if parameters:
                history_entry = ProcessingHistory(
                    step=step,
                    timestamp=datetime.now().isoformat(),
                    parameters=parameters,
                    description=STEP_DESCRIPTIONS.get(step, ''),
                )
                self.metadata.history.append(history_entry)
            self.save_metadata()

    # ----------------------------------------------------------------------
    # Navigation
    # ----------------------------------------------------------------------

    def get_current_step(self) -> str:
        """Get current processing step."""
        return self.metadata.current_step if self.metadata else 'raw'

    def get_next_step(self, current_step: Optional[str] = None) -> Optional[str]:
        """
        Get the next processing step in the pipeline.
        
        Args:
            current_step: Current step (uses project.current_step if None)
            
        Returns:
            Next step name, or None if at the end
        """
        step = current_step or self.get_current_step()
        try:
            idx = PROCESSING_STEPS.index(step)
            if idx < len(PROCESSING_STEPS) - 1:
                return PROCESSING_STEPS[idx + 1]
        except ValueError:
            LOG.warning(f'Unknown step: {step}')
        return None

    def get_previous_step(self, current_step: Optional[str] = None) -> Optional[str]:
        """
        Get the previous processing step in the pipeline.
        
        Args:
            current_step: Current step (uses project.current_step if None)
            
        Returns:
            Previous step name, or None if at the beginning
        """
        step = current_step or self.get_current_step()
        try:
            idx = PROCESSING_STEPS.index(step)
            if idx > 0:
                return PROCESSING_STEPS[idx - 1]
        except ValueError:
            LOG.warning(f'Unknown step: {step}')
        return None

    def get_step_index(self, step: str) -> int:
        """Get numeric index of processing step."""
        try:
            return PROCESSING_STEPS.index(step)
        except ValueError:
            return -1

    # ----------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------

    def get_processing_history(self) -> List[ProcessingHistory]:
        """Get full processing history."""
        return self.metadata.history if self.metadata else []

    def export_summary(self) -> Dict[str, Any]:
        """Export project summary as dictionary."""
        if not self.metadata:
            return {}
        
        return {
            'name': self.metadata.name,
            'path': str(self.project_path),
            'created': self.metadata.created,
            'modified': self.metadata.modified,
            'current_step': self.metadata.current_step,
            'total_steps': len(self.metadata.history),
            'files_count': {step: len(self.list_files(step)) for step in PROCESSING_STEPS},
        }

    def __repr__(self) -> str:
        if self.metadata:
            return (
                f"GPRProject(name='{self.metadata.name}', "
                f"step='{self.metadata.current_step}', "
                f"files={len(self.list_files('raw'))})"
            )
        return f"GPRProject(path='{self.project_path}', metadata=None)"
