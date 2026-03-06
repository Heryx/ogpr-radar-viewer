"""
OGPR Radar Viewer

A professional application for visualizing and processing GPR data in OGPR format.
"""

__version__ = "1.0.0"
__author__ = "Heryx"

from .ogpr_parser import OGPRParser
from .signal_processing import SignalProcessor

__all__ = ['OGPRParser', 'SignalProcessor']