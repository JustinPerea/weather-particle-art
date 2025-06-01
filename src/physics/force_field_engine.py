#!/usr/bin/env python3
"""
Physics Force Field Engine
Converts weather data into 3D force fields using General Relativity and Quantum Field Theory
Part of Weather-Driven Viscous Particle Art System
"""

import numpy as np
import time
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

# Add parent directory for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.weather.noaa_api import WeatherObservation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Physics constants
SPEED_OF_LIGHT = 299792458.0  # m/s
GRAVITATIONAL_CONSTANT = 6.67430e-11  # m^3 kg^-1 s^-2
BOLTZMANN_CONSTANT = 1.38064852e-23  # J/K
PLANCK_CONSTANT = 6.62607015e-34  # J⋅s

class PhysicsEngine:
    """
    Physics engine that converts weather data into force fields
    using General Relativity and Quantum Field Theory principles
    """
    
    def __init__(self):
        # Container dimensions (normalized coordinates)
        self.box_size = np.array([2.0, 2.0, 1.0])  # Shallow box for poster display
        self.center = self.box_size / 2.0
        
        # Force field parameters
        self.max_force_magnitude = 10.0  # Maximum force magnitude
        self.smoothing_radius = 0.1  # Spatial smoothing for continuity
        
    def generate_3d_force_field(self, 
                               weather_data: WeatherObservation, 
                               resolution: Tuple[int, int, int] = (64, 32, 16)) -> np.ndarray:
        """
        Convert weather physics calculations to 3D spatial force field arrays
        
        Args:
            weather_data: WeatherObservation object with all parameters
            resolution: Tuple (x, y, z) grid resolution for force field
            
        Returns:
            np.array: Shape (64, 32, 16, 3) with force vectors at each grid point
            
        Requirements:
            - Generation time <200ms for real-time operation
            - All force values finite (no NaN/infinite)
            - Force magnitudes reasonable (<10.0 units)
        """
        start_time = time.time()
        
        # Create 3D grid
        x = np.linspace(0, self.box_size[0], resolution[0])
        y = np.linspace(0, self.box_size[1], resolution[1])
        z = np.linspace(0, self.box_size[2], resolution[2])
        
        # Create meshgrid for all points
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Initialize force field
        force_field = np.zeros(resolution + (3,), dtype=np.float32)
        
        # 1. Calculate metric tensor from weather (spacetime curvature)
        metric_tensor = self._calculate_metric_tensor(weather_data, X, Y, Z)
        
        # 2. Calculate Christoffel symbols (connection coefficients)
        christoffel = self._calculate_christoffel_symbols(metric_tensor, resolution)
        
        # 3. Calculate quantum field contributions
        quantum_fields = self._calculate_quantum_fields(weather_data, X, Y, Z)
        
        # 4. Combine all physics into force field
        # Temperature → spacetime curvature strength
        curvature_strength = self._map_temperature_to_curvature(weather_data.temperature)
        force_field += christoffel * curvature_strength
        
        # Pressure → gravitational well strength
        gravity_strength = self._map_pressure_to_gravity(weather_data.pressure)
        force_field += self._create_gravity_wells(X, Y, Z, gravity_strength)
        
        # Wind → directional flow fields
        wind_field = self._create_wind_field(weather_data, X, Y, Z)
        force_field += wind_field
        
        # Humidity → particle cohesion (modifies existing forces)
        cohesion_factor = weather_data.humidity / 100.0
        force_field *= (1.0 + cohesion_factor * 0.5)
        
        # UV Index → field coupling strength
        coupling = weather_data.uv_index / 11.0
        force_field += quantum_fields * coupling
        
        # Normalize and limit force magnitudes (with small safety margin)
        force_magnitudes = np.linalg.norm(force_field, axis=-1, keepdims=True)
        max_allowed = self.max_force_magnitude * 0.9999  # Tiny margin for floating point
        force_field = np.where(
            force_magnitudes > max_allowed,
            force_field * max_allowed / (force_magnitudes + 1e-6),
            force_field
        )
        
        # Ensure no NaN or infinite values
        force_field = np.nan_to_num(force_field, nan=0.0, posinf=0.0, neginf=0.0)
        
        generation_time = time.time() - start_time
        logger.info(f"Force field generated in {generation_time*1000:.1f}ms")
        
        # Verify performance requirement
        if generation_time > 0.2:
            logger.warning(f"Force field generation slower than 200ms target: {generation_time*1000:.1f}ms")
        
        return force_field
    
    def _calculate_metric_tensor(self, weather: WeatherObservation, X, Y, Z) -> np.ndarray:
        """Calculate spacetime metric tensor from weather conditions"""
        # Schwarzschild-like metric with weather-driven parameters
        # g_μν represents spacetime curvature
        
        # Temperature affects overall curvature
        T_factor = weather.temperature_kelvin / 300.0  # Normalized to room temp
        
        # Pressure creates local spacetime distortions
        P_factor = weather.pressure / 1013.25  # Normalized to standard pressure
        
        # Create radial distance from center
        R = np.sqrt((X - self.center[0])**2 + (Y - self.center[1])**2 + (Z - self.center[2])**2)
        R = np.maximum(R, 0.1)  # Avoid singularity
        
        # Schwarzschild radius analog (weather-driven)
        rs = 0.1 * (2.0 - P_factor) * T_factor
        
        # Metric components (simplified 3+1 decomposition)
        g_tt = -(1 - rs/R)  # Time-time component
        g_rr = 1 / (1 - rs/R)  # Radial-radial component
        
        # Combine into effective spatial metric
        metric = np.stack([g_tt, g_rr, g_rr, g_rr], axis=-1)
        
        return metric
    
    def _calculate_christoffel_symbols(self, metric: np.ndarray, resolution: Tuple) -> np.ndarray:
        """Calculate Christoffel symbols (connection coefficients) from metric"""
        # Γ^k_ij = (1/2) g^kl (∂g_il/∂x^j + ∂g_jl/∂x^i - ∂g_ij/∂x^l)
        
        force_field = np.zeros(resolution + (3,), dtype=np.float32)
        
        # Simplified calculation using metric gradients
        for i in range(3):  # For each spatial dimension
            # Calculate metric gradients
            if i == 0:  # X direction
                grad = np.gradient(metric[..., 1], axis=0)
            elif i == 1:  # Y direction
                grad = np.gradient(metric[..., 2], axis=1)
            else:  # Z direction
                grad = np.gradient(metric[..., 3], axis=2)
            
            # Christoffel symbols create forces based on metric gradients
            force_field[..., i] = -grad * 0.5
        
        return force_field
    
    def _calculate_quantum_fields(self, weather: WeatherObservation, X, Y, Z) -> np.ndarray:
        """Calculate quantum field contributions (scalar, vector, spinor, tensor)"""
        force_field = np.zeros(X.shape + (3,), dtype=np.float32)
        
        # 1. Scalar field (Klein-Gordon) - creates radial forces
        # ∇²φ - m²φ = 0
        mass_param = 1.0 / (weather.visibility + 1.0)  # Visibility affects field range
        
        # Create potential wells
        R = np.sqrt((X - self.center[0])**2 + (Y - self.center[1])**2 + (Z - self.center[2])**2)
        scalar_field = np.exp(-mass_param * R) * np.cos(2 * np.pi * R / 0.5)
        
        # Gradient of scalar field creates forces
        force_field[..., 0] = -np.gradient(scalar_field, axis=0)
        force_field[..., 1] = -np.gradient(scalar_field, axis=1)
        force_field[..., 2] = -np.gradient(scalar_field, axis=2)
        
        # 2. Vector field (Maxwell-like) - creates circulation
        # Cloud cover affects field circulation
        circulation = weather.cloud_cover / 100.0
        
        # Create vector potential
        A_x = -circulation * (Y - self.center[1]) / (R + 0.1)
        A_y = circulation * (X - self.center[0]) / (R + 0.1)
        
        # Curl of vector potential
        force_field[..., 0] += np.gradient(A_y, axis=2) * 0.1
        force_field[..., 1] += -np.gradient(A_x, axis=2) * 0.1
        force_field[..., 2] += (np.gradient(A_x, axis=1) - np.gradient(A_y, axis=0)) * 0.1
        
        # 3. Spinor field effects (simplified Dirac-like)
        # Precipitation creates helical forces
        if weather.precipitation > 0:
            helix_strength = weather.precipitation / 50.0  # Normalize
            theta = np.arctan2(Y - self.center[1], X - self.center[0])
            
            force_field[..., 0] += helix_strength * np.cos(theta + Z * 2 * np.pi)
            force_field[..., 1] += helix_strength * np.sin(theta + Z * 2 * np.pi)
            force_field[..., 2] += helix_strength * 0.2
        
        # 4. Tensor field (gravitational wave-like)
        # UV index creates oscillating quadrupole forces
        if weather.uv_index > 0:
            wave_amp = weather.uv_index / 11.0 * 0.1
            k = 2 * np.pi / 0.3  # Wave number
            
            # Quadrupole pattern
            h_plus = wave_amp * (np.cos(k * X) - np.cos(k * Y))
            h_cross = wave_amp * np.sin(k * X) * np.sin(k * Y)
            
            force_field[..., 0] += np.gradient(h_plus, axis=0) + np.gradient(h_cross, axis=1)
            force_field[..., 1] += np.gradient(h_plus, axis=1) - np.gradient(h_cross, axis=0)
        
        return force_field
    
    def _map_temperature_to_curvature(self, temperature: float) -> float:
        """Map temperature to spacetime curvature strength"""
        # Cold = high curvature, Hot = low curvature
        # Maps -20°C to 40°C → 2.0 to 0.5
        normalized_temp = (temperature + 20) / 60.0  # 0 to 1
        return 2.0 - 1.5 * normalized_temp
    
    def _map_pressure_to_gravity(self, pressure: float) -> float:
        """Map atmospheric pressure to gravitational well strength"""
        # Low pressure = strong gravity, High pressure = weak gravity
        # Maps 980-1040 hPa → 2.0 to 0.5
        normalized_pressure = (pressure - 980) / 60.0  # 0 to 1
        return 2.0 - 1.5 * normalized_pressure
    
    def _create_gravity_wells(self, X, Y, Z, strength: float) -> np.ndarray:
        """Create gravitational well forces"""
        force_field = np.zeros(X.shape + (3,), dtype=np.float32)
        
        # Create multiple gravity wells based on pressure patterns
        num_wells = int(3 + strength * 2)
        
        for i in range(num_wells):
            # Random well positions (deterministic from weather)
            angle = i * 2 * np.pi / num_wells
            well_x = self.center[0] + 0.5 * np.cos(angle)
            well_y = self.center[1] + 0.5 * np.sin(angle)
            well_z = self.center[2]
            
            # Distance to well
            R = np.sqrt((X - well_x)**2 + (Y - well_y)**2 + (Z - well_z)**2)
            R = np.maximum(R, 0.1)
            
            # Gravitational force (1/r² law with cutoff)
            F_magnitude = strength * 0.5 / (R**2 + 0.1)
            
            # Direction towards well
            force_field[..., 0] += F_magnitude * (well_x - X) / R
            force_field[..., 1] += F_magnitude * (well_y - Y) / R
            force_field[..., 2] += F_magnitude * (well_z - Z) / R
        
        return force_field
    
    def _create_wind_field(self, weather: WeatherObservation, X, Y, Z) -> np.ndarray:
        """Create wind-driven flow fields"""
        force_field = np.zeros(X.shape + (3,), dtype=np.float32)
        
        # Base wind direction and speed
        wind_vec = weather.wind_vector
        
        # Add turbulence based on wind speed
        if weather.wind_speed > 0:
            # Kolmogorov turbulence cascade
            turbulence_scales = [0.5, 0.25, 0.125]
            
            for scale in turbulence_scales:
                # Create vortices at different scales
                freq = 2 * np.pi / scale
                
                # Turbulent components
                turb_x = 0.1 * weather.wind_speed * np.sin(freq * Y) * np.cos(freq * Z)
                turb_y = 0.1 * weather.wind_speed * np.sin(freq * X) * np.cos(freq * Z)
                turb_z = 0.05 * weather.wind_speed * np.sin(freq * X) * np.sin(freq * Y)
                
                force_field[..., 0] += turb_x / len(turbulence_scales)
                force_field[..., 1] += turb_y / len(turbulence_scales)
                force_field[..., 2] += turb_z / len(turbulence_scales)
        
        # Add base wind flow (stronger at top of container)
        height_factor = Z / self.box_size[2]  # 0 at bottom, 1 at top
        force_field[..., 0] += wind_vec[0] * height_factor
        force_field[..., 1] += wind_vec[1] * height_factor
        
        # Boundary layer effects (reduced wind near walls)
        wall_distance = np.minimum(
            np.minimum(X, self.box_size[0] - X),
            np.minimum(Y, self.box_size[1] - Y)
        )
        boundary_factor = np.tanh(wall_distance / 0.1)
        force_field *= boundary_factor[..., np.newaxis]
        
        return force_field
    
    def sample_force_at_position(self, force_field: np.ndarray, position: np.ndarray) -> np.ndarray:
        """
        Sample force field at arbitrary 3D world position using trilinear interpolation
        
        Args:
            force_field: np.array from generate_3d_force_field()
            position: np.array [x, y, z] in world coordinates
            
        Returns:
            np.array: [fx, fy, fz] force vector at position
            
        Requirements:
            - Trilinear interpolation for smooth sampling
            - Performance: <1μs per sample for 1M particles
        """
        # Get field dimensions
        nx, ny, nz = force_field.shape[:3]
        
        # Convert world position to grid coordinates
        grid_pos = np.array([
            position[0] / self.box_size[0] * (nx - 1),
            position[1] / self.box_size[1] * (ny - 1),
            position[2] / self.box_size[2] * (nz - 1)
        ])
        
        # Clamp to valid range
        grid_pos = np.clip(grid_pos, 0, [nx-1.001, ny-1.001, nz-1.001])
        
        # Get integer indices
        i0, j0, k0 = grid_pos.astype(int)
        i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
        
        # Clamp indices
        i1 = min(i1, nx - 1)
        j1 = min(j1, ny - 1)
        k1 = min(k1, nz - 1)
        
        # Get fractional parts
        fx, fy, fz = grid_pos - np.array([i0, j0, k0])
        
        # Trilinear interpolation
        # Bottom face
        c000 = force_field[i0, j0, k0]
        c100 = force_field[i1, j0, k0]
        c010 = force_field[i0, j1, k0]
        c110 = force_field[i1, j1, k0]
        
        # Top face
        c001 = force_field[i0, j0, k1]
        c101 = force_field[i1, j0, k1]
        c011 = force_field[i0, j1, k1]
        c111 = force_field[i1, j1, k1]
        
        # Interpolate along x
        c00 = c000 * (1 - fx) + c100 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c11 = c011 * (1 - fx) + c111 * fx
        
        # Interpolate along y
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        
        # Interpolate along z
        force = c0 * (1 - fz) + c1 * fz
        
        return force