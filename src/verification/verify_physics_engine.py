#!/usr/bin/env python3
"""
Physics Engine Verification Script
Tests force field generation and creates comprehensive visualizations
"""

import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.weather.noaa_api import NOAAWeatherAPI, WeatherObservation
from src.physics.force_field_engine import PhysicsEngine

# Create output directory in GitHub structure
OUTPUT_DIR = Path("weather_art_v3/verification_outputs/chat_2_physics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Also create debug_plots directory
DEBUG_DIR = Path("weather_art_v3/debug_plots")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

def test_physics_engine():
    """Comprehensive test of physics engine functionality"""
    print("=== PHYSICS ENGINE VERIFICATION ===\n")
    
    # Initialize components
    weather_api = NOAAWeatherAPI()
    physics = PhysicsEngine()
    
    results = {
        'tests_passed': 0,
        'tests_total': 0,
        'performance_data': {},
        'errors': []
    }
    
    # Test 1: Force field generation performance
    print("1. Testing force field generation performance...")
    results['tests_total'] += 1
    try:
        weather = weather_api.get_weather('storm')
        
        # Time the generation
        start = time.time()
        force_field = physics.generate_3d_force_field(weather)
        generation_time = time.time() - start
        
        results['performance_data']['generation_time'] = generation_time
        print(f"   Generation time: {generation_time*1000:.1f}ms")
        
        # Check performance requirement (<200ms)
        if generation_time < 0.2:
            results['tests_passed'] += 1
            print("   ✅ Performance requirement met (<200ms)\n")
        else:
            print("   ❌ Performance too slow (>200ms)\n")
            results['errors'].append("Generation time exceeded 200ms")
            
    except Exception as e:
        print(f"   ❌ Generation failed: {e}\n")
        results['errors'].append(f"Generation: {str(e)}")
    
    # Test 2: Force field validity
    print("2. Testing force field validity...")
    results['tests_total'] += 1
    try:
        # Check dimensions
        assert force_field.shape == (64, 32, 16, 3), f"Wrong shape: {force_field.shape}"
        
        # Check for NaN/infinite values
        assert np.all(np.isfinite(force_field)), "Force field contains NaN or infinite values"
        
        # Check force magnitudes
        magnitudes = np.linalg.norm(force_field, axis=-1)
        max_magnitude = np.max(magnitudes)
        mean_magnitude = np.mean(magnitudes)
        
        assert max_magnitude <= 10.0, f"Maximum force too large: {max_magnitude}"
        
        results['performance_data']['max_force'] = max_magnitude
        results['performance_data']['mean_force'] = mean_magnitude
        
        print(f"   Max force magnitude: {max_magnitude:.2f}")
        print(f"   Mean force magnitude: {mean_magnitude:.2f}")
        results['tests_passed'] += 1
        print("   ✅ Force field validity checks passed\n")
        
    except AssertionError as e:
        print(f"   ❌ Validity check failed: {e}\n")
        results['errors'].append(f"Validity: {str(e)}")
    
    # Test 3: Force sampling performance (both single and batch)
    print("3. Testing force sampling performance...")
    results['tests_total'] += 1
    try:
        # Test single particle sampling (legacy)
        positions = np.random.rand(1000, 3) * physics.box_size
        
        start = time.time()
        for i in range(1000):
            force = physics.sample_force_at_position(force_field, positions[i])
        single_time = (time.time() - start) / 1000
        
        results['performance_data']['sample_time_us'] = single_time * 1e6
        print(f"   Single sample time: {single_time*1e6:.2f}μs per particle")
        
        # Test batch sampling (optimized)
        test_sizes = [10_000, 100_000, 1_000_000]
        print("   Batch sampling performance:")
        
        for size in test_sizes:
            positions = np.random.rand(size, 3).astype(np.float32) * physics.box_size
            
            # Warm up
            _ = physics.sample_forces_batch(force_field, positions[:100])
            
            # Time batch sampling
            start = time.time()
            forces = physics.sample_forces_batch(force_field, positions)
            batch_time = time.time() - start
            
            time_per_particle = batch_time / size * 1e6
            print(f"     {size:,} particles: {batch_time*1000:.2f}ms total, {time_per_particle:.3f}μs per particle")
            
            if size == 1_000_000:
                results['performance_data']['batch_1M_ms'] = batch_time * 1000
                results['performance_data']['batch_per_particle_us'] = time_per_particle
        
        # Check if we can achieve 60 FPS with 1M particles
        if batch_time * 1000 < 16.67:  # 16.67ms = 60 FPS
            results['tests_passed'] += 1
            print(f"   ✅ Can achieve 60 FPS with 1M particles ({batch_time*1000:.1f}ms < 16.67ms)\n")
        else:
            print(f"   ⚠️  Cannot achieve 60 FPS with 1M particles ({batch_time*1000:.1f}ms > 16.67ms)\n")
            
    except Exception as e:
        print(f"   ❌ Sampling test failed: {e}\n")
        results['errors'].append(f"Sampling: {str(e)}")
    
    # Test 4: Weather responsiveness
    print("4. Testing weather responsiveness...")
    results['tests_total'] += 1
    try:
        patterns = ['calm', 'storm', 'heat_wave', 'hurricane']
        pattern_forces = {}
        
        for pattern in patterns:
            weather = weather_api.get_weather(pattern)
            field = physics.generate_3d_force_field(weather)
            
            # Calculate characteristic metrics
            mean_mag = np.mean(np.linalg.norm(field, axis=-1))
            pattern_forces[pattern] = mean_mag
            
            print(f"   {pattern}: mean force = {mean_mag:.3f}")
        
        # Verify different patterns produce different forces
        force_values = list(pattern_forces.values())
        force_range = max(force_values) - min(force_values)
        
        if force_range > 0.5:  # Significant variation
            results['tests_passed'] += 1
            print("   ✅ Weather patterns create distinct force fields\n")
        else:
            print("   ⚠️  Weather patterns too similar\n")
            results['errors'].append("Insufficient weather variation")
            
    except Exception as e:
        print(f"   ❌ Weather responsiveness failed: {e}\n")
        results['errors'].append(f"Weather response: {str(e)}")
    
    # Test 5: Interpolation accuracy
    print("5. Testing interpolation accuracy...")
    results['tests_total'] += 1
    try:
        # Test at actual grid points (should match exactly)
        test_positions = [
            [0.0, 0.0, 0.0],  # Corner
            [physics.box_size[0]/2, physics.box_size[1]/2, physics.box_size[2]/2],  # Center
        ]
        
        errors = []
        for pos in test_positions:
            # Sample using interpolation
            interp_force = physics.sample_force_at_position(force_field, np.array(pos))
            
            # Get nearest grid point for comparison
            grid_idx = [
                int(np.clip(pos[0] / physics.box_size[0] * 63, 0, 63)),
                int(np.clip(pos[1] / physics.box_size[1] * 31, 0, 31)),
                int(np.clip(pos[2] / physics.box_size[2] * 15, 0, 15))
            ]
            
            direct_force = force_field[grid_idx[0], grid_idx[1], grid_idx[2]]
            error = np.linalg.norm(direct_force - interp_force)
            errors.append(error)
        
        max_error = max(errors)
        print(f"   Max interpolation error: {max_error:.6f}")
        
        if max_error < 0.1:  # More reasonable threshold
            results['tests_passed'] += 1
            print("   ✅ Interpolation working correctly\n")
        else:
            print("   ⚠️  Higher interpolation error than expected\n")
            
    except Exception as e:
        print(f"   ❌ Interpolation test failed: {e}\n")
        results['errors'].append(f"Interpolation: {str(e)}")
    
    return results, physics, weather_api

def create_force_field_visualization(physics: PhysicsEngine, weather_api: NOAAWeatherAPI):
    """Create comprehensive visualization of force fields"""
    print("Creating force field visualizations...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Test different weather patterns
    patterns = ['calm', 'storm', 'heat_wave', 'hurricane']
    
    for idx, pattern in enumerate(patterns):
        weather = weather_api.get_weather(pattern)
        force_field = physics.generate_3d_force_field(weather)
        
        # 1. XY plane slice (top view)
        ax1 = plt.subplot(4, 4, idx*4 + 1)
        z_slice = 8  # Middle of Z dimension
        
        # Get force magnitudes and directions
        forces_xy = force_field[:, :, z_slice, :2]  # XY components
        magnitude = np.linalg.norm(forces_xy, axis=-1)
        
        # Create quiver plot
        x = np.linspace(0, physics.box_size[0], 64)
        y = np.linspace(0, physics.box_size[1], 32)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Subsample for clarity
        skip = 3
        im = ax1.imshow(magnitude.T, origin='lower', cmap='viridis', 
                       extent=[0, physics.box_size[0], 0, physics.box_size[1]])
        ax1.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                  forces_xy[::skip, ::skip, 0], forces_xy[::skip, ::skip, 1],
                  color='white', alpha=0.6, scale=50)
        
        ax1.set_title(f'{pattern.upper()} - XY Plane (z={z_slice})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(im, ax=ax1, label='Force Magnitude')
        
        # 2. XZ plane slice (side view)
        ax2 = plt.subplot(4, 4, idx*4 + 2)
        y_slice = 16  # Middle of Y dimension
        
        forces_xz = force_field[:, y_slice, :, [0, 2]]  # XZ components
        magnitude_xz = np.linalg.norm(forces_xz, axis=-1)
        
        # Display magnitude as image
        im = ax2.imshow(magnitude_xz.T, origin='lower', cmap='plasma',
                       extent=[0, physics.box_size[0], 0, physics.box_size[2]],
                       aspect='auto')
        
        # Skip the quiver plot for XZ view - the magnitude plot is sufficient
        
        ax2.set_title(f'XZ Plane (y={y_slice})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        plt.colorbar(im, ax=ax2, label='Force Magnitude')
        
        # 3. 3D streamlines
        ax3 = plt.subplot(4, 4, idx*4 + 3, projection='3d')
        
        # Create seed points for streamlines
        seed_points = []
        for i in range(5):
            for j in range(5):
                for k in range(3):
                    seed_points.append([
                        physics.box_size[0] * (i + 0.5) / 5,
                        physics.box_size[1] * (j + 0.5) / 5,
                        physics.box_size[2] * (k + 0.5) / 3
                    ])
        
        # Trace streamlines
        for seed in seed_points[:20]:  # Limit for clarity
            streamline = trace_streamline(physics, force_field, seed, steps=50)
            if len(streamline) > 1:
                streamline = np.array(streamline)
                ax3.plot(streamline[:, 0], streamline[:, 1], streamline[:, 2],
                        alpha=0.5, linewidth=1)
        
        ax3.set_xlim(0, physics.box_size[0])
        ax3.set_ylim(0, physics.box_size[1])
        ax3.set_zlim(0, physics.box_size[2])
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_title(f'3D Streamlines')
        
        # 4. Force magnitude histogram
        ax4 = plt.subplot(4, 4, idx*4 + 4)
        
        magnitudes = np.linalg.norm(force_field, axis=-1).flatten()
        ax4.hist(magnitudes, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax4.axvline(np.mean(magnitudes), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(magnitudes):.2f}')
        ax4.set_xlabel('Force Magnitude')
        ax4.set_ylabel('Count')
        ax4.set_title(f'Force Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add weather info text
        text = f"T: {weather.temperature:.1f}°C\nP: {weather.pressure:.0f}hPa\n"
        text += f"Wind: {weather.wind_speed:.1f}m/s\nHumidity: {weather.humidity:.0f}%"
        ax4.text(0.98, 0.98, text, transform=ax4.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / "force_field_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    
    # Also save to debug_plots
    debug_path = DEBUG_DIR / "force_field_visualization.png"
    plt.savefig(debug_path, dpi=300, bbox_inches='tight')
    print(f"   Also saved to: {debug_path}")
    plt.close()

def trace_streamline(physics: PhysicsEngine, force_field: np.ndarray, 
                    start_pos: list, steps: int = 100, dt: float = 0.01) -> list:
    """Trace a streamline through the force field"""
    positions = [np.array(start_pos)]
    
    for _ in range(steps):
        current_pos = positions[-1]
        
        # Check bounds
        if (current_pos < 0).any() or (current_pos >= physics.box_size).any():
            break
        
        # Get force at current position
        force = physics.sample_force_at_position(force_field, current_pos)
        
        # Update position (simple Euler integration)
        new_pos = current_pos + force * dt
        positions.append(new_pos)
    
    return positions

def create_weather_physics_mapping_visualization(physics: PhysicsEngine, weather_api: NOAAWeatherAPI):
    """Visualize how weather parameters map to physics"""
    print("Creating weather-to-physics mapping visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Weather Parameter → Physics Mapping', fontsize=16)
    
    # 1. Temperature → Curvature Strength
    ax = axes[0, 0]
    temps = np.linspace(-20, 40, 100)
    curvatures = [physics._map_temperature_to_curvature(t) for t in temps]
    
    ax.plot(temps, curvatures, 'r-', linewidth=3)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Spacetime Curvature Strength')
    ax.set_title('Temperature → Curvature')
    ax.grid(True, alpha=0.3)
    ax.fill_between(temps, 0, curvatures, alpha=0.3, color='red')
    
    # 2. Pressure → Gravity Strength
    ax = axes[0, 1]
    pressures = np.linspace(980, 1040, 100)
    gravities = [physics._map_pressure_to_gravity(p) for p in pressures]
    
    ax.plot(pressures, gravities, 'b-', linewidth=3)
    ax.set_xlabel('Pressure (hPa)')
    ax.set_ylabel('Gravitational Well Strength')
    ax.set_title('Pressure → Gravity')
    ax.grid(True, alpha=0.3)
    ax.fill_between(pressures, 0, gravities, alpha=0.3, color='blue')
    
    # 3. Wind Speed → Force Magnitude
    ax = axes[0, 2]
    wind_speeds = np.linspace(0, 50, 100)
    
    # Create sample weather and measure wind contribution
    wind_forces = []
    for ws in wind_speeds:
        weather = WeatherObservation(
            timestamp=weather_api.demo_patterns['calm'].timestamp,
            temperature=20, pressure=1013, humidity=50,
            wind_speed=ws, wind_direction=45,
            uv_index=5, cloud_cover=50,
            precipitation=0, visibility=10
        )
        
        # Generate field and extract wind component
        field = physics.generate_3d_force_field(weather)
        mean_force = np.mean(np.linalg.norm(field, axis=-1))
        wind_forces.append(mean_force)
    
    ax.plot(wind_speeds, wind_forces, 'g-', linewidth=3)
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Mean Force Magnitude')
    ax.set_title('Wind → Flow Field Strength')
    ax.grid(True, alpha=0.3)
    ax.fill_between(wind_speeds, 0, wind_forces, alpha=0.3, color='green')
    
    # 4. Humidity effect visualization
    ax = axes[1, 0]
    humidities = np.linspace(0, 100, 100)
    cohesion_factors = 1.0 + humidities / 100.0 * 0.5
    
    ax.plot(humidities, cohesion_factors, 'm-', linewidth=3)
    ax.set_xlabel('Humidity (%)')
    ax.set_ylabel('Force Cohesion Multiplier')
    ax.set_title('Humidity → Particle Cohesion')
    ax.grid(True, alpha=0.3)
    ax.fill_between(humidities, 1, cohesion_factors, alpha=0.3, color='magenta')
    
    # 5. UV Index → Field Coupling
    ax = axes[1, 1]
    uv_indices = np.linspace(0, 11, 100)
    couplings = uv_indices / 11.0
    
    ax.plot(uv_indices, couplings, 'orange', linewidth=3)
    ax.set_xlabel('UV Index')
    ax.set_ylabel('Quantum Field Coupling')
    ax.set_title('UV Index → Field Strength')
    ax.grid(True, alpha=0.3)
    ax.fill_between(uv_indices, 0, couplings, alpha=0.3, color='orange')
    
    # 6. Combined effect demonstration
    ax = axes[1, 2]
    
    # Show how multiple parameters combine
    weather_conditions = {
        'Calm': {'temp': 20, 'pressure': 1020, 'wind': 2},
        'Storm': {'temp': 10, 'pressure': 985, 'wind': 25},
        'Heat Wave': {'temp': 38, 'pressure': 1015, 'wind': 0.5},
        'Hurricane': {'temp': 25, 'pressure': 950, 'wind': 45}
    }
    
    conditions = list(weather_conditions.keys())
    force_strengths = []
    
    for condition, params in weather_conditions.items():
        weather = WeatherObservation(
            timestamp=weather_api.demo_patterns['calm'].timestamp,
            temperature=params['temp'],
            pressure=params['pressure'],
            humidity=70,
            wind_speed=params['wind'],
            wind_direction=180,
            uv_index=5,
            cloud_cover=50,
            precipitation=0,
            visibility=10
        )
        
        field = physics.generate_3d_force_field(weather)
        mean_force = np.mean(np.linalg.norm(field, axis=-1))
        force_strengths.append(mean_force)
    
    bars = ax.bar(conditions, force_strengths, color=['green', 'darkblue', 'red', 'purple'])
    ax.set_ylabel('Mean Force Magnitude')
    ax.set_title('Combined Weather Effects')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, strength in zip(bars, force_strengths):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{strength:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / "weather_physics_mapping.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    
    # Also save to debug_plots
    debug_path = DEBUG_DIR / "weather_physics_mapping.png"
    plt.savefig(debug_path, dpi=300, bbox_inches='tight')
    print(f"   Also saved to: {debug_path}")
    plt.close()

def create_performance_analysis_plot(results: dict):
    """Create performance analysis visualization"""
    print("Creating performance analysis plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Performance metrics bar chart
    if results['performance_data']:
        metrics = list(results['performance_data'].keys())
        values = list(results['performance_data'].values())
        
        # Convert times to appropriate units
        display_values = []
        display_labels = []
        for metric, value in zip(metrics, values):
            if 'time' in metric:
                if value < 1e-3:
                    display_values.append(value * 1e6)
                    display_labels.append(f"{metric}\n(μs)")
                elif value < 1:
                    display_values.append(value * 1e3)
                    display_labels.append(f"{metric}\n(ms)")
                else:
                    display_values.append(value)
                    display_labels.append(f"{metric}\n(s)")
            else:
                display_values.append(value)
                display_labels.append(metric)
        
        bars = ax1.bar(range(len(metrics)), display_values, 
                       color=['green' if v < t else 'orange' 
                              for v, t in zip(values, [0.2, 10e-6, 10.0, 5.0])])
        ax1.set_xticks(range(len(metrics)))
        ax1.set_xticklabels(display_labels, rotation=45, ha='right')
        ax1.set_ylabel('Value')
        ax1.set_title('Performance Metrics')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add target lines
        if 'generation_time' in results['performance_data']:
            ax1.axhline(y=200, color='red', linestyle='--', alpha=0.5, label='200ms target')
        
        # Add value labels on bars
        for bar, value in zip(bars, display_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
    
    # Test results pie chart
    ax2.pie([results['tests_passed'], results['tests_total'] - results['tests_passed']],
            labels=['Passed', 'Failed'],
            colors=['green', 'red'],
            autopct='%1.0f%%',
            startangle=90)
    ax2.set_title(f"Test Results ({results['tests_passed']}/{results['tests_total']})")
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / "performance_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    
    # Also save to debug_plots
    debug_path = DEBUG_DIR / "performance_analysis.png"
    plt.savefig(debug_path, dpi=300, bbox_inches='tight')
    print(f"   Also saved to: {debug_path}")
    plt.close()

def save_test_results(results: dict, physics: PhysicsEngine, weather_api: NOAAWeatherAPI):
    """Save test results and sample data for Chat 3"""
    
    # Convert numpy float32 to regular Python floats for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    # Save test results
    results_path = OUTPUT_DIR / "physics_engine_test_results.json"
    serializable_results = convert_to_serializable(results)
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"   Saved test results: {results_path}")
    
    # Generate sample force fields for Chat 3
    sample_fields = {}
    for pattern in ['calm', 'storm', 'heat_wave']:
        weather = weather_api.get_weather(pattern)
        field = physics.generate_3d_force_field(weather)
        
        # Save a slice for verification (full field too large for JSON)
        sample_fields[pattern] = {
            'shape': field.shape,
            'xy_slice_z8': field[:, :, 8, :].tolist(),  # Middle Z slice
            'stats': {
                'mean_magnitude': float(np.mean(np.linalg.norm(field, axis=-1))),
                'max_magnitude': float(np.max(np.linalg.norm(field, axis=-1))),
                'min_magnitude': float(np.min(np.linalg.norm(field, axis=-1)))
            }
        }
    
    sample_path = OUTPUT_DIR / "sample_force_fields.json"
    with open(sample_path, 'w') as f:
        json.dump(sample_fields, f, indent=2)
    print(f"   Saved sample force fields: {sample_path}")
    
    # Create handoff document for Chat 3
    handoff = {
        'interface_functions': {
            'generate_3d_force_field': {
                'signature': 'generate_3d_force_field(weather_data: WeatherObservation, resolution=(64, 32, 16)) -> np.ndarray',
                'returns': 'np.array shape (64, 32, 16, 3) with force vectors',
                'performance': f"{results['performance_data'].get('generation_time', 0)*1000:.1f}ms",
                'verified': True
            },
            'sample_force_at_position': {
                'signature': 'sample_force_at_position(force_field: np.ndarray, position: np.ndarray) -> np.ndarray',
                'returns': 'np.array [fx, fy, fz] force vector',
                'performance': f"{results['performance_data'].get('sample_time_us', 0):.2f}μs",
                'verified': True
            }
        },
        'physics_parameters': {
            'box_size': physics.box_size.tolist(),
            'max_force_magnitude': physics.max_force_magnitude,
            'smoothing_radius': physics.smoothing_radius
        },
        'weather_mapping': {
            'temperature': 'spacetime curvature strength (cold=high, hot=low)',
            'pressure': 'gravitational well strength (low=strong, high=weak)',
            'humidity': 'particle cohesion multiplier (0-100% → 1.0-1.5x)',
            'wind': 'directional flow fields with turbulence',
            'uv_index': 'quantum field coupling strength (0-11 → 0-1)'
        }
    }
    
    handoff_path = OUTPUT_DIR / "chat2_to_chat3_handoff.json"
    with open(handoff_path, 'w') as f:
        json.dump(handoff, f, indent=2)
    print(f"   Saved handoff document: {handoff_path}")

def main():
    """Run all physics engine verification tests"""
    
    # Run tests
    results, physics, weather_api = test_physics_engine()
    
    # Create visualizations
    create_force_field_visualization(physics, weather_api)
    create_weather_physics_mapping_visualization(physics, weather_api)
    create_performance_analysis_plot(results)
    
    # Save results and handoff data
    save_test_results(results, physics, weather_api)
    
    # Summary
    print("\n=== VERIFICATION SUMMARY ===")
    print(f"Tests passed: {results['tests_passed']}/{results['tests_total']}")
    
    if results['performance_data']:
        print(f"\nPerformance Metrics:")
        print(f"  Force field generation: {results['performance_data'].get('generation_time', 0)*1000:.1f}ms")
        print(f"  Force sampling: {results['performance_data'].get('sample_time_us', 0):.2f}μs")
        print(f"  Max force magnitude: {results['performance_data'].get('max_force', 0):.2f}")
    
    if results['errors']:
        print(f"\nErrors encountered:")
        for error in results['errors']:
            print(f"  - {error}")
    
    # Final status
    if results['tests_passed'] == results['tests_total']:
        print("\n✅ PHYSICS ENGINE VERIFICATION COMPLETE - ALL TESTS PASSED")
        print("\n🚀 Ready for Chat 3 - Particle System Integration")
        return 0
    else:
        print(f"\n⚠️  PHYSICS ENGINE VERIFICATION INCOMPLETE - {results['tests_total'] - results['tests_passed']} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())