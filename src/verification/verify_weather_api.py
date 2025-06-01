#!/usr/bin/env python3
"""
Weather API Verification Script
Tests NOAA API integration and generates verification outputs
"""

import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.weather.noaa_api import NOAAWeatherAPI, WeatherObservation

# Create output directory in GitHub structure
OUTPUT_DIR = Path("verification_outputs/chat_2_physics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Also create debug_plots directory
DEBUG_DIR = Path("debug_plots")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

def test_weather_api():
    """Comprehensive test of weather API functionality"""
    print("=== WEATHER API VERIFICATION ===\n")
    
    api = NOAAWeatherAPI(cache_dir="weather_cache")
    results = {
        'tests_passed': 0,
        'tests_total': 0,
        'performance_data': {},
        'errors': []
    }
    
    # Test 1: Demo patterns availability
    print("1. Testing demo weather patterns...")
    results['tests_total'] += 1
    try:
        patterns = ['calm', 'storm', 'heat_wave', 'fog', 'hurricane']
        for pattern in patterns:
            weather = api.get_weather(pattern)
            assert isinstance(weather, WeatherObservation)
            print(f"   ✓ {pattern}: temp={weather.temperature}°C, pressure={weather.pressure}hPa")
        results['tests_passed'] += 1
        print("   ✅ All demo patterns working\n")
    except Exception as e:
        print(f"   ❌ Demo patterns failed: {e}\n")
        results['errors'].append(f"Demo patterns: {str(e)}")
    
    # Test 2: API performance
    print("2. Testing API performance...")
    results['tests_total'] += 1
    try:
        # Test with Baltimore coordinates
        location = "39.2904,-76.6122"
        
        # First call (should hit API or use cache)
        start = time.time()
        weather1 = api.get_weather(location)
        time1 = time.time() - start
        
        # Second call (should use cache)
        start = time.time()
        weather2 = api.get_weather(location)
        time2 = time.time() - start
        
        results['performance_data']['first_call'] = time1
        results['performance_data']['cached_call'] = time2
        
        print(f"   First call: {time1:.3f}s")
        print(f"   Cached call: {time2:.3f}s")
        
        # Performance requirement: <1s for weather fetch
        if time1 < 1.0:
            results['tests_passed'] += 1
            print("   ✅ Performance requirement met (<1s)\n")
        else:
            print("   ⚠️  Performance slower than 1s target\n")
            
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}\n")
        results['errors'].append(f"Performance: {str(e)}")
    
    # Test 3: Weather data validation
    print("3. Testing weather data validation...")
    results['tests_total'] += 1
    try:
        weather = api.get_weather('storm')
        
        # Check all required fields
        assert 0 <= weather.temperature <= 50, f"Temperature out of range: {weather.temperature}"
        assert 900 <= weather.pressure <= 1100, f"Pressure out of range: {weather.pressure}"
        assert 0 <= weather.humidity <= 100, f"Humidity out of range: {weather.humidity}"
        assert 0 <= weather.wind_speed <= 100, f"Wind speed out of range: {weather.wind_speed}"
        assert 0 <= weather.wind_direction <= 360, f"Wind direction out of range: {weather.wind_direction}"
        assert 0 <= weather.uv_index <= 15, f"UV index out of range: {weather.uv_index}"
        assert 0 <= weather.cloud_cover <= 100, f"Cloud cover out of range: {weather.cloud_cover}"
        assert 0 <= weather.precipitation <= 100, f"Precipitation out of range: {weather.precipitation}"
        assert 0 <= weather.visibility <= 50, f"Visibility out of range: {weather.visibility}"
        
        # Check computed properties
        assert weather.temperature_kelvin > 0
        assert weather.pressure_pascals > 0
        assert len(weather.wind_vector) == 3
        
        results['tests_passed'] += 1
        print("   ✅ All weather data fields valid\n")
        
    except AssertionError as e:
        print(f"   ❌ Validation failed: {e}\n")
        results['errors'].append(f"Validation: {str(e)}")
    
    # Test 4: Weather interpolation
    print("4. Testing weather interpolation...")
    results['tests_total'] += 1
    try:
        sequence = api.get_weather_sequence('storm_approach')
        assert len(sequence) == 4
        
        # Check smooth transitions
        temps = [w.temperature for w in sequence]
        pressures = [w.pressure for w in sequence]
        
        assert temps[0] > temps[-1], "Temperature should decrease in storm approach"
        assert pressures[0] > pressures[-1], "Pressure should drop in storm approach"
        
        results['tests_passed'] += 1
        print("   ✅ Weather interpolation working\n")
        
    except Exception as e:
        print(f"   ❌ Interpolation failed: {e}\n")
        results['errors'].append(f"Interpolation: {str(e)}")
    
    # Test 5: Offline fallback
    print("5. Testing offline fallback...")
    results['tests_total'] += 1
    try:
        # Test with invalid location that will fail
        weather = api.get_weather("invalid_location_xyz")
        assert isinstance(weather, WeatherObservation)
        results['tests_passed'] += 1
        print("   ✅ Offline fallback working\n")
        
    except Exception as e:
        print(f"   ❌ Offline fallback failed: {e}\n")
        results['errors'].append(f"Offline fallback: {str(e)}")
    
    return results, api

def create_weather_visualization(api: NOAAWeatherAPI):
    """Create visualization of weather patterns"""
    print("Creating weather pattern visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Weather Pattern Verification', fontsize=16)
    
    patterns = ['calm', 'storm', 'heat_wave', 'fog', 'hurricane']
    colors = ['green', 'darkblue', 'red', 'gray', 'purple']
    
    for idx, (pattern, color) in enumerate(zip(patterns, colors)):
        weather = api.get_weather(pattern)
        
        # Create polar plot showing all parameters
        ax = plt.subplot(2, 3, idx + 1, projection='polar')
        
        # Parameters to plot
        params = {
            'Temperature': weather.temperature / 40,  # Normalize to 0-1
            'Pressure': (weather.pressure - 950) / 100,  # Normalize
            'Humidity': weather.humidity / 100,
            'Wind Speed': weather.wind_speed / 50,
            'UV Index': weather.uv_index / 11,
            'Cloud Cover': weather.cloud_cover / 100,
            'Precipitation': min(weather.precipitation / 20, 1),
            'Visibility': weather.visibility / 15
        }
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(params), endpoint=False).tolist()
        values = list(params.values())
        
        # Close the plot
        angles += angles[:1]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
        ax.set_ylim(0, 1)
        
        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(params.keys(), size=8)
        ax.set_title(f'{pattern.upper()}\nT={weather.temperature}°C, P={weather.pressure}hPa', 
                     pad=20, fontsize=12, fontweight='bold')
        
        # Wind vector indication
        wind_angle = np.radians(weather.wind_direction)
        ax.arrow(wind_angle, 0, 0, weather.wind_speed / 50, 
                head_width=0.1, head_length=0.05, fc=color, ec=color)
    
    # Create time series plot
    ax = plt.subplot(2, 3, 6)
    sequence = api.get_weather_sequence('storm_approach')
    
    times = range(len(sequence))
    temps = [w.temperature for w in sequence]
    pressures = [w.pressure for w in sequence]
    winds = [w.wind_speed for w in sequence]
    
    ax2 = ax.twinx()
    
    line1 = ax.plot(times, temps, 'r-', linewidth=2, label='Temperature (°C)')
    line2 = ax2.plot(times, pressures, 'b-', linewidth=2, label='Pressure (hPa)')
    line3 = ax.plot(times, winds, 'g--', linewidth=2, label='Wind (m/s)')
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Temperature (°C) / Wind (m/s)', color='red')
    ax2.set_ylabel('Pressure (hPa)', color='blue')
    ax.set_title('Storm Approach Sequence')
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center left')
    
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='y', labelcolor='red')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / "weather_patterns_verification.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    
    # Also save to debug_plots
    debug_path = DEBUG_DIR / "weather_patterns_verification.png"
    plt.savefig(debug_path, dpi=300, bbox_inches='tight')
    print(f"   Also saved to: {debug_path}")
    
    return fig

def save_test_results(results: dict, api: NOAAWeatherAPI):
    """Save test results and sample data"""
    
    # Save test results
    results_path = OUTPUT_DIR / "weather_api_test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Saved test results: {results_path}")
    
    # Save sample weather data for Chat 3
    sample_data = {}
    for pattern in ['calm', 'storm', 'heat_wave']:
        weather = api.get_weather(pattern)
        sample_data[pattern] = weather.to_dict()
    
    sample_path = OUTPUT_DIR / "sample_weather_data.json"
    with open(sample_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    print(f"   Saved sample data: {sample_path}")

def main():
    """Run all weather API verification tests"""
    
    # Run tests
    results, api = test_weather_api()
    
    # Create visualizations
    create_weather_visualization(api)
    
    # Save results
    save_test_results(results, api)
    
    # Summary
    print("\n=== VERIFICATION SUMMARY ===")
    print(f"Tests passed: {results['tests_passed']}/{results['tests_total']}")
    
    if results['performance_data']:
        print(f"\nPerformance:")
        for key, value in results['performance_data'].items():
            print(f"  {key}: {value:.3f}s")
    
    if results['errors']:
        print(f"\nErrors encountered:")
        for error in results['errors']:
            print(f"  - {error}")
    
    # Final status
    if results['tests_passed'] == results['tests_total']:
        print("\n✅ WEATHER API VERIFICATION COMPLETE - ALL TESTS PASSED")
        return 0
    else:
        print(f"\n⚠️  WEATHER API VERIFICATION INCOMPLETE - {results['tests_total'] - results['tests_passed']} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())