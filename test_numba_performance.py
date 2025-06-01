#!/usr/bin/env python3
"""
Test performance with Numba JIT compilation
"""

import subprocess
import sys
import time

print("=== NUMBA PERFORMANCE TEST ===\n")

# Check if numba is installed
try:
    import numba
    print(f"✅ Numba {numba.__version__} is installed")
except ImportError:
    print("❌ Numba not installed. Installing now...")
    print("This will significantly improve performance!\n")
    
    # Install numba
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numba"])
    print("\n✅ Numba installed successfully!")
    
    # Restart message
    print("\n⚠️  Please restart Python/terminal for changes to take effect")
    print("Then run this script again to test performance")
    sys.exit(0)

# Now test with numba
from src.weather.noaa_api import NOAAWeatherAPI
from src.physics.force_field_engine import PhysicsEngine
import numpy as np

def test_numba_performance():
    """Test force field sampling with Numba acceleration"""
    
    # Initialize
    weather_api = NOAAWeatherAPI()
    physics = PhysicsEngine()
    
    # Check if Numba methods exist
    if hasattr(physics, 'sample_forces_batch_numba'):
        print("\n✅ Numba-optimized methods found!")
    else:
        print("\n❌ Numba methods not found in PhysicsEngine")
        print("Please update src/physics/force_field_engine.py with Numba methods")
        return
    
    # Generate test data
    print("\nGenerating test data...")
    weather = weather_api.get_weather('storm')
    force_field = physics.generate_3d_force_field(weather)
    
    # Test with 1M particles
    positions = np.random.rand(1_000_000, 3).astype(np.float32) * physics.box_size
    
    print("\nWarming up Numba JIT compiler...")
    # First call compiles the function
    _ = physics.sample_forces_batch_numba(force_field, positions[:1000])
    
    print("\nTesting performance with 1M particles:")
    print("-" * 60)
    
    # Time standard batch method
    start = time.time()
    forces_standard = physics.sample_forces_batch(force_field, positions)
    time_standard = time.time() - start
    
    # Time Numba method
    start = time.time()
    forces_numba = physics.sample_forces_batch_numba(force_field, positions)
    time_numba = time.time() - start
    
    # Results
    print(f"Standard batch method: {time_standard*1000:.1f}ms")
    print(f"Numba JIT method:      {time_numba*1000:.1f}ms")
    print(f"Speedup:               {time_standard/time_numba:.1f}x")
    print("-" * 60)
    
    # Check 60 FPS feasibility
    print(f"\nFrame budget for 60 FPS: 16.67ms")
    print(f"Numba method uses: {time_numba*1000/16.67*100:.1f}% of frame budget")
    
    if time_numba * 1000 < 16.67:
        print("\n✅ SUCCESS! Can achieve 60 FPS with 1M particles!")
    else:
        print(f"\n⚠️  Still need {time_numba*1000/16.67:.1f}x more speed")
    
    # Verify accuracy
    max_diff = np.max(np.abs(forces_standard - forces_numba))
    print(f"\nMax difference between methods: {max_diff:.6f}")
    if max_diff < 1e-5:
        print("✅ Results match - Numba optimization is accurate")

if __name__ == "__main__":
    test_numba_performance()