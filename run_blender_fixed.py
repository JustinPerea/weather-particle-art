#!/usr/bin/env python3
"""
Fixed Blender Integration Runner
Handles import path issues when running in Blender

Usage:
1. Open Blender
2. In Python Console: exec(open('/Users/justinperea/Documents/Art/Pulsaur/3D/Weather_API/weather_art_v3/run_blender_fixed.py').read())
"""

import sys
import os

# Add project root to Python path
project_root = "/Users/justinperea/Documents/Art/Pulsaur/3D/Weather_API/weather_art_v3"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also add src directory
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print("""
╔══════════════════════════════════════════════════════════════╗
║           Weather Particle Art - Blender Integration         ║
║                     Chat 4: GPU Rendering                    ║
╚══════════════════════════════════════════════════════════════╝
""")

# Check if we're in Blender
if "bpy" not in sys.modules:
    print("ERROR: This script must be run within Blender!")
    sys.exit(1)

import bpy

# Now try imports with better error handling
try:
    # Try absolute imports first
    from weather.weather_api import WeatherAPI
    from physics.physics_engine import PhysicsEngine
    from particles.viscous_particle_system import ViscousParticleSystem
except ImportError as e:
    print(f"Import error: {e}")
    print("\nChecking file structure...")
    
    # Debug: Check what files exist
    weather_path = os.path.join(src_path, "weather", "weather_api.py")
    physics_path = os.path.join(src_path, "physics", "physics_engine.py")
    particles_path = os.path.join(src_path, "particles", "viscous_particle_system.py")
    
    print(f"Weather API exists: {os.path.exists(weather_path)}")
    print(f"Physics Engine exists: {os.path.exists(physics_path)}")
    print(f"Particle System exists: {os.path.exists(particles_path)}")
    
    print("\nPlease ensure Chat 2 and Chat 3 files are in place.")
    sys.exit(1)

# Import Blender components
try:
    from blender.particle_renderer import BlenderParticleRenderer
    from blender.optimization_utils import PerformanceMonitor, BlenderMemoryOptimizer
except ImportError as e:
    print(f"Blender module import error: {e}")
    sys.exit(1)

print("✅ All modules imported successfully!")

# Simple test class if main imports fail
class SimpleParticleTest:
    """Minimal particle test without full integration"""
    
    def __init__(self, particle_count=100_000):
        self.particle_count = particle_count
        print(f"\n🌊 Creating simple particle test with {particle_count:,} particles...")
        
        # Just test the Blender renderer
        self.renderer = BlenderParticleRenderer(particle_count=particle_count)
        
        # Create dummy data
        import numpy as np
        self.positions = np.random.rand(particle_count, 3).astype(np.float32)
        self.positions[:, 0] *= 2.0  # Scale to container
        self.positions[:, 1] *= 2.0
        self.positions[:, 2] *= 1.0
        
        self.velocities = np.random.randn(particle_count, 3).astype(np.float32) * 0.01
        self.colors = np.random.rand(particle_count, 3).astype(np.float32)
        
        print("✅ Particle renderer initialized!")
    
    def update(self):
        """Update particles with simple motion"""
        # Simple physics
        self.positions += self.velocities
        
        # Bounce at boundaries
        self.positions[:, 0] = np.clip(self.positions[:, 0], 0, 2)
        self.positions[:, 1] = np.clip(self.positions[:, 1], 0, 2)
        self.positions[:, 2] = np.clip(self.positions[:, 2], 0, 1)
        
        # Update renderer
        self.renderer.update_particles(self.positions, self.velocities, self.colors)
    
    def run_test(self, frames=300):
        """Run simple animation test"""
        print(f"\n🎬 Running {frames} frame test...")
        
        import time
        for frame in range(frames):
            start = time.perf_counter()
            
            self.update()
            bpy.context.scene.frame_set(frame + 1)
            
            elapsed = (time.perf_counter() - start) * 1000
            
            if frame % 60 == 0:
                print(f"Frame {frame}: {elapsed:.2f}ms")
            
            # Update viewport
            if not bpy.app.background:
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        
        print("✅ Test complete!")

def setup_blender_scene():
    """Prepare Blender for particle rendering"""
    
    print("\n🔧 Setting up Blender scene...")
    
    # Clean existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Set viewport to rendered
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'RENDERED'
                    space.overlay.show_overlays = False
    
    print("✅ Scene prepared")

def run_simple_test():
    """Run a simple test without full integration"""
    
    setup_blender_scene()
    
    print("\n" + "="*60)
    print("Running Simple Particle Test (without weather/physics)")
    print("This tests just the Blender rendering component")
    print("="*60)
    
    # Create and run simple test
    test = SimpleParticleTest(particle_count=100_000)
    
    # Set up animation
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 300
    
    # Run test
    test.run_test(frames=300)
    
    # Store for console access
    bpy.particle_test = test
    
    print("\n💡 Access test object with: bpy.particle_test")
    print("💡 Update particles with: bpy.particle_test.update()")
    
    return test

def main():
    """Main entry point with error handling"""
    
    try:
        # Try full integration first
        from blender.blender_integration import WeatherParticleBlenderIntegration
        
        print("\n✅ Full integration available!")
        setup_blender_scene()
        
        # Create integration
        integration = WeatherParticleBlenderIntegration(
            particle_count=100_000,
            demo_mode=True
        )
        
        # Run animation
        print("\n🎬 Starting full weather particle animation...")
        print("Press SPACE to play/pause")
        
        # Store for console access
        bpy.integration = integration
        
        # Animate
        for frame in range(300):
            integration.update_frame(frame)
            if frame % 60 == 0:
                print(f"Frame {frame}/300")
        
        print("\n✅ Full integration running!")
        print("💡 Access with: bpy.integration")
        
        return integration
        
    except ImportError as e:
        print(f"\n⚠️  Full integration not available: {e}")
        print("Running simple Blender renderer test instead...")
        
        # Fall back to simple test
        return run_simple_test()

if __name__ == "__main__":
    # Run appropriate version
    result = main()
    
    # Frame the view
    if not bpy.app.background:
        bpy.ops.view3d.view_all()
        
    print("\n🎉 Ready!")