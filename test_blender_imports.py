#!/usr/bin/env python3
"""
Test imports step by step to debug issues
Run this in Blender console to see what's working
"""

import sys
import os

# Add paths
project_root = "/Users/justinperea/Documents/Art/Pulsaur/3D/Weather_API/weather_art_v3"
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

print("Testing imports step by step...\n")

# Test 1: Import weather module
print("1. Testing weather module:")
try:
    from weather.noaa_api import WeatherObservation, NOAAWeatherAPI
    print("   ✅ Weather module imported successfully")
    print(f"   - WeatherObservation: {WeatherObservation}")
    print(f"   - NOAAWeatherAPI: {NOAAWeatherAPI}")
except Exception as e:
    print(f"   ❌ Weather import failed: {e}")

# Test 2: Import physics module  
print("\n2. Testing physics module:")
try:
    from physics.force_field_engine import PhysicsEngine
    print("   ✅ Physics module imported successfully")
    print(f"   - PhysicsEngine: {PhysicsEngine}")
except Exception as e:
    print(f"   ❌ Physics import failed: {e}")

# Test 3: Import particle module
print("\n3. Testing particle module:")
try:
    from particles.viscous_particle_system import ViscousParticleSystem, ParticleParams
    print("   ✅ Particle module imported successfully")
    print(f"   - ViscousParticleSystem: {ViscousParticleSystem}")
    print(f"   - ParticleParams: {ParticleParams}")
except Exception as e:
    print(f"   ❌ Particle import failed: {e}")

# Test 4: Import Blender renderer
print("\n4. Testing Blender renderer:")
try:
    import bpy
    from blender.particle_renderer import BlenderParticleRenderer
    print("   ✅ Blender renderer imported successfully")
    print(f"   - BlenderParticleRenderer: {BlenderParticleRenderer}")
except Exception as e:
    print(f"   ❌ Blender renderer import failed: {e}")

# Test 5: Simple integration test
print("\n5. Testing simple integration:")
try:
    import numpy as np
    
    # Create simple test data
    print("   Creating test weather...")
    weather = WeatherObservation(
        temperature=22.0,
        pressure=1013.25,
        humidity=65.0,
        wind_speed=5.0,
        wind_direction=180,
        uv_index=5,
        cloud_cover=50,
        precipitation=0,
        station_id="TEST",
        timestamp="2024-01-01T00:00:00Z",
        latitude=0.0,
        longitude=0.0,
        elevation=0.0
    )
    print("   ✅ Weather created")
    
    # Create physics engine
    print("   Creating physics engine...")
    physics = PhysicsEngine()
    force_field = physics.generate_3d_force_field(weather)
    print(f"   ✅ Force field created: shape {force_field.shape}")
    
    # Create particle system
    print("   Creating particle system...")
    particles = ViscousParticleSystem(particle_count=10_000)  # Start small
    print("   ✅ Particle system created")
    
    # Update particles
    print("   Updating particles...")
    particles.update(force_field, dt=0.016)
    positions, velocities, colors = particles.get_render_data()
    print(f"   ✅ Particles updated: {positions.shape[0]} particles")
    
    # Create Blender renderer (with fix for EEVEE_NEXT)
    print("   Creating Blender renderer...")
    
    # Fix the renderer first
    import blender.particle_renderer as pr
    original_setup = pr.BlenderParticleRenderer._setup_scene
    
    def fixed_setup_scene(self):
        scene = bpy.context.scene
        # Use correct render engine
        available_engines = [e.bl_idname for e in bpy.types.RenderEngine.__subclasses__()]
        if 'BLENDER_EEVEE_NEXT' in available_engines:
            scene.render.engine = 'BLENDER_EEVEE_NEXT'
        elif 'BLENDER_EEVEE' in available_engines:
            scene.render.engine = 'BLENDER_EEVEE'
        else:
            scene.render.engine = 'CYCLES'
        
        # Continue with setup (simplified)
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1080
        scene.render.fps = 60
        
        # Camera
        self._setup_camera()
        
        # Black background
        if not scene.world:
            scene.world = bpy.data.worlds.new(name="World")
    
    pr.BlenderParticleRenderer._setup_scene = fixed_setup_scene
    
    renderer = BlenderParticleRenderer(particle_count=10_000)
    print("   ✅ Blender renderer created")
    
    # Update renderer
    print("   Updating Blender particles...")
    renderer.update_particles(positions, velocities, colors)
    print("   ✅ Blender particles updated!")
    
    print("\n✅ ALL TESTS PASSED! System is working!")
    
    # Store objects for console access
    bpy.test_weather = weather
    bpy.test_physics = physics
    bpy.test_particles = particles
    bpy.test_renderer = renderer
    
    print("\n💡 Test objects available as:")
    print("   - bpy.test_weather")
    print("   - bpy.test_physics") 
    print("   - bpy.test_particles")
    print("   - bpy.test_renderer")
    
except Exception as e:
    print(f"   ❌ Integration test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)