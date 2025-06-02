#!/usr/bin/env python3
"""
Simple Blender Test - Minimal working demo without matplotlib dependencies
This creates a working particle system in Blender without needing the full integration
"""

import sys
import os
import numpy as np
import time

# Add paths
project_root = "/Users/justinperea/Documents/Art/Pulsaur/3D/Weather_API/weather_art_v3"
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

print("""
╔══════════════════════════════════════════════════════════════╗
║         Simplified Weather Particle System for Blender       ║
║                  (Without matplotlib dependencies)           ║
╚══════════════════════════════════════════════════════════════╝
""")

# Import what we can
import bpy
from weather.noaa_api import WeatherObservation, NOAAWeatherAPI
from physics.force_field_engine import PhysicsEngine

# Create a simplified particle system that doesn't need matplotlib
class SimpleViscousParticles:
    """Simplified particle system without matplotlib dependencies"""
    
    def __init__(self, particle_count=100_000):
        self.particle_count = particle_count
        
        # Initialize particles in a shallow box [0,2] x [0,2] x [0,1]
        self.positions = np.random.rand(particle_count, 3).astype(np.float32)
        self.positions[:, 0] *= 2.0  # X: 0 to 2
        self.positions[:, 1] *= 2.0  # Y: 0 to 2
        self.positions[:, 2] *= 1.0  # Z: 0 to 1
        
        # Start with zero velocity
        self.velocities = np.zeros((particle_count, 3), dtype=np.float32)
        
        # Random colors
        self.colors = np.random.rand(particle_count, 3).astype(np.float32)
        
        # Viscous parameters (from Anadol aesthetic)
        self.viscosity = 100.0
        self.damping = 0.95
        self.max_speed = 0.1
        
        print(f"✅ Created {particle_count:,} viscous particles")
    
    def update(self, force_field, dt=0.016):
        """Update particles with viscous dynamics"""
        
        # Simple force sampling (without the optimized batch sampling)
        # For demo purposes, use a simpler approach
        forces = np.zeros_like(self.velocities)
        
        # Sample forces at particle positions (simplified)
        # This is less efficient but works for demo
        grid_shape = force_field.shape[:3]
        for i in range(min(1000, self.particle_count)):  # Sample subset for speed
            # Map position to grid
            grid_pos = self.positions[i] / np.array([2.0, 2.0, 1.0]) * (np.array(grid_shape) - 1)
            grid_idx = np.clip(grid_pos.astype(int), 0, np.array(grid_shape) - 1)
            
            # Get force at grid point
            forces[i] = force_field[grid_idx[0], grid_idx[1], grid_idx[2]]
        
        # Apply forces with extreme viscosity
        acceleration = forces / self.viscosity
        
        # Update velocities with damping
        self.velocities += acceleration * dt
        self.velocities *= self.damping
        
        # Limit speed
        speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        self.velocities = np.where(
            speeds > self.max_speed,
            self.velocities * self.max_speed / (speeds + 1e-6),
            self.velocities
        )
        
        # Update positions
        self.positions += self.velocities * dt
        
        # Boundary conditions (soft aquarium-style)
        for dim in range(3):
            max_val = 2.0 if dim < 2 else 1.0
            
            # Approaching boundaries
            near_min = self.positions[:, dim] < 0.1
            near_max = self.positions[:, dim] > (max_val - 0.1)
            
            # Slow down near boundaries
            self.velocities[near_min | near_max, dim] *= 0.8
            
            # Soft clamping
            self.positions[:, dim] = np.clip(self.positions[:, dim], 0.0, max_val)
    
    def get_render_data(self):
        """Get data for rendering"""
        return self.positions, self.velocities, self.colors


# Fix and import Blender renderer
print("\n1. Fixing Blender renderer for version 4.0+...")

import blender.particle_renderer as pr

# Patch the setup method
def fixed_setup_scene(self):
    scene = bpy.context.scene
    
    # Use correct render engine - simpler approach
    try:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    except:
        try:
            scene.render.engine = 'BLENDER_EEVEE'
        except:
            scene.render.engine = 'CYCLES'
    
    # Resolution
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.fps = 60
    
    # Camera
    self._setup_camera()
    
    # Black background
    if not scene.world:
        scene.world = bpy.data.worlds.new(name="World")
    
    scene.world.use_nodes = True
    if "Background" in scene.world.node_tree.nodes:
        bg_node = scene.world.node_tree.nodes["Background"]
        bg_node.inputs[0].default_value = (0, 0, 0, 1)

pr.BlenderParticleRenderer._setup_scene = fixed_setup_scene

from blender.particle_renderer import BlenderParticleRenderer

print("✅ Blender renderer patched for version 4.0+")

# Create simple demo
print("\n2. Creating demo weather observation...")

# Create weather with all required fields
from datetime import datetime

weather = WeatherObservation(
    timestamp=datetime.now().isoformat(),
    temperature=22.0,
    pressure=1013.25,
    humidity=65.0,
    wind_speed=5.0,
    wind_direction=180,
    uv_index=5,
    cloud_cover=50,
    precipitation=0,
    visibility=10.0
)
print("✅ Weather created")

# Create physics engine
print("\n3. Generating force field from weather...")
physics = PhysicsEngine()
force_field = physics.generate_3d_force_field(weather)
print(f"✅ Force field created: shape {force_field.shape}")

# Create particle system
print("\n4. Creating particle system...")
particles = SimpleViscousParticles(particle_count=100_000)

# Create Blender renderer
print("\n5. Setting up Blender renderer...")
renderer = BlenderParticleRenderer(particle_count=100_000)

# Animation function
def update_animation(frame):
    """Update one frame of animation"""
    # Update particles
    particles.update(force_field, dt=0.016)
    
    # Get render data
    positions, velocities, colors = particles.get_render_data()
    
    # Update Blender
    renderer.update_particles(positions, velocities, colors)
    
    # Set frame
    bpy.context.scene.frame_set(frame)

# Run test animation
print("\n6. Running test animation...")

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 300  # 5 seconds

# Animate
start_time = time.time()
for frame in range(1, 61):  # First second
    update_animation(frame)
    
    if frame % 30 == 0:
        elapsed = time.time() - start_time
        fps = frame / elapsed
        print(f"   Frame {frame}: {fps:.1f} FPS")

print("\n✅ SUCCESS! Particle system is running!")

# Store objects for console access
bpy.demo_weather = weather
bpy.demo_physics = physics
bpy.demo_particles = particles
bpy.demo_renderer = renderer

print("\n💡 Objects available in console:")
print("   - bpy.demo_weather   (weather observation)")
print("   - bpy.demo_physics   (physics engine)")
print("   - bpy.demo_particles (particle system)")
print("   - bpy.demo_renderer  (Blender renderer)")

print("\n🎬 To continue animation:")
print("   for frame in range(61, 301):")
print("       update_animation(frame)")

print("\n📊 To benchmark performance:")
print("   bpy.demo_renderer.benchmark_performance()")

print("\n" + "="*60)