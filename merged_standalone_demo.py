#!/usr/bin/env python3
"""
Standalone Blender Weather Particle Demo with Anadol Materials
Complete working system combining Chat 4 foundation + Chat 5 aesthetics
"""

import bpy
import numpy as np
import time
import sys
import os
from datetime import datetime

# Add paths
project_root = "/Users/justinperea/Documents/Art/Pulsaur/3D/Weather_API/weather_art_v3"
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

print("""
╔══════════════════════════════════════════════════════════════╗
║     Standalone Weather Particle Demo with Anadol Materials   ║
║              Chat 4 Foundation + Chat 5 Aesthetics           ║
╚══════════════════════════════════════════════════════════════╝
""")

# Import weather and physics (these work)
from weather.noaa_api import WeatherObservation
from physics.force_field_engine import PhysicsEngine


class AnadolMaterialSystem:
    """Weather-responsive HDR material system"""
    
    def __init__(self):
        self.material = None
        self.material_nodes = {}
        self.update_times = []
        
    def create_particle_material(self, weather_data):
        """Create weather-responsive HDR particle material"""
        print("   Creating Anadol material...")
        
        # Create new material
        self.material = bpy.data.materials.new(name="AnadolParticleMaterial")
        self.material.use_nodes = True
        
        # Clear default nodes
        nodes = self.material.node_tree.nodes
        nodes.clear()
        
        # === OUTPUT NODE ===
        output_node = nodes.new('ShaderNodeOutputMaterial')
        output_node.location = (800, 0)
        
        # === EMISSION SHADER (particles ARE light) ===
        emission_node = nodes.new('ShaderNodeEmission')
        emission_node.location = (600, 0)
        self.material_nodes['emission'] = emission_node
        
        # === COLOR TEMPERATURE → RGB ===
        blackbody_node = nodes.new('ShaderNodeBlackbody')
        blackbody_node.location = (400, 100)
        self.material_nodes['blackbody'] = blackbody_node
        
        # === WEATHER-DRIVEN PARAMETERS ===
        
        # Temperature → Color Temperature
        temp_map = nodes.new('ShaderNodeMapRange')
        temp_map.location = (200, 100)
        temp_map.inputs['From Min'].default_value = -20
        temp_map.inputs['From Max'].default_value = 40
        temp_map.inputs['To Min'].default_value = 8000  # Cold = blue
        temp_map.inputs['To Max'].default_value = 2000  # Hot = red
        self.material_nodes['temp_map'] = temp_map
        
        # Pressure → Emission Intensity
        pressure_map = nodes.new('ShaderNodeMapRange')
        pressure_map.location = (200, -100)
        pressure_map.inputs['From Min'].default_value = 980
        pressure_map.inputs['From Max'].default_value = 1040
        pressure_map.inputs['To Min'].default_value = 0.5
        pressure_map.inputs['To Max'].default_value = 3.0
        self.material_nodes['pressure_map'] = pressure_map
        
        # UV Index → Brightness
        uv_map = nodes.new('ShaderNodeMapRange')
        uv_map.location = (200, -300)
        uv_map.inputs['From Min'].default_value = 0
        uv_map.inputs['From Max'].default_value = 11
        uv_map.inputs['To Min'].default_value = 0.5
        uv_map.inputs['To Max'].default_value = 3.0
        self.material_nodes['uv_map'] = uv_map
        
        # HDR multiplier
        hdr_mult = nodes.new('ShaderNodeMath')
        hdr_mult.operation = 'MULTIPLY'
        hdr_mult.location = (400, -200)
        hdr_mult.inputs[1].default_value = 2.0  # Base HDR boost
        self.material_nodes['hdr_mult'] = hdr_mult
        
        # Final strength combiner
        strength_mult = nodes.new('ShaderNodeMath')
        strength_mult.operation = 'MULTIPLY'
        strength_mult.location = (600, -100)
        self.material_nodes['strength_mult'] = strength_mult
        
        # === CONNECT NODES ===
        links = self.material.node_tree.links
        
        # Temperature → Color
        links.new(temp_map.outputs['Result'], blackbody_node.inputs['Temperature'])
        links.new(blackbody_node.outputs['Color'], emission_node.inputs['Color'])
        
        # Pressure → HDR multiplier
        links.new(pressure_map.outputs['Result'], hdr_mult.inputs[0])
        
        # UV → Final strength
        links.new(uv_map.outputs['Result'], strength_mult.inputs[0])
        links.new(hdr_mult.outputs['Value'], strength_mult.inputs[1])
        
        # Final emission strength
        links.new(strength_mult.outputs['Value'], emission_node.inputs['Strength'])
        
        # Connect to output
        links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])
        
        # Set initial weather values
        self.update_material_parameters(weather_data)
        
        print("   ✅ Anadol material created")
        return self.material
    
    def update_material_parameters(self, weather_data):
        """Update material with new weather data"""
        if not self.material or not self.material_nodes:
            return
            
        start = time.perf_counter()
        
        # Update weather-driven values
        self.material_nodes['temp_map'].inputs['Value'].default_value = weather_data.temperature
        self.material_nodes['pressure_map'].inputs['Value'].default_value = weather_data.pressure
        self.material_nodes['uv_map'].inputs['Value'].default_value = weather_data.uv_index
        
        elapsed = time.perf_counter() - start
        self.update_times.append(elapsed * 1000)
        
    def setup_compositor_effects(self):
        """Setup HDR bloom and post-processing"""
        scene = bpy.context.scene
        scene.use_nodes = True
        
        # Note: In Blender 4.0+, bloom is handled differently
        # We'll use compositor nodes for glow effects
        
        # Setup compositor nodes
        tree = scene.node_tree
        nodes = tree.nodes
        nodes.clear()
        
        # Render layers
        render_layers = nodes.new('CompositorNodeRLayers')
        render_layers.location = (0, 0)
        
        # Glare node for bloom/glow effect
        glare = nodes.new('CompositorNodeGlare')
        glare.location = (200, 0)
        glare.glare_type = 'GHOSTS'
        glare.threshold = 0.8
        glare.mix = 0.3
        
        # Composite output
        composite = nodes.new('CompositorNodeComposite')
        composite.location = (400, 0)
        
        # Connect nodes
        links = tree.links
        links.new(render_layers.outputs['Image'], glare.inputs['Image'])
        links.new(glare.outputs['Image'], composite.inputs['Image'])
        
        print("   ✅ HDR bloom configured")


class EnhancedParticleRenderer:
    """Particle renderer with Anadol materials"""
    
    def __init__(self, particle_count=100_000):
        self.particle_count = particle_count
        self.update_times = []
        self.material_system = AnadolMaterialSystem()
        
        print(f"Initializing enhanced renderer for {particle_count:,} particles...")
        
        # Clean scene
        self._clean_scene()
        
        # Create particle mesh
        self._create_particle_mesh()
        
        # Setup camera and world
        self._setup_camera()
        self._setup_world()
        
        # Setup rendering
        self._setup_rendering()
        
        print("✅ Enhanced renderer initialized")
    
    def _clean_scene(self):
        """Remove all objects"""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # Clear orphan data
        for mesh in bpy.data.meshes:
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
    
    def _create_particle_mesh(self):
        """Create mesh for particles"""
        # Create mesh
        self.mesh = bpy.data.meshes.new(name="ParticleCloud")
        self.obj = bpy.data.objects.new("ParticleCloud", self.mesh)
        bpy.context.collection.objects.link(self.obj)
        
        # Add vertices
        self.mesh.vertices.add(self.particle_count)
        self.mesh.update()
        
        print(f"✅ Created mesh with {self.particle_count:,} vertices")
    
    def _setup_camera(self):
        """Create and position camera"""
        # Create camera
        cam_data = bpy.data.cameras.new("Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        bpy.context.collection.objects.link(cam)
        
        # Gallery viewing angle
        cam.location = (1, -5, 2.5)
        cam.rotation_euler = (1.2, 0, 0)
        
        # Set as active
        bpy.context.scene.camera = cam
    
    def _setup_world(self):
        """Pure black background - Anadol signature"""
        world = bpy.data.worlds.get("World")
        if not world:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        
        world.use_nodes = True
        nodes = world.node_tree.nodes
        
        # Pure black background
        bg = nodes.get("Background")
        if bg:
            bg.inputs[0].default_value = (0, 0, 0, 1)  # Pure black
            bg.inputs[1].default_value = 0  # No emission
    
    def _setup_rendering(self):
        """Configure render settings"""
        scene = bpy.context.scene
        
        # Use EEVEE for real-time
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1080
        scene.render.fps = 60
        
        # Quality settings
        scene.eevee.taa_samples = 16
        # Note: Bloom is handled differently in Blender 4.0
    
    def apply_anadol_material(self, weather_data):
        """Apply weather-responsive material to particles"""
        # Create material
        material = self.material_system.create_particle_material(weather_data)
        
        # Apply to base particle for geometry instances
        if hasattr(self, 'base_particle'):
            self.base_particle.data.materials.clear()
            self.base_particle.data.materials.append(material)
        
        # Also apply to mesh (for viewport display)
        if self.obj.data.materials:
            self.obj.data.materials[0] = material
        else:
            self.obj.data.materials.append(material)
        
        # Setup compositor
        self.material_system.setup_compositor_effects()
        
        # Make particles render as points in viewport
        self.obj.display_type = 'WIRE'
        self.obj.show_in_front = False
    
    def update_particles(self, positions, weather_data=None):
        """Update particle positions and materials"""
        start = time.perf_counter()
        
        # Update positions
        self.coord_array = positions.flatten(order='C')
        self.mesh.vertices.foreach_set("co", self.coord_array)
        self.mesh.update_tag()
        
        # Update material if weather provided
        if weather_data:
            self.material_system.update_material_parameters(weather_data)
        
        # Track performance
        elapsed = time.perf_counter() - start
        self.update_times.append(elapsed)
        
        if len(self.update_times) % 60 == 0:
            avg_ms = np.mean(self.update_times[-60:]) * 1000
            mat_ms = np.mean(self.material_system.update_times[-60:]) if self.material_system.update_times else 0
            print(f"  Render: {avg_ms:.2f}ms | Material: {mat_ms:.2f}ms")


class SimpleParticleSystem:
    """Minimal particle system with viscous dynamics"""
    
    def __init__(self, particle_count=100_000):
        self.particle_count = particle_count
        
        # Initialize in box with some variation
        self.positions = np.random.rand(particle_count, 3).astype(np.float32)
        self.positions[:, 0] = self.positions[:, 0] * 3.6 - 1.8  # X: -1.8 to 1.8
        self.positions[:, 1] = self.positions[:, 1] * 3.6 - 1.8  # Y: -1.8 to 1.8
        self.positions[:, 2] = self.positions[:, 2] * 1.7 + 0.1  # Z: 0.1 to 1.8
        
        self.velocities = np.zeros((particle_count, 3), dtype=np.float32)
        
        # Viscous parameters (Anadol-style)
        self.viscosity = 100.0
        self.damping = 0.95
        self.max_speed = 0.1
    
    def update(self, force_field, dt=0.016):
        """Viscous particle update"""
        # Sample force field at particle positions
        # For demo, use simplified forces
        angle = time.time() * 0.5
        forces = np.zeros((self.particle_count, 3), dtype=np.float32)
        
        # Swirling forces
        forces[:, 0] = -self.positions[:, 1] * 0.3 + np.sin(angle) * 0.1
        forces[:, 1] = self.positions[:, 0] * 0.3 + np.cos(angle) * 0.1
        forces[:, 2] = np.sin(self.positions[:, 0] * 2) * 0.1
        
        # Viscous dynamics
        self.velocities += (forces / self.viscosity) * dt
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
        
        # Soft boundaries
        for dim in range(3):
            if dim < 2:  # X, Y
                mask = self.positions[:, dim] < -1.8
                self.positions[mask, dim] = -1.8
                self.velocities[mask, dim] *= -0.5
                
                mask = self.positions[:, dim] > 1.8
                self.positions[mask, dim] = 1.8
                self.velocities[mask, dim] *= -0.5
            else:  # Z
                mask = self.positions[:, 2] < 0.1
                self.positions[mask, 2] = 0.1
                self.velocities[mask, 2] *= -0.5
                
                mask = self.positions[:, 2] > 1.8
                self.positions[mask, 2] = 1.8
                self.velocities[mask, 2] *= -0.5


# Main demo
print("\n=== CREATING ENHANCED DEMO SYSTEM ===")

# 1. Weather
print("\n1. Creating weather...")
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

# 2. Physics
print("\n2. Generating force field...")
physics = PhysicsEngine()
force_field = physics.generate_3d_force_field(weather)
print(f"✅ Force field: {force_field.shape}")

# 3. Particles
print("\n3. Creating particles...")
particles = SimpleParticleSystem(100_000)
print("✅ Particles created")

# 4. Enhanced Renderer with Anadol Materials
print("\n4. Creating enhanced renderer...")
renderer = EnhancedParticleRenderer(100_000)
renderer.apply_anadol_material(weather)
print("✅ Enhanced renderer created with Anadol materials")

# 5. Test animation
print("\n5. Running test animation with weather-responsive materials...")
print("   (Check your 3D viewport - particles should glow!)")

# Animate with weather variations
start_time = time.time()
for frame in range(240):  # 4 seconds
    # Vary weather over time
    t = frame / 240.0
    weather.temperature = 22 + 10 * np.sin(t * np.pi * 2)  # 12°C to 32°C
    weather.pressure = 1013 + 20 * np.cos(t * np.pi * 2)  # 993 to 1033 hPa
    weather.uv_index = int(5 + 4 * np.sin(t * np.pi))  # 1 to 9
    
    # Update physics
    particles.update(force_field)
    
    # Update renderer with weather
    renderer.update_particles(particles.positions, weather)
    
    # Update frame
    bpy.context.scene.frame_set(frame + 1)
    
    # Progress
    if frame % 30 == 0:
        elapsed = time.time() - start_time
        fps = (frame + 1) / elapsed if elapsed > 0 else 0
        print(f"   Frame {frame}: {fps:.1f} FPS | Temp: {weather.temperature:.1f}°C | Pressure: {weather.pressure:.0f} hPa")
    
    # Update viewport
    if frame % 10 == 0:
        bpy.context.view_layer.update()

# Calculate final performance
total_time = time.time() - start_time
avg_fps = 240 / total_time
avg_render = np.mean(renderer.update_times) * 1000 if renderer.update_times else 0
avg_material = np.mean(renderer.material_system.update_times) if renderer.material_system.update_times else 0

print("\n" + "="*60)
print("ENHANCED PERFORMANCE SUMMARY")
print("="*60)
print(f"Total frames: 240")
print(f"Total time: {total_time:.2f}s")
print(f"Average FPS: {avg_fps:.1f}")
print(f"Average render update: {avg_render:.2f}ms")
print(f"Average material update: {avg_material:.2f}ms")
print(f"Total update: {avg_render + avg_material:.2f}ms")
print(f"Render target met (<4.95ms): {'YES' if avg_render < 4.95 else 'NO'}")
print(f"Material target met (<3.10ms): {'YES' if avg_material < 3.10 else 'NO'}")
print("="*60)

# Store for console
bpy.test_weather = weather
bpy.test_physics = physics  
bpy.test_particles = particles
bpy.test_renderer = renderer

print("\n✅ SUCCESS! Enhanced demo with Anadol materials complete!")
print("\n💡 Objects available in console:")
print("   bpy.test_weather    - Weather data")
print("   bpy.test_physics    - Physics engine")
print("   bpy.test_particles  - Particle system")
print("   bpy.test_renderer   - Enhanced renderer with materials")

print("\n🎬 To continue animating with weather changes:")
print("   for i in range(100):")
print("       bpy.test_weather.temperature = 22 + i * 0.1")
print("       bpy.test_particles.update(bpy.test_physics.generate_3d_force_field(bpy.test_weather))")
print("       bpy.test_renderer.update_particles(bpy.test_particles.positions, bpy.test_weather)")
print("       bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)")

print("\n🎨 Material responds to:")
print("   - Temperature → Color (blue=cold, red=hot)")
print("   - Pressure → Brightness intensity")
print("   - UV Index → Overall emission strength")