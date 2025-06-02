#!/usr/bin/env python3
"""
Chat 5: Anadol-Inspired Weather-Responsive Material System
Self-illuminating particles with HDR emission and weather-driven aesthetics
Performance target: <3.10ms for material updates
Updated with geometry-based particle rendering for visible emission
"""

import bpy
import numpy as np
import time
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path
import os
try:
    # If running from file
    project_root = Path(__file__).parent.parent.parent
except NameError:
    # If running from Blender console
    project_root = Path("/Users/justinperea/Documents/Art/Pulsaur/3D/Weather_API/weather_art_v3")
    
sys.path.append(str(project_root))

# Try to import our working systems
try:
    from src.weather.weather_data import WeatherObservation
    from src.physics.physics_engine import PhysicsEngine
    from src.particles.particle_system_final import ViscousParticleSystem
except ImportError:
    # Define minimal versions for standalone operation
    print("Note: Running in standalone mode without full system imports")
    
    class WeatherObservation:
        """Minimal weather data for standalone operation"""
        def __init__(self, temperature=20, pressure=1013, humidity=50,
                     wind_speed=5, wind_direction=0, uv_index=5,
                     cloud_cover=50, precipitation=0, visibility=10):
            self.temperature = temperature
            self.pressure = pressure
            self.humidity = humidity
            self.wind_speed = wind_speed
            self.wind_direction = wind_direction
            self.uv_index = uv_index
            self.cloud_cover = cloud_cover
            self.precipitation = precipitation
            self.visibility = visibility


class AnadolMaterialSystem:
    """Weather-responsive material system with Anadol aesthetics"""
    
    def __init__(self):
        self.material = None
        self.emission_node = None
        self.color_ramp = None
        self.math_nodes = {}
        self.setup_time = 0
        
        # Performance tracking
        self.material_update_times = []
        
        # Aesthetic parameters from Anadol research
        self.emission_range = (0.1, 10.0)  # HDR range
        self.color_temp_range = (2000, 10000)  # Kelvin
        
    def create_particle_material(self, weather_data):
        """Create weather-responsive HDR particle material"""
        start_time = time.perf_counter()
        
        # Create new material
        self.material = bpy.data.materials.new(name="AnadolParticleMaterial")
        self.material.use_nodes = True
        
        # Clear default nodes
        nodes = self.material.node_tree.nodes
        nodes.clear()
        
        # Create shader network
        self._create_shader_network(nodes, weather_data)
        
        self.setup_time = time.perf_counter() - start_time
        print(f"Material creation time: {self.setup_time*1000:.2f}ms")
        
        return self.material
    
    def _create_shader_network(self, nodes, weather_data):
        """Build the complete shader network"""
        tree = self.material.node_tree
        
        # === OUTPUT NODE ===
        output_node = nodes.new('ShaderNodeOutputMaterial')
        output_node.location = (800, 0)
        
        # === EMISSION SHADER (particles ARE light) ===
        self.emission_node = nodes.new('ShaderNodeEmission')
        self.emission_node.location = (600, 0)
        
        # === COLOR TEMPERATURE → RGB ===
        # Use Blackbody node for physically accurate color temperature
        blackbody_node = nodes.new('ShaderNodeBlackbody')
        blackbody_node.location = (200, 100)
        
        # === WEATHER-DRIVEN PARAMETERS ===
        
        # Temperature → Color Temperature (2000K-10000K)
        temp_map = nodes.new('ShaderNodeMapRange')
        temp_map.location = (0, 100)
        temp_map.inputs['From Min'].default_value = -20  # °C
        temp_map.inputs['From Max'].default_value = 40   # °C
        temp_map.inputs['To Min'].default_value = 8000   # Cold = bluish
        temp_map.inputs['To Max'].default_value = 2000   # Hot = reddish
        self.math_nodes['temp_to_kelvin'] = temp_map
        
        # Pressure → Base Emission Intensity
        pressure_map = nodes.new('ShaderNodeMapRange')
        pressure_map.location = (0, -100)
        pressure_map.inputs['From Min'].default_value = 980   # hPa
        pressure_map.inputs['From Max'].default_value = 1040  # hPa
        pressure_map.inputs['To Min'].default_value = 0.3    # Low pressure = dimmer
        pressure_map.inputs['To Max'].default_value = 2.0    # High pressure = brighter
        self.math_nodes['pressure_to_emission'] = pressure_map
        
        # Humidity → Glow Spread (via emission multiplier)
        humidity_map = nodes.new('ShaderNodeMapRange')
        humidity_map.location = (0, -300)
        humidity_map.inputs['From Min'].default_value = 0     # %
        humidity_map.inputs['From Max'].default_value = 100   # %
        humidity_map.inputs['To Min'].default_value = 0.8     # Dry = sharper
        humidity_map.inputs['To Max'].default_value = 1.5     # Humid = softer glow
        self.math_nodes['humidity_to_glow'] = humidity_map
        
        # Wind → Emission Variation (flicker effect)
        wind_noise = nodes.new('ShaderNodeTexNoise')  # Changed from ShaderNodeNoise
        wind_noise.location = (0, -500)
        wind_noise.inputs['Scale'].default_value = 10.0
        
        wind_map = nodes.new('ShaderNodeMapRange')
        wind_map.location = (200, -500)
        wind_map.inputs['From Min'].default_value = 0     # m/s
        wind_map.inputs['From Max'].default_value = 50    # m/s
        wind_map.inputs['To Min'].default_value = 0.95    # Calm = stable
        wind_map.inputs['To Max'].default_value = 0.7     # Windy = more flicker
        self.math_nodes['wind_to_variation'] = wind_map
        
        # UV Index → Overall Brightness Multiplier
        uv_map = nodes.new('ShaderNodeMapRange')
        uv_map.location = (0, -700)
        uv_map.inputs['From Min'].default_value = 0      # UV index
        uv_map.inputs['From Max'].default_value = 11     # UV index
        uv_map.inputs['To Min'].default_value = 0.5      # Night/cloudy
        uv_map.inputs['To Max'].default_value = 3.0      # Intense sun
        self.math_nodes['uv_to_brightness'] = uv_map
        
        # === COMBINE WEATHER EFFECTS ===
        
        # Multiply pressure and humidity effects
        multiply1 = nodes.new('ShaderNodeMath')
        multiply1.operation = 'MULTIPLY'
        multiply1.location = (400, -200)
        
        # Apply wind variation
        multiply2 = nodes.new('ShaderNodeMath')
        multiply2.operation = 'MULTIPLY'
        multiply2.location = (400, -400)
        
        # Apply UV brightness
        multiply3 = nodes.new('ShaderNodeMath')
        multiply3.operation = 'MULTIPLY'
        multiply3.location = (400, -600)
        
        # === HDR EMISSION BOOST ===
        # Final multiplier for HDR range (0.1 - 10.0)
        hdr_boost = nodes.new('ShaderNodeMath')
        hdr_boost.operation = 'MULTIPLY'
        hdr_boost.location = (600, -200)
        hdr_boost.inputs[1].default_value = 2.0  # Base HDR multiplier
        self.math_nodes['hdr_boost'] = hdr_boost
        
        # === CONNECT NODES ===
        links = tree.links
        
        # Temperature → Color
        links.new(temp_map.outputs['Result'], blackbody_node.inputs['Temperature'])
        links.new(blackbody_node.outputs['Color'], self.emission_node.inputs['Color'])
        
        # Pressure → Emission base
        links.new(pressure_map.outputs['Result'], multiply1.inputs[0])
        links.new(humidity_map.outputs['Result'], multiply1.inputs[1])
        
        # Wind variation
        links.new(wind_noise.outputs['Fac'], multiply2.inputs[0])
        links.new(wind_map.outputs['Result'], multiply2.inputs[1])
        links.new(multiply1.outputs['Value'], multiply2.inputs[0])
        
        # UV brightness
        links.new(uv_map.outputs['Result'], multiply3.inputs[0])
        links.new(multiply2.outputs['Value'], multiply3.inputs[1])
        
        # HDR boost
        links.new(multiply3.outputs['Value'], hdr_boost.inputs[0])
        
        # Final emission strength
        links.new(hdr_boost.outputs['Value'], self.emission_node.inputs['Strength'])
        
        # Connect to output
        links.new(self.emission_node.outputs['Emission'], output_node.inputs['Surface'])
        
        # Set initial weather values
        self.update_material_parameters(weather_data)
    
    def update_material_parameters(self, weather_data):
        """Update material with new weather data - must be <3.10ms"""
        start_time = time.perf_counter()
        
        if not self.material or not self.math_nodes:
            return
        
        # Update weather-driven values
        self.math_nodes['temp_to_kelvin'].inputs['Value'].default_value = weather_data.temperature
        self.math_nodes['pressure_to_emission'].inputs['Value'].default_value = weather_data.pressure
        self.math_nodes['humidity_to_glow'].inputs['Value'].default_value = weather_data.humidity
        self.math_nodes['wind_to_variation'].inputs['Value'].default_value = weather_data.wind_speed
        self.math_nodes['uv_to_brightness'].inputs['Value'].default_value = weather_data.uv_index
        
        # Update time-based animation for wind effect
        frame = bpy.context.scene.frame_current
        wind_noise = self.material.node_tree.nodes.get('Texture Noise')  # Updated name
        if wind_noise:
            wind_noise.inputs['W'].default_value = frame * 0.1 * (1 + weather_data.wind_speed / 10)
        
        update_time = time.perf_counter() - start_time
        self.material_update_times.append(update_time * 1000)
        
        # Keep only last 60 samples
        if len(self.material_update_times) > 60:
            self.material_update_times.pop(0)
    
    def setup_compositor_effects(self):
        """Setup HDR bloom and post-processing in compositor"""
        scene = bpy.context.scene
        scene.use_nodes = True
        
        # Setup compositor nodes for Blender 4.0+
        tree = scene.node_tree
        nodes = tree.nodes
        nodes.clear()
        
        # Render layers
        render_layers = nodes.new('CompositorNodeRLayers')
        render_layers.location = (0, 0)
        
        # Glare node for bloom/glow
        glare = nodes.new('CompositorNodeGlare')
        glare.location = (200, 0)
        glare.glare_type = 'GHOSTS'
        glare.threshold = 0.8
        glare.mix = 0.3
        
        # Color correction for weather mood
        color_correct = nodes.new('CompositorNodeColorCorrection')
        color_correct.location = (400, 0)
        
        # Composite output
        composite = nodes.new('CompositorNodeComposite')
        composite.location = (600, 0)
        
        # Connect nodes
        links = tree.links
        links.new(render_layers.outputs['Image'], glare.inputs['Image'])
        links.new(glare.outputs['Image'], color_correct.inputs['Image'])
        links.new(color_correct.outputs['Image'], composite.inputs['Image'])
        
        # Setup world for pure black background
        world = bpy.data.worlds['World']
        world.use_nodes = True
        world_nodes = world.node_tree.nodes
        world_nodes.clear()
        
        # Pure black background
        bg_node = world_nodes.new('ShaderNodeBackground')
        bg_node.inputs['Color'].default_value = (0, 0, 0, 1)  # Pure black
        bg_node.inputs['Strength'].default_value = 0  # No emission
        
        world_output = world_nodes.new('ShaderNodeOutputWorld')
        world.node_tree.links.new(bg_node.outputs['Background'], world_output.inputs['Surface'])
        
        print("✅ Compositor setup complete for Blender 4.0+")
    
    def get_performance_stats(self):
        """Return material system performance statistics"""
        if not self.material_update_times:
            return "No performance data yet"
        
        avg_time = np.mean(self.material_update_times)
        max_time = np.max(self.material_update_times)
        
        return f"""
Material Performance:
- Average update: {avg_time:.2f}ms
- Maximum update: {max_time:.2f}ms
- Setup time: {self.setup_time*1000:.2f}ms
- Within budget: {'✅' if max_time < 3.10 else '❌'}
"""


class GeometryParticleRenderer:
    """Enhanced renderer with geometry-based particles for visible emission"""
    
    def __init__(self, particle_count=100_000, particle_scale=1.0):
        self.particle_count = particle_count
        self.particle_scale = particle_scale
        self.base_particle = None
        self.particle_instances = None
        self.instance_collection = None
        self.material_system = AnadolMaterialSystem()
        
        # Performance tracking
        self.render_times = []
        
    def setup_scene(self):
        """Setup Blender scene with Anadol aesthetics and geometry particles"""
        # Clear existing objects
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
        # Setup camera
        camera_data = bpy.data.cameras.new("Camera")
        camera = bpy.data.objects.new("Camera", camera_data)
        bpy.context.collection.objects.link(camera)
        camera.location = (0, -5, 2)
        camera.rotation_euler = (1.396, 0, 0)
        bpy.context.scene.camera = camera
        
        # Create camera target for automatic tracking
        bpy.ops.object.empty_add(location=(0, 0, 1), type='SPHERE')
        self.camera_target = bpy.context.active_object
        self.camera_target.name = "ParticleCenter"
        self.camera_target.empty_display_size = 0.1
        
        # Add tracking constraint to camera
        constraint = camera.constraints.new(type='TRACK_TO')
        constraint.target = self.camera_target
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'
        
        # NO LIGHTS - particles are self-illuminating
        
        # Configure render settings
        scene = bpy.context.scene
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
        scene.render.resolution_x = 3840
        scene.render.resolution_y = 2160
        scene.render.resolution_percentage = 100
        
        # EEVEE settings for best quality
        scene.eevee.taa_render_samples = 64
        scene.eevee.taa_samples = 16
        
        # Create base particle geometry
        self._create_base_particle()
        
        # Setup compositor
        self.material_system.setup_compositor_effects()
        
        # Create elegant container frame
        self._create_container_frame()
        
        print("✅ Scene setup complete with geometry particles and camera tracking")
    
    def _create_base_particle(self):
        """Create base particle sphere for instancing"""
        # Calculate particle size based on count
        if self.particle_count > 50000:
            radius = 0.005 * self.particle_scale
        elif self.particle_count > 10000:
            radius = 0.01 * self.particle_scale
        else:
            radius = 0.02 * self.particle_scale
            
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=1,
            radius=radius,
            location=(0, 0, -10)  # Hide below scene
        )
        self.base_particle = bpy.context.active_object
        self.base_particle.name = "BaseParticle"
        
        print(f"✅ Created base particle with radius {radius}")
    
    def _create_container_frame(self):
        """Create elegant frame for particle container"""
        # Frame material
        frame_mat = bpy.data.materials.new("ContainerFrame")
        frame_mat.use_nodes = True
        nodes = frame_mat.node_tree.nodes
        
        # Subtle emission for frame
        emission = nodes.new('ShaderNodeEmission')
        emission.inputs['Strength'].default_value = 0.1
        emission.inputs['Color'].default_value = (0.2, 0.2, 0.3, 1.0)  # Slight blue tint
        
        output = nodes.get('Material Output')
        frame_mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
        
        # Create frame edges
        frame_thickness = 0.02
        positions = [
            # Bottom edges
            [(-2, -1, 0), (2, -1, 0)],
            [(-2, 1, 0), (2, 1, 0)],
            [(-2, -1, 0), (-2, 1, 0)],
            [(2, -1, 0), (2, 1, 0)],
            # Top edges
            [(-2, -1, 2), (2, -1, 2)],
            [(-2, 1, 2), (2, 1, 2)],
            [(-2, -1, 2), (-2, 1, 2)],
            [(2, -1, 2), (2, 1, 2)],
            # Vertical edges
            [(-2, -1, 0), (-2, -1, 2)],
            [(2, -1, 0), (2, -1, 2)],
            [(-2, 1, 0), (-2, 1, 2)],
            [(2, 1, 0), (2, 1, 2)]
        ]
        
        for i, (start, end) in enumerate(positions):
            # Create curve
            curve = bpy.data.curves.new(f'frame_edge_{i}', 'CURVE')
            curve.dimensions = '3D'
            curve.bevel_depth = frame_thickness
            curve.bevel_resolution = 4
            
            # Add points
            spline = curve.splines.new('NURBS')
            spline.points.add(1)
            spline.points[0].co = (*start, 1)
            spline.points[1].co = (*end, 1)
            
            # Create object
            obj = bpy.data.objects.new(f'frame_edge_{i}', curve)
            obj.data.materials.append(frame_mat)
            bpy.context.collection.objects.link(obj)
    
    def create_particle_instances(self, positions, weather_data):
        """Create particle instances at given positions"""
        print(f"Creating {len(positions):,} particle instances...")
        
        # Create collection for instances
        if "ParticleInstances" not in bpy.data.collections:
            self.instance_collection = bpy.data.collections.new("ParticleInstances")
            bpy.context.scene.collection.children.link(self.instance_collection)
        else:
            self.instance_collection = bpy.data.collections["ParticleInstances"]
            # Clear existing
            for obj in self.instance_collection.objects:
                bpy.data.objects.remove(obj)
        
        # Create material and apply to base particle
        material = self.material_system.create_particle_material(weather_data)
        self.base_particle.data.materials.clear()
        self.base_particle.data.materials.append(material)
        
        # For large particle counts, use instancing
        if len(positions) > 10000:
            # Use a subset for manageable rendering
            step = max(1, len(positions) // 10000)
            positions = positions[::step]
            print(f"  Using subset of {len(positions):,} particles for performance")
        
        # Calculate bounds for camera framing
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        center = (min_pos + max_pos) / 2
        size = np.max(max_pos - min_pos)
        
        # Update camera target position
        if hasattr(self, 'camera_target'):
            self.camera_target.location = center
        
        # Position camera based on particle cloud size
        camera = bpy.data.objects.get("Camera")
        if camera:
            # Calculate distance based on cloud size
            distance = size * 2.5  # Adjust multiplier for framing
            camera.location = (center[0], center[1] - distance, center[2] + distance * 0.5)
            print(f"✅ Camera positioned to frame particle cloud (size: {size:.2f})")
        
        # Create instances
        for i, pos in enumerate(positions):
            instance = self.base_particle.copy()
            instance.data = self.base_particle.data  # Share mesh data
            instance.location = pos
            self.instance_collection.objects.link(instance)
            
            if i % 1000 == 0:
                print(f"  Created {i:,}/{len(positions):,} instances...")
        
        print("✅ Particle instances created and camera framed")
    
    def update_particles(self, positions, weather_data):
        """Update particle positions and materials"""
        start_time = time.perf_counter()
        
        # Update material parameters
        self.material_system.update_material_parameters(weather_data)
        
        # For now, we'll update a subset of particles for performance
        # In production, use geometry nodes for better performance
        if self.instance_collection and len(self.instance_collection.objects) > 0:
            instances = list(self.instance_collection.objects)
            num_to_update = min(len(positions), len(instances))
            
            for i in range(num_to_update):
                instances[i].location = positions[i]
        
        update_time = time.perf_counter() - start_time
        self.render_times.append(update_time * 1000)
        
        if len(self.render_times) > 60:
            self.render_times.pop(0)
    
    def get_performance_report(self):
        """Get combined performance statistics"""
        if not self.render_times:
            return "No performance data yet"
        
        render_avg = np.mean(self.render_times)
        render_max = np.max(self.render_times)
        
        material_stats = self.material_system.get_performance_stats()
        
        return f"""
Render Update Performance:
- Average: {render_avg:.2f}ms
- Maximum: {render_max:.2f}ms

{material_stats}
"""


def run_geometry_particle_demo():
    """Demonstration of geometry-based weather particles"""
    print("\n=== GEOMETRY-BASED WEATHER PARTICLE DEMO ===\n")
    
    # Test weather conditions
    weather = WeatherObservation(
        temperature=22.0,
        pressure=1013.25,
        humidity=65.0,
        wind_speed=5.0,
        uv_index=7,
        cloud_cover=50
    )
    
    # Create particle positions (simplified for demo)
    particle_count = 5000  # Manageable for geometry instances
    positions = np.random.rand(particle_count, 3).astype(np.float32)
    positions[:, 0] = positions[:, 0] * 3.6 - 1.8  # X: -1.8 to 1.8
    positions[:, 1] = positions[:, 1] * 3.6 - 1.8  # Y: -1.8 to 1.8
    positions[:, 2] = positions[:, 2] * 1.7 + 0.1  # Z: 0.1 to 1.8
    
    # Create renderer
    renderer = GeometryParticleRenderer(particle_count=particle_count)
    renderer.setup_scene()
    renderer.create_particle_instances(positions, weather)
    
    # Animate with weather changes
    print("\nAnimating weather changes...")
    for frame in range(120):
        bpy.context.scene.frame_set(frame + 1)
        
        # Vary weather
        t = frame / 120.0
        weather.temperature = 22 + 10 * np.sin(t * np.pi * 2)
        weather.pressure = 1013 + 20 * np.cos(t * np.pi * 2)
        weather.uv_index = int(5 + 4 * np.sin(t * np.pi))
        
        # Update particles (simplified movement)
        positions += np.random.randn(*positions.shape) * 0.001
        
        # Clamp to boundaries
        positions[:, 0] = np.clip(positions[:, 0], -1.8, 1.8)
        positions[:, 1] = np.clip(positions[:, 1], -1.8, 1.8)
        positions[:, 2] = np.clip(positions[:, 2], 0.1, 1.8)
        
        # Update renderer
        renderer.update_particles(positions, weather)
        
        if frame % 30 == 0:
            print(f"Frame {frame}: Temp {weather.temperature:.1f}°C, Pressure {weather.pressure:.0f} hPa")
    
    # Final report
    print("\n" + renderer.get_performance_report())
    
    # Save verification render
    bpy.context.scene.render.filepath = "/tmp/geometry_particles_final.png"
    bpy.ops.render.render(write_still=True)
    print(f"\n✅ Final render saved to: {bpy.context.scene.render.filepath}")


if __name__ == "__main__":
    # Run in Blender
    if "bpy" in locals():
        run_geometry_particle_demo()
    else:
        print("This script must be run within Blender!")
        print("Use: blender --python anadol_materials.py")