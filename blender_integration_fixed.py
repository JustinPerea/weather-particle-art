#!/usr/bin/env python3
"""
Fixed Blender Integration - Using correct module names
Connects Chat 3 Particles to Chat 4 Renderer with actual file names
"""

import sys
import os
import time
import numpy as np
import logging
from typing import Optional

# Add project root to path
project_root = "/Users/justinperea/Documents/Art/Pulsaur/3D/Weather_API/weather_art_v3"
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import with CORRECT names based on your file structure
try:
    from weather.noaa_api import WeatherAPI  # Changed from weather_api
    from physics.force_field_engine import PhysicsEngine  # Changed from physics_engine
    from particles.viscous_particle_system import ViscousParticleSystem
    logger.info("✅ All modules imported successfully!")
except ImportError as e:
    logger.error(f"Import error: {e}")
    # Try alternative import names
    try:
        # Maybe the classes have different names?
        from weather.noaa_api import NOAAAPI as WeatherAPI
    except:
        pass

# Check if we're running in Blender
IN_BLENDER = "bpy" in sys.modules

if IN_BLENDER:
    import bpy
    from blender.particle_renderer import BlenderParticleRenderer
else:
    logger.warning("Not running in Blender - rendering functions disabled")


class WeatherParticleBlenderIntegration:
    """
    Complete integration of weather → physics → particles → Blender rendering
    Fixed to use correct module names from your file structure
    """
    
    def __init__(self, particle_count: int = 100_000, demo_mode: bool = True):
        """Initialize complete system integration"""
        self.particle_count = particle_count
        self.demo_mode = demo_mode
        self.frame_count = 0
        
        logger.info(f"Initializing integration with {particle_count:,} particles")
        
        # Initialize components
        self._init_weather_system()
        self._init_physics_engine()
        self._init_particle_system()
        
        if IN_BLENDER:
            self._init_blender_renderer()
        
        # Performance tracking
        self.frame_times = []
        self.component_times = {
            'weather': [],
            'physics': [],
            'particles': [],
            'rendering': []
        }
        
        logger.info("Integration initialized successfully")
    
    def _init_weather_system(self):
        """Initialize weather data source"""
        if self.demo_mode:
            # Create demo weather data
            logger.info("Using demo weather data")
            self.current_weather = self._create_demo_weather()
        else:
            try:
                # Try to use the NOAA API
                self.weather_api = WeatherAPI()
                self.current_weather = self.weather_api.get_weather(
                    station_id="KIAD"
                )
            except:
                logger.warning("Weather API failed, using demo data")
                self.current_weather = self._create_demo_weather()
    
    def _create_demo_weather(self):
        """Create demo weather observation"""
        # Create a simple weather observation object
        class WeatherObservation:
            def __init__(self):
                self.temperature = 22.0  # Celsius
                self.pressure = 1013.25  # hPa
                self.humidity = 65.0  # %
                self.wind_speed = 5.0  # m/s
                self.wind_direction = 180  # degrees
                self.uv_index = 5
                self.cloud_cover = 50  # %
                self.precipitation = 0  # mm
        
        return WeatherObservation()
    
    def _init_physics_engine(self):
        """Initialize physics force field generator"""
        try:
            self.physics_engine = PhysicsEngine()
            
            # Generate initial force field
            self.force_field = self.physics_engine.generate_3d_force_field(
                self.current_weather,
                resolution=(64, 32, 16)
            )
            logger.info("Physics engine initialized with force field")
        except Exception as e:
            logger.warning(f"Physics engine failed: {e}, using simple force field")
            # Create simple force field for testing
            self.force_field = np.random.randn(64, 32, 16, 3).astype(np.float32) * 0.1
    
    def _init_particle_system(self):
        """Initialize viscous particle system"""
        self.particle_system = ViscousParticleSystem(
            particle_count=self.particle_count
        )
        logger.info(f"Particle system initialized with {self.particle_count:,} particles")
    
    def _init_blender_renderer(self):
        """Initialize Blender rendering system"""
        # Fix the renderer to handle Blender 4.0+
        self._fix_particle_renderer()
        
        self.renderer = BlenderParticleRenderer(
            particle_count=self.particle_count
        )
        
        # Get material nodes for Chat 5
        self.material_nodes = self.renderer.get_material_nodes()
        
        logger.info("Blender renderer initialized")
    
    def _fix_particle_renderer(self):
        """Fix the particle renderer for Blender 4.0+"""
        # Temporarily patch the renderer to use correct engine name
        import blender.particle_renderer as pr
        
        # Store original _setup_scene method
        original_setup = pr.BlenderParticleRenderer._setup_scene
        
        def fixed_setup_scene(self):
            """Fixed setup for Blender 4.0+"""
            scene = bpy.context.scene
            
            # Use correct render engine name
            available_engines = [e.bl_idname for e in bpy.types.RenderEngine.__subclasses__()]
            
            if 'BLENDER_EEVEE_NEXT' in available_engines:
                scene.render.engine = 'BLENDER_EEVEE_NEXT'
            elif 'BLENDER_EEVEE' in available_engines:
                scene.render.engine = 'BLENDER_EEVEE'
            else:
                scene.render.engine = 'CYCLES'
            
            # Continue with rest of setup
            scene.render.resolution_x = 3840
            scene.render.resolution_y = 2160
            scene.render.resolution_percentage = 100
            
            # Optimize settings
            if hasattr(scene, 'eevee'):
                eevee = scene.eevee
                eevee.taa_render_samples = 1
                eevee.taa_samples = 1
                eevee.use_ssr = False
                eevee.use_bloom = False
                eevee.use_volumetric_shadows = False
                eevee.use_motion_blur = False
            
            scene.render.fps = 60
            scene.render.fps_base = 1.0
            
            scene.view_settings.view_transform = 'Standard'
            scene.view_settings.look = 'None'
            scene.view_settings.exposure = 0
            scene.view_settings.gamma = 1.0
            
            # Setup camera
            self._setup_camera()
            
            # Black background
            if not scene.world:
                scene.world = bpy.data.worlds.new(name="World")
            
            scene.world.use_nodes = True
            if "Background" in scene.world.node_tree.nodes:
                bg_node = scene.world.node_tree.nodes["Background"]
                bg_node.inputs[0].default_value = (0, 0, 0, 1)
                bg_node.inputs[1].default_value = 0
        
        # Replace the method
        pr.BlenderParticleRenderer._setup_scene = fixed_setup_scene
    
    def update_frame(self, frame: Optional[int] = None):
        """Update one frame of the complete system"""
        frame_start = time.perf_counter()
        
        # Component timing
        times = {}
        
        # 1. Weather update (every 5 minutes = 18000 frames at 60 FPS)
        weather_start = time.perf_counter()
        if self.frame_count % 18000 == 0 and self.frame_count > 0:
            self._update_weather()
        times['weather'] = time.perf_counter() - weather_start
        
        # 2. Physics update (already generated)
        physics_start = time.perf_counter()
        times['physics'] = time.perf_counter() - physics_start
        
        # 3. Particle update
        particle_start = time.perf_counter()
        self.particle_system.update(self.force_field, dt=0.016)
        times['particles'] = time.perf_counter() - particle_start
        
        # 4. Render update (only in Blender)
        render_start = time.perf_counter()
        if IN_BLENDER:
            # Get particle data
            positions, velocities, colors = self.particle_system.get_render_data()
            
            # Update Blender
            self.renderer.update_particles(positions, velocities, colors)
            
            # Update frame in Blender
            if frame is not None:
                bpy.context.scene.frame_set(frame)
        times['rendering'] = time.perf_counter() - render_start
        
        # Total frame time
        frame_time = time.perf_counter() - frame_start
        self.frame_times.append(frame_time)
        
        # Store component times
        for component, duration in times.items():
            self.component_times[component].append(duration)
        
        # Performance monitoring every second
        if self.frame_count % 60 == 0 and self.frame_count > 0:
            self._log_performance()
        
        self.frame_count += 1
    
    def _update_weather(self):
        """Update weather data and regenerate force field"""
        if self.demo_mode:
            # Cycle through different weather conditions
            patterns = ['storm', 'calm', 'heat_wave', 'fog']
            pattern_idx = (self.frame_count // 18000) % len(patterns)
            
            # Modify demo weather based on pattern
            if patterns[pattern_idx] == 'storm':
                self.current_weather.pressure = 990
                self.current_weather.wind_speed = 20
            elif patterns[pattern_idx] == 'calm':
                self.current_weather.pressure = 1020
                self.current_weather.wind_speed = 2
            elif patterns[pattern_idx] == 'heat_wave':
                self.current_weather.temperature = 38
                self.current_weather.humidity = 20
            elif patterns[pattern_idx] == 'fog':
                self.current_weather.humidity = 95
                self.current_weather.cloud_cover = 100
            
            logger.info(f"Weather pattern: {patterns[pattern_idx]}")
        
        # Regenerate force field
        try:
            self.force_field = self.physics_engine.generate_3d_force_field(
                self.current_weather,
                resolution=(64, 32, 16)
            )
        except:
            # Keep using existing force field if generation fails
            pass
    
    def _log_performance(self):
        """Log performance metrics"""
        recent_frames = self.frame_times[-60:]
        
        avg_frame = np.mean(recent_frames) * 1000
        max_frame = np.max(recent_frames) * 1000
        fps = 1.0 / np.mean(recent_frames)
        
        # Component breakdown
        component_avgs = {}
        for component, times in self.component_times.items():
            if times:
                recent = times[-60:] if len(times) >= 60 else times
                component_avgs[component] = np.mean(recent) * 1000
        
        logger.info(
            f"Performance: {fps:.1f} FPS | "
            f"Frame: {avg_frame:.2f}ms avg, {max_frame:.2f}ms max | "
            f"Particles: {component_avgs.get('particles', 0):.2f}ms, "
            f"Render: {component_avgs.get('rendering', 0):.2f}ms"
        )
    
    def run_animation(self, duration_seconds: int = 10):
        """Run the complete animation for a specified duration"""
        if not IN_BLENDER:
            logger.error("Animation requires running within Blender")
            return
        
        logger.info(f"Starting {duration_seconds} second animation...")
        
        start_time = time.time()
        frame = 0
        
        while time.time() - start_time < duration_seconds:
            self.update_frame(frame)
            frame += 1
            
            # Update Blender viewport
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        
        # Final performance summary
        self._print_performance_summary()
    
    def _print_performance_summary(self):
        """Print detailed performance summary"""
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY - Chat 4 Integration")
        print("="*60)
        
        avg_frame = np.mean(self.frame_times) * 1000
        fps = 1.0 / np.mean(self.frame_times)
        
        print(f"\nOverall Performance:")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  Average frame time: {avg_frame:.2f}ms")
        print(f"  Target achieved: {'YES' if fps >= 60 else 'NO'}")
        
        print(f"\nComponent Timing (ms):")
        for component, times in self.component_times.items():
            if times:
                avg_time = np.mean(times) * 1000
                max_time = np.max(times) * 1000
                print(f"  {component.capitalize()}: {avg_time:.2f}ms avg, {max_time:.2f}ms max")
        
        total_avg = sum(np.mean(times) * 1000 for times in self.component_times.values() if times)
        print(f"\nTotal frame budget used: {total_avg:.2f}ms / 16.67ms")
        print("="*60)


def run_blender_integration_test():
    """Run integration test within Blender"""
    
    print("\n" + "="*60)
    print("CHAT 4 BLENDER INTEGRATION TEST")
    print("="*60)
    
    # Create integration
    integration = WeatherParticleBlenderIntegration(
        particle_count=100_000,
        demo_mode=True
    )
    
    # Run for 5 seconds
    integration.run_animation(duration_seconds=5)
    
    print("\n✅ Integration test complete!")
    
    return integration


if __name__ == "__main__":
    if IN_BLENDER:
        # Store in Blender for console access
        bpy.integration = run_blender_integration_test()
        print("\n💡 Access integration with: bpy.integration")
    else:
        print("This script must be run within Blender!")