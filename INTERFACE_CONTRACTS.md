# Interface Contracts

## Chat 2 → Chat 3: Physics to Particles

```python
def generate_3d_force_field(weather_data, resolution=(64, 32, 16)):
    '''
    Convert weather physics calculations to 3D spatial force field arrays

    Args:
        weather_data: WeatherObservation object with all parameters
        resolution: Tuple (x, y, z) grid resolution for force field

    Returns:
        np.array: Shape (64, 32, 16, 3) with force vectors at each grid point
    '''

def sample_force_at_position(force_field, position):
    '''
    Sample force field at arbitrary 3D world position

    Args:
        force_field: np.array from generate_3d_force_field()
        position: np.array [x, y, z] in world coordinates

    Returns:
        np.array: [fx, fy, fz] force vector at position
    '''
```

## Chat 3 → Chat 4: Particles to Blender

```python
class ViscousParticleSystem:
    def get_render_data(self):
        '''
        Return data needed for Blender rendering

        Returns:
            tuple: (positions, velocities, colors)
            - positions: np.array shape (N, 3) world coordinates  
            - velocities: np.array shape (N, 3) for motion blur
            - colors: np.array shape (N, 3) RGB values [0,1]
        '''
```

## Chat 4 → Chat 5: Blender to Materials

```python
class BlenderParticleRenderer:
    def get_material_nodes(self):
        '''
        Return material node setup for particle shaders

        Returns:
            dict: Blender material node references for shader setup
        '''
```

## Chat 5 → Chat 6: Materials to Gallery

```python
class AnadolMaterialSystem:
    def create_particle_material(self, weather_data):
        '''
        Generate weather-responsive material properties

        Args:
            weather_data: Current weather observation

        Returns:
            dict: Material properties for particle emission
        '''
```
