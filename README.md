# Weather-Driven Viscous Particle Art System

<div align="center">

  **A real-time 3D art installation transforming atmospheric data into mesmerizing viscous particle flows**

  [![Verification Status](https://github.com/JustinPerea/weather-particle-art/workflows/Automated%20Verification/badge.svg)](https://github.com/JustinPerea/weather-particle-art/actions)
  [![Documentation](https://img.shields.io/badge/docs-online-blue)](https://justinperea.github.io/weather-particle-art/)
  [![Gallery Ready](https://img.shields.io/badge/gallery-ready-green)](docs/gallery/installation_guide.md)
</div>

## Overview

This project creates a living, breathing visualization of our planet's atmosphere through the lens of theoretical physics and contemporary digital art. By translating real-time weather data through general relativity equations and quantum field theory, we generate invisible force fields that guide millions of viscous particles in honey-like flows across a 4K display.

Inspired by the pioneering work of Refik Anadol, this installation treats data as "pigment that never dries," creating an ever-evolving artwork that responds to the atmospheric conditions of its exhibition location.

## Key Features

- **Scientific Foundation**: Authentic physics equations create force fields from weather data
- **Million-Particle Performance**: GPU-optimized system renders 1M+ particles at 60 FPS
- **Gallery-Grade Reliability**: 24/7 operation with automatic recovery systems  
- **Weather Responsive**: Real-time NOAA data drives continuous evolution
- **Anadol Aesthetics**: Self-illuminating particles with HDR emission effects
- **Professional Installation**: One-click setup for gallery technicians

## System Architecture

```
Weather Data (NOAA API)
    ↓
Physics Equations (GR + Quantum Fields)
    ↓
3D Force Field Generation
    ↓
Viscous Particle System (1M particles)
    ↓
Blender GPU Rendering
    ↓
4K Gallery Display
```

## Gallery Installation

For gallery installation instructions, see our [Gallery Installation Guide](docs/gallery/installation_guide.md).

### Quick Start

```bash
# Clone repository
git clone https://github.com/JustinPerea/weather-particle-art.git
cd weather-particle-art

# Run automated installer
python scripts/create_installer.py
./install_gallery.sh

# Launch system
python src/gallery/installation_system.py
```

## Development

This project uses a unique multi-chat development approach with 6 specialized teams:

1. **Chat 2**: Weather Data & Physics Equations
2. **Chat 3**: Particle System & Viscous Dynamics  
3. **Chat 4**: Blender Integration & Rendering
4. **Chat 5**: Materials, Lighting & Aesthetics
5. **Chat 6**: Gallery Installation & Production

See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow details.

## Technical Requirements

### Hardware
- **GPU**: NVIDIA RTX 3080 or better (16GB+ VRAM recommended)
- **CPU**: 8+ cores recommended
- **RAM**: 32GB minimum, 64GB recommended
- **Display**: 4K (3840x2160) or higher

### Software
- Python 3.9+
- Blender 3.6+
- CUDA 11.8+
- See [requirements.txt](requirements.txt) for Python dependencies

## Documentation

- [Technical Documentation](https://justinperea.github.io/weather-particle-art/)
- [API Reference](docs/api/)
- [Gallery Setup Guide](docs/gallery/installation_guide.md)
- [Troubleshooting](docs/gallery/troubleshooting.md)

## Media Kit

Gallery curators and press can find high-resolution images, technical specifications, and artist statements in our [Media Kit](media_kit/).

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by the groundbreaking work of [Refik Anadol](http://refikanadol.com/)
- Weather data provided by [NOAA](https://www.noaa.gov/)
- Built with [Blender](https://www.blender.org/) and Python

## Contact

For gallery inquiries, technical support, or press requests:
- Gallery: gallery@weatherparticleart.com
- Technical: support@weatherparticleart.com
- Press: press@weatherparticleart.com
