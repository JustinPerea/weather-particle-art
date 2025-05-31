# Contributing to Weather Particle Art

## Multi-Chat Development Workflow

This project uses a specialized multi-chat development approach where 6 different Claude instances work on specific components. Each chat has deep expertise in their domain while maintaining clear interfaces for integration.

### Chat Responsibilities

#### Chat 1: GitHub & Project Management
- Repository structure and maintenance
- Issue tracking and project board management
- Cross-chat coordination
- Documentation standards

#### Chat 2: Weather Data & Physics Equations  
- NOAA API integration
- Physics equation implementation
- Force field generation
- Mathematical verification

#### Chat 3: Particle System & Viscous Dynamics
- Viscous particle behavior
- Force field integration
- Boundary systems
- Performance optimization

#### Chat 4: Blender Integration & Rendering
- Blender Python API
- GPU rendering optimization
- Real-time updates
- Memory management

#### Chat 5: Materials, Lighting & Aesthetics
- Anadol-inspired visuals
- Self-illuminating materials
- Weather-responsive aesthetics
- Post-processing effects

#### Chat 6: Gallery Installation & Production
- System integration
- Installation tools
- Reliability systems
- Gallery documentation

## Development Process

### 1. Issue Creation
Every development task starts with a GitHub issue using the appropriate template.

### 2. Branch Creation
Create a feature branch for your chat's work:
```bash
git checkout -b chat-X/feature-name
```

### 3. Development
Follow your chat's specific guidelines in the development plan.

### 4. Verification
Run verification before committing:
```bash
python scripts/run_verification.py --chat X
```

### 5. Pull Request
Create a PR with verification evidence and performance data.

### 6. Handoff
Document interfaces for the next chat when ready.

## Code Standards

- Follow PEP 8
- Use type hints
- Document all public functions
- Include doctest examples

## Quality Checklist

Before submitting work, verify:
- [ ] All tests pass
- [ ] Performance targets met
- [ ] Visual verification complete
- [ ] Documentation updated
- [ ] No hardcoded values
- [ ] Error handling implemented
- [ ] Interface contracts fulfilled
- [ ] Code follows standards
