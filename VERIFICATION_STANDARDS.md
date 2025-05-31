# Verification Standards

## Overview

Every development step must include comprehensive verification to ensure gallery-quality results.

## Visual Verification Requirements

### Required Outputs
1. **Primary Visualization**: Main functionality demonstration
2. **Comparison/Analysis**: Before/after or parameter variations
3. **Performance Plots**: FPS, memory usage, scaling behavior

### File Naming Convention
```
verification_outputs/chat_X_component/step_Y_description/
├── primary_visualization.png
├── comparison_analysis.png
├── performance_metrics.png
└── verification_data.json
```

## Numerical Verification Requirements

### Performance Metrics
- Execution time (milliseconds)
- Memory usage (GB)
- Frame rate (FPS)
- Particle count

### Accuracy Metrics
- Numerical precision
- Physics accuracy
- Visual fidelity

## Documentation Requirements

### Code Documentation
- All functions must have docstrings
- Complex algorithms need inline comments
- Type hints required for all parameters

### Verification Reports
- Summary of what was tested
- Results and analysis
- Any issues discovered
- Next steps

## Error Handling

### Error Documentation
- Screenshot of error
- Full stack trace
- Steps to reproduce
- Solution applied
- Prevention measures

### Error Log Format
```markdown
## Error: [Brief Description]
Date: YYYY-MM-DD
Chat: X
Severity: [Critical/High/Medium/Low]

### Description
What went wrong

### Solution
How it was fixed

### Prevention
How to avoid in future
```
