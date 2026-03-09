# SE(2) Odometry Uncertainty Propagation

Interactive visualization of covariance propagation for differential-drive odometry, based on Siegwart & Nourbakhsh (2011), §5.2.4.

**Live Demo:**: https://dgist-slam.github.io/se2-diff-odom-uncertainty/

## Features

- Ground truth vs noisy odometry estimate (dual trajectory)
- Analytical covariance propagation with xy-ellipse and θ-wedge visualization
- Adjustable robot parameters (r, l), noise parameters (kR, kL), and display settings
- Preset trajectories: Straight Line, Circle, Square, Figure-8, Zigzag
- Interactive keyboard-driven robot control
- Step-by-step animation with scrubbing
- Hover tooltips with det(Σ), σ_θ, position error

## Development

```bash
npm install
npm run dev
```

## Deploy

Pushes to `main` automatically deploy to GitHub Pages via GitHub Actions.

---

DGIST · Dept. of Robotics & Mechatronics · RT604 SLAM Course
