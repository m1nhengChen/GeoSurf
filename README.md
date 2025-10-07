# GeoSurf 🧭
*A lightweight Python toolkit for geodesic and Euclidean distance transforms on surface meshes.*

---

## Overview

**GeoSurf** provides an easy command-line interface (CLI) and Python functions to compute
distance transforms on 3D surface meshes (e.g., cortical surfaces, anatomical or geometric models).

It supports three major algorithms:
- **Dijkstra** – Graph-based shortest path (approximate geodesic)
- **Heat** – Fast and accurate heat-method geodesic (via `potpourri3d`)
- **Euclid** – Straight-line 3D distance (baseline reference)

---

## Features

- 🔹 Read and write both `.vtk` (legacy) and `.vtp` (XML) PolyData formats  
- 🔹 Compute per-vertex distance to one or multiple seed points  
- 🔹 Output results as scalar arrays ready for visualization in ParaView or FreeView  
- 🔹 CLI and library-level API (both supported)  
- 🔹 Cross-platform, lightweight dependencies  

---

## Installation

```bash
git clone https://github.com/<your-username>/GeoSurf.git
cd GeoSurf

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

