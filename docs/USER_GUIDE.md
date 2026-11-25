# User Guide — Symbiote: Abiogenesis (v0.3.0)

## Launching the Application

- **Windows Prebuilt Executable**: Download `symbiote-abiogenesis-v0.3.0-windows-x86_64.exe` from [GitHub Releases](https://github.com/ShamelesAbyss/Symbiote-Abiogenesis/releases/tag/v0.3.0) and double-click it to launch the native **Abiogenesis Research Laboratory**.
- **From Source (Rust 1.80+)**:
  ```bash
  git clone https://github.com/ShamelesAbyss/Symbiote-Abiogenesis.git
  cd Symbiote-Abiogenesis
  cargo run --release
  ```

## Command-Line Modes

| Command | Behavior |
| :--- | :--- |
| `symbiote-abiogenesis` *(default)* | Launches the native `winit` + `wgpu` + `egui` Research Laboratory |
| `symbiote-abiogenesis --cpu` | Forces CPU-Rayon execution for `EcoField` instead of WGSL GPU compute |
| `symbiote-abiogenesis --experiment` | Runs the 4-part quantitative validation suite (`--cpu` supported) |
| `symbiote-abiogenesis --snapshot` | Renders `1440×810` PNG captures to `assets/screenshots/` |
| `symbiote-abiogenesis --legacy-tui` | Launches the original Symbiote terminal simulation |

## Research Laboratory Controls

| Control | Action |
| :--- | :--- |
| **Left Click** (on organism) | Select organism in the Microscopic Inspector |
| **Mouse Wheel** / `+` / `-` | Seamless zoom (`0.7×` overview → `18.0×` microscopic) |
| **Arrow Keys** / `W A S D` | Pan camera across the toroidal abyssal ocean |
| `Space` | Pause / resume simulation |
| `H` | Toggle distraction-free **Cinematic Observation Mode** |
| `0` | Reset camera to default view |
| `Q` / `Esc` | Exit application |

## World Persistence

- Clicking **Save** writes an atomic versioned snapshot (`v2`) to `saves/worlds/autosave.abio.json` via temporary file + atomic rename.
- Clicking **Load** restores organisms, `40×40×3` cellular fields, `MorphGenome`s, `NeuralBrain` weights, `EcoField` chemistry, `PhylogenyArchive`, and universe settings.
