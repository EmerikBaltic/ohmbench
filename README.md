# Ohmbench — Desktop Electrical Circuit Simulator

[![CI](https://github.com/EmerikBaltic/ohmbench/actions/workflows/ci.yml/badge.svg)](https://github.com/EmerikBaltic/ohmbench/actions/workflows/ci.yml)

**Ohmbench** is an open-source, educational desktop electrical circuit simulator built with **Python 3.11+**, **NumPy / SciPy** (Modified Nodal Analysis physics engine), and **PySide6 (Qt 6 Widgets)**.

It is designed for electrical engineering students and hobbyists who want both an interactive schematic capture + virtual instrument bench and a clean, readable Python codebase that demonstrates how real circuit simulators (like SPICE) work under the hood.

---

## Current Project Status: Release Candidate (Milestones 1–5 Complete)

- **Milestone 1 (Simulation Engine — Complete)**:
  - Pure-Python/NumPy/SciPy **Modified Nodal Analysis (MNA)** solver with cached LU factorization (`scipy.linalg.lu_factor` / `lu_solve`), strictly decoupled from Qt.
  - **DC Operating Point**, **AC Small-Signal Sweep**, and **Transient Analysis** using **Trapezoidal integration** with automatic **Backward Euler** startup and switch-event damping.
  - **Components**: Resistor, Potentiometer, Capacitor (ESR & $V_0$), Inductor (DCR, $X_L = 2\pi f L$, $\tau = L / R$, reverse calculator), DC/AC/WAV Voltage & Current Sources, SPST Switch ($350\text{ V}$ arc model), SPDT Relay, Fuse, Transformer ($N_p/N_s$, Ideal vs. Non-Ideal coupled coils), Incandescent Lightbulb ($R_{\text{hot}} = V^2/P$), Diode/Zener/LED, BJT (NPN/PNP), MOSFET (NMOS/PMOS), Op-Amp, Comparator, NE555, Digital Logic Gates & Sequential Blocks (D/JK Flip-Flops, 4-bit Counter, 7-Seg Display), Python-scriptable MCU Block, Voltmeter, Ammeter, and **4-pin Benchtop Oscilloscope (`OSC1` / DSO-1062D)**.
- **Milestones 2–5 (Workbench, Benchtop DSO-1062D & I/O — Complete)**:
  - Dockable **PySide6 (Qt 6)** Main Window with Dark & Light themes (`Ctrl+T`), live searchable Palette & Problems Dock, immutable `WireGraph` orthogonal routing & auto-connect, `QUndoStack` Undo/Redo, and animated Switches/Fuses/LEDs.
  - Placeable **Voltcraft DSO-1062D** benchtop oscilloscope (`CH1`, `CH2`, `EXT_TRIG`, `GND` BNC inputs, miniature live LCD preview, double-click front-panel UI with 11 menus, `F0`–`F7` softkeys, rotary `QDial`s, `AUTO SET`, `SET TO 50%`, `SINGLE SEQ`, FFT, Cursors, and PNG/CSV export).
  - Save/Load `.circ` v2, shareable `ohmbench://c/...` URLs, BOM/Power report, KiCad `.kicad_sch` export, **Standard SPICE Netlist export (`.cir` with `MEG` suffixes)**, and 3rd-party SPICE `.MODEL`/`.SUBCKT` import.

---

## How to Use

1. **Place Parts**: Click or drag a part from the left **Component Palette** onto the schematic canvas (or double-click a palette item to place it at the center). Always include a **Ground** part wired to your circuit.
2. **Rotate & Flip**: Select one or more parts and press **`R`** to rotate $90^\circ$ clockwise or **`F`** to flip horizontally. Labels and values stay upright and readable at any angle.
3. **Draw & Edit Wires**: Drag from any component terminal dot or press **`W`** for Wire Mode (**`Esc`** returns to Select Mode). Click empty grid intersections to place orthogonal waypoints (**`Backspace`** removes the last waypoint; double-click or click a terminal/junction finishes). Drag any wire segment orthogonally to slide it, or drag a wire endpoint handle to detach (`FreeEnd` hollow marker) or reattach to another terminal/junction (10 px snap radius). Crossing wires do *not* connect unless connected through an explicit **Junction Dot**.
4. **Edit Component Values**: **Double-click** any component on the canvas to open its value editor. Type values using engineering prefixes (`4.7k`, `5,65 G`, `100n`, `10K`)—a live preview (`= 4.7 kΩ`) appears below the input, and changes apply immediately even while the simulation is running (and can be undone with **`Ctrl+Z`**).
5. **Run / Pause / Stop**: Use the top toolbar or keyboard shortcuts—**Run (`F5`)**, **Pause / Resume (`F6` or `Space`)**, and **Stop (`F7`)**—and choose simulation speed (`1x Real-Time`, `0.1x Slow`, `10x Fast`).
6. **Toggle Switches Live**: Click directly on any **Switch** on the canvas while running to animate its lever open/closed and watch meter readings and lightbulb brightness respond in real time.

---

## Saving and Opening Circuits

Circuits are stored as `.circ` files (plain JSON text that you can open in any editor).

- **Save**: `Ctrl+S` (or the toolbar **Save** button). The first time you are asked for a name; `Ctrl+Shift+S` (**Save As**) always asks.
- **Open**: `Ctrl+O`. **New**: `Ctrl+N`.
- **Open Recent**: `File > Open Recent` preserves the last 8 opened circuits.
- **Example Circuits**: `File > Examples` loads bundled reference schematics (Voltage Divider, RC Low-Pass Filter, Lightbulb & Switch, Series Sources, Wheatstone Bridge).
- **Export Schematic**: `File > Export as PNG...` and `File > Export as SVG...` export clean high-resolution schematics with White or Theme canvas backgrounds.
- **Zoom to Fit**: `View > Zoom to fit` (`Ctrl+0`) frames the whole circuit cleanly.
- **Preferences**: `Edit > Preferences` configures regional decimal number formatting (Auto, Dot, Comma) and default newly-placed meter behavior (Ideal, Realistic).
- **Help & Diagnostics**: `Help > About` displays software and platform versions, while unhandled crashes are logged safely to the user app-data folder.
- The window title shows the file name and an asterisk (`*`) when there are unsaved changes. Ohmbench asks before New, Open or closing if you would lose changes.
- Saving is safe: the file is written to a temporary file first and only then swapped in, and the previous version is kept as `yourfile.circ.bak`.
- A damaged, empty or unknown file (or one made by a newer Ohmbench) shows a friendly message and leaves your current circuit untouched.
- Saved with the file: parts, values, labels, rotation/flip, wires, junction dots, theme, zoom, view position and simulation speed.

---

## Screenshots

> *Placeholder — drop screenshots of the schematic canvas, live simulation, and theme views into `docs/screenshots/` and link them here:*
>
> - `[Screenshot: Dark Theme Schematic & Live Meters]`
> - `[Screenshot: Lightbulb Glow & Double-Click Value Editor]`
> - `[Screenshot: Light Theme & Canvas Problem Highlighting]`

---

## Quick Start & Installation

### 1. Clone the Repository and Create a Virtual Environment

```powershell
git clone https://github.com/<your-username>/circuitsim.git
cd circuitsim
python -m venv .venv
```

### 2. Install Dependencies and `circuitsim` in Editable Mode

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\pip.exe install -e .
```

### 3. Run the Unit Test Suite

Run the full automated test suite (uses a shared offscreen `QApplication` for Qt tests and clean subprocesses for Qt-free module import checks so `sys.modules` is verified without Qt already loaded):

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run the fast subset (excludes `@pytest.mark.slow` tests taking over 3 seconds, such as the 1,000-exception crash-handler deduplication test):

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not slow"
```

### 4. Run the Application / Engine Demo

```powershell
.\.venv\Scripts\python.exe -m circuitsim.main
```

---

## Development Roadmap

| Milestone | Scope | Status |
|-----------|-------|--------|
| **Milestone 1** | Core MNA DC & Transient Engine, R/L/C, Sources, Switch, Bulb, Meters, Parser, Analytical Tests | **Done** |
| **Milestone 2** | Dockable Main Window, Component Palette, Interactive IEC Schematic Canvas, Worker Thread, Live Meters/Switch/Bulb | **Done** |
| **Milestone 3** | Double-Click Live Properties Dialog, Inductor $X_L / \tau$ Calculator, `.circ` JSON Save/Load (Saved Parts bar cancelled) | **Done** |
| **Milestone 4** | Benchtop DSO-1062D Oscilloscope (Probes, Trigger, Cursors, Auto Measurements, X-Y, Math & FFT, PNG/CSV Export) | **Done** |
| **Milestone 5** | Semiconductors (Diode, Zener, LED, BJT, MOSFET, Op-Amp), Transformers, MCU, Themes, SPICE Import/Export, Packaging | **Done** |

---

## Building and Packaging (Windows)

Ohmbench ships a PyInstaller `--onedir` spec ([`packaging/ohmbench.spec`](packaging/ohmbench.spec)) and an Inno Setup 6 installer script ([`packaging/installer.iss`](packaging/installer.iss)).

### Why `--onedir` over `--onefile`?

- **Fast startup**: `--onedir` launches `dist\Ohmbench\Ohmbench.exe` directly without extracting Qt 6 DLLs, NumPy, and SciPy into `%TEMP%` on every launch (which `--onefile` must repeat every time).
- **Antivirus & Clean-PC reliability**: Fewer false-positive Windows Defender heuristics than self-extracting single-file stubs, and all Qt LGPL shared libraries remain cleanly replaceable in the application directory.

### Exact Build Commands (PowerShell)

1. Install PyInstaller in your virtual environment (if not already installed):
```powershell
.\.venv\Scripts\pip.exe install pyinstaller
```

2. Build the portable `--onedir` application and create a distributable `.zip`:
```powershell
.\.venv\Scripts\pyinstaller.exe --noconfirm --clean packaging\ohmbench.spec
Compress-Archive -Path .\dist\Ohmbench\* -DestinationPath .\dist\Ohmbench-win64.zip -Force
```

3. *(Optional)* Build the Windows installer `.exe` using Inno Setup 6 (`ISCC.exe`):
```powershell
iscc.exe packaging\installer.iss
```
*(Outputs `packaging\Output\Ohmbench-Setup-0.1.0.exe`.)*

### Clean-PC / Windows Sandbox Verification Steps

1. Copy `dist\Ohmbench-win64.zip` (or run `Ohmbench-Setup-0.1.0.exe`) inside a clean Windows Sandbox or VM without Python installed.
2. Launch `Ohmbench.exe`, verify Dark/Light theme switching, open an example from `File > Examples`, drop and double-click `Oscilloscope (DSO-1062D)`, run simulation (`F5`), and export `.circ` and `.cir` files to confirm all bundled resources resolve via `sys._MEIPASS`.

---

## Project Structure

```text
Ohmbench/
├── LICENSE
├── README.md
├── PROGRESS.md
├── LEARNING_NOTES.md
├── SUGGESTIONS.md
├── AGENT_INSTRUCTIONS.md
├── circuit_simulator_specification.txt
├── requirements.txt
├── pyproject.toml
├── src/circuitsim/
│   ├── main.py
│   ├── app/            main window, docks, menus, settings
│   ├── canvas/         scene, view, wire item, component item
│   ├── components/     resistor, capacitor, inductor, bulb, switch, meters...
│   ├── dialogs/        properties dialog, engineering-notation value parser
│   ├── scope/          scope widget, trigger, measurements, FFT
│   ├── engine/         netlist builder, MNA matrix, integrators, Newton solver
│   ├── io/             JSON save/load, SPICE import/export, image export
│   └── resources/      icons, themes (QSS), example circuits, symbols/iec/
└── tests/              solver tests, parser tests, transient & reactive tests
```

---

## Licensing & Third-Party Libraries

- **Ohmbench** is released under the [MIT License](LICENSE).
- **PySide6 / Qt 6** (Qt for Python / Qt Widgets) is used under the **LGPL v3 (GNU Lesser General Public License v3.0)**; dynamic linking in the `--onedir` Windows distribution allows end users to replace or relink Qt shared libraries cleanly.
- **NumPy** and **SciPy** are licensed under the **BSD 3-Clause License (BSD-3-Clause)**.

---

*Formerly CircuitSim.*
