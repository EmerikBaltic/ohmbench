# Project Progress — Ohmbench (formerly CircuitSim)

## FLASH QUEUE
- Phase 1 Task 1: Housekeeping (.gitignore, switch breakdown voltage hint, label collision avoidance) - [done]
- Phase 1 Task 2: Recent Files (File > Open Recent, 8 files, injected path, missing file handling) - [done]
- Phase 1 Task 3: Example Circuits (File > Examples, 5 bundled circuits, package data, analytic tests) - [done]
- Phase 1 Task 4: Zoom to fit (Ctrl+0) and shortcuts (F5, F6, Shift+F5, info panel, tests) - [done]
- Phase 1 Task 5: Hover tooltips with live values (part & terminal hover, immutable frame, tests) - [done]
- Phase 1 Task 6: Export schematic (File > Export PNG/SVG, QSvgGenerator, white/dark option, tests) - [done]
- Phase 1 Task 7: About dialog, feedback link, crash log exception hook - [done]
- Phase 1 Task 8: Bulb burn-out option (dialog UI, dark+cross drawing, amber note, tests) - [done]
- Phase 1 Task 9: Number format setting (Preferences: Auto/Dot/Comma, display only, tests) - [done]
- Phase 1 Task 10: Continuous integration (.github/workflows/ci.yml, README badge) - [done]
- Phase 2: Step 3C Full Properties Dialog (Identity, parameters, live readout, inductor helpers) - [todo]
- Phase 3: Oscilloscope Plan (measure solver step rate, plan max 50 lines, wait for approval) - [todo]

## MASTER QUEUE
### PART A — HARDENING (branch `main`)
- [x] `A1` (`done (no commit)`): Delete accidental untracked file `"pytest.exe -q"` after reporting first 10 lines (no commit needed).
- [x] `A2` (`done`): Make `tests/test_scope_performance.py` robust on shared CI runners (`os.environ.get("CI")`) while keeping local thresholds strict.
- [x] `A3` (`done`): Diagnose and fix the intermittent Windows Qt teardown access violation (`MainWindow.closeEvent`, `SimulationWorkerController.shutdown`, `tests/conftest.py`).
- [x] `A4` (`done`): Add exact-value Qt-free scope tests in `tests/test_scope_core_exact.py` (`roll`, `single`, `auto_set` [skipped & reported: rule not documented in docstring/PROGRESS.md], `hysteresis`, `holdoff`, `10x probe`, `NOT gate`).
- [x] `GATE A` (`done`): Run `.\.venv\Scripts\pytest.exe -q` 3 times consecutively (`370 passed` x3, 0 failures, 0 access violations) and tag `checkpoint/A-done`.

### PART B — WIRE REDESIGN PHASE 4 (branch `wire-redesign`)
- [x] `B0` (`done`): Switch to `wire-redesign`, fast-forward merge `main`, run fast suite (`369 passed, 1 deselected in 49.48s`), inspect & discard `stash@{0}` without dropping stash entry, inspect `backup/stash-4-1`, write `OLD CODE MAP`.
- [x] `B1` (`done`): HYDROGEN QUEUE `4.1` — `CircuitScene` backed by `WireGraph` (`scene.wire_graph`), attachment-based & legacy `add_wire`, graph undo/redo & `MoveComponentsCommand` merge (`mergeWith`).
- [x] `B2` (`done`): HYDROGEN QUEUE `4.2` — Display (`WireItem` using `circuitsim.wiring.router.route()`, `JunctionItem` when `junction_dot_visible(jid)` / degree >= 3, `FreeEnd` hollow marker, 10 px hit stroke).
- [x] `B3` (`done`, `17aabda`, 373 fast): HYDROGEN QUEUE `4.3` — Netlist via `circuitsim.wiring.netlist.build_circuit` and diagnostics (including `FreeEnd` amber open-wire note).
- [x] `B4` (`done` `c1c1820`, 375 fast): HYDROGEN QUEUE `4.4` — Move, rotate, flip, delete components/wires/junctions preserving `TerminalRef` attachments + 100 random ops + S1 + S2.
- [x] `B5` (`done` `3040025`, 376 fast): HYDROGEN QUEUE `4.5` — Save and load schema v2 (`CircuitDocumentV2`) + v1 auto-migration in `CircuitScene` and `MainWindow` (round-trip v2 + v1 divider/12V+12V + error safety + clean undo stack).
- [x] `B6` (`done` `f0b29dc`, 377 fast): HYDROGEN QUEUE `4.6` — Existing features that touch wires (examples, export PNG/SVG, label collision avoidance, hover tooltips, zoom-to-fit, recent/dirty/undo).
- [x] `GATE B` (`done` `fa985b2`, 378 full): Full test suite (`378 passed in 74.18s`) + offscreen `10 V` / `1 kΩ` / `2 kΩ` voltage divider smoke test (`6.666667 V`), tagged `checkpoint/B-done`.

### OLD CODE MAP (to remove in D5)
- `src/circuitsim/canvas/connectivity.py`: `route_orthogonal_wire`, `point_on_wire`, `point_strictly_inside_wire`, `wires_overlap_collinear`, `find_wires_attached_to_components`, `_route_wire_avoiding_body`, `reroute_wires_for_updated_terminals`, and geometric endpoint-clustering / T-junction point-on-segment pass inside `extract_connectivity`.
- `src/circuitsim/canvas/scene.py`: legacy coordinate wire/junction storage and routing helpers (`_explicit_junctions`, `_active_junction_dots`, `_move_start_wires`, `_suppress_live_reroute`, `_insert_wire_item`, `_remove_wire_item`, `_set_explicit_junction`, `add_routed_wire`, `toggle_junction_at`) once all callers and `D1`–`D4` interactions run on `WireGraph`.
- `src/circuitsim/canvas/wire_item.py`: legacy coordinate-only mode (`WireSpec`, `to_wire_spec()`) without `wire_id` / `Attachment` metadata.
- `src/circuitsim/canvas/commands.py`: coordinate-only wire point dicts in `AddWireCommand`, `ToggleJunctionCommand`, `MoveComponentsCommand`, `RotateComponentsCommand`, `FlipComponentsCommand`, `DeleteItemsCommand`, `PasteItemsCommand` once replaced by `WireGraph` transitions.

### PART C — WIRE REDESIGN PHASE 5 & 6 (branch `wire-redesign`)
- [x] `C1` (`done` `70fe674`, 378 fast): HYDROGEN QUEUE `5.1` — Interactive wire drawing tool re-enabled (live preview via `route()`, `10 px` terminal/junction snap, `FreeEnd`, wire-on-wire notice, `Esc`/right-click cancel, 1 undo step).
- [x] `C2` (`done` `258e8cd`, 379 fast): HYDROGEN QUEUE `5.2` — Waypoint clicks while drawing + `Backspace` removes last waypoint + double-click finish + readable zero-undo rejections (same terminal/junction, duplicate wire, zero-length wire).
- [x] `C3` (`done` `e594c34`, 380 fast): HYDROGEN QUEUE `5.3` — Interactive wire segment dragging (parallel slide updating `hints`, live preview via `route()`, 1 undo step). Acceptance: dragging an interior/movable horizontal or vertical wire segment slides it orthogonally, updates `Wire.hints` and reroutes cleanly via `route()`, and pushes 1 undo step on release (0 if unchanged).
- [x] `C4` (`done` `875b03f`, 381 fast): HYDROGEN QUEUE `5.4` — Editing wire ends (`Acceptance S3`: drag handle, detach, reattach to terminal/junction/wire-split/`FreeEnd`, 1 undo step) + synthetic `QTest` mouse tests.
- [x] `C5` (`done` `4d552ec`, 382 fast): HYDROGEN QUEUE `5.5` — Update/audit existing wire tests (document any replaced coordinate-mechanism test in `PROGRESS.md`) + performance benchmark (`route()` + 50-part drag step <= ~30 ms).
- [x] `C6` (`done` `c053c45`, 383 fast): HYDROGEN QUEUE `6.1` — Verification matrix S1–S5, golden circuit round-trips, update `PROGRESS.md` and `README.md`.
- [x] `GATE C` (`done` `4d973a0`, 384 full x3 + MainWindow divider smoke): Full test suite + tag `checkpoint/C-done`.

### PART D — WIRE REDESIGN STEP 4B (branch `wire-redesign`)
- [x] `D1` (`done`, 384 fast): Split-on-drop (dropping component onto wire splits inline or taps junction in one undo step).
- [ ] `D2` (`in_progress`): Explicit `J` key junction placement on wire at snapped cursor via `split_wire_at`.
- [ ] `D3` (`todo`): Junction dragging (`MoveJunctionCommand(junction_id, old_pos, new_pos)` updating `Junction.pos` and rerouting incident wires).
- [ ] `D4` (`todo`): Copy/paste of selections containing components, wires, and junctions (`Ctrl+C` / `Ctrl+V`).
- [ ] `D5` (`todo`): Remove dead old wire routing/endpoint code listed in `OLD CODE MAP`.
- [ ] `GATE D` (`todo`): Full test suite + `python -m pytest -q` + tag `checkpoint/D-done`.

## HYDROGEN QUEUE (Last Commit: phase-3)
### Phase 0: Verify
- [x] Verify branch `main`, clean status, full suite baseline (`290 passed in 28.08s`), check prior work - *done*

### Phase 1: Deprecation Warnings (on `main`)
- [x] 1.1 Replace deprecated Qt calls (`event.pos()`, `globalPos()`, `x()`, `y()`, etc.) across `src/` with Qt 6 API (`position().toPoint()`, etc.) - *done*
- [x] 1.2 Add `filterwarnings = ["error::DeprecationWarning:circuitsim.*"]` to `pyproject.toml`, list third-party warnings, add filter test - *done (0 third-party warnings remain; 291 passed in 26.32s)*

### Phase 2: Rename to Ohmbench, Stage 1 (on `main`)
- [x] 2.1 Rename Inventory in `PROGRESS.md` - *done*
- [x] 2.2 Single source of truth `src/circuitsim/app_info.py` (no Qt) and user-visible strings - *done*
- [x] 2.3 App Identity (`QCoreApplication` org/app = Ohmbench, one-time migration from `python`, Windows `AppUserModelID`) - *done*
- [x] 2.4 Stored Data (`.circ` backward compatibility and format identifier choice) - *done*
- [x] 2.5 Tests (verify user-visible strings use `APP_NAME`, list legitimate old-name strings) - *done*
- [x] 2.6 Docs (`README.md`, docs, `"Formerly CircuitSim"`, Stage 2 plan in `PROGRESS.md`) - *done*

### Phase 3: Mutation Check (on `main`)
- [x] 3a Component empty value plate mutation vs `test_reproduce_and_verify_no_stray_background_under_led` - *done (FAILED: 2592 stray pixels, restored cleanly)*
- [x] 3b Problems panel close button blank draw mutation vs `test_no_blank_buttons_or_icons_in_both_themes` - *done (FAILED: Blank button ProblemsCloseBtn, restored cleanly)*
- [x] 3c Light theme palette missing attribute mutation vs `test_every_palette_attribute_in_code_exists_in_both_themes` - *done (FAILED: missing symbol_label, restored cleanly)*
- [x] 3d Swap LED anode/cathode terminals vs `test_led_palette_entry_and_metadata` - *done (FAILED: swapped terminals & SVG viewBox, restored cleanly)*
- [x] 3e Unguarded `paint()` exception vs `test_paint_safety_guard_on_component_wire_and_junction` - *done (FAILED: exception uncaught, restored cleanly)*

### Phase 4: Wire Redesign, Step 4A-1 (branch `wire-redesign`)
- [x] 4.1 Scene state (`WireGraph`) and undo/redo + move merge + `add_wire` between attachments - *done*
- [x] 4.2 Display (`WireItem` using `route()`, `JunctionItem` degree >= 3, `FreeEnd` hollow marker, 10 px hit stroke, wire tool notice) - *done*
- [x] 4.3 Netlist (`circuitsim.wiring.netlist`) and diagnostics (including `FreeEnd` amber note) - *done*
- [x] 4.4 Moving, rotating, flipping, deleting (100 random ops + S1 + S2) - *done (`c1c1820`, 375 fast)*
- [x] 4.5 Save and load schema v2 + v1 migration - *done (`3040025`, 376 fast)*
- [x] 4.6 Existing features that touch wires (examples, export, labels, tooltips, zoom-to-fit, recent/dirty/undo) - *done (377 fast)*
- [x] CHECKPOINT 4A-1 DONE - *done (`f0b29dc`, 378 full; `checkpoint/B-done`)*

### Phase 5: Wire Tool, Step 4A-2 (branch `wire-redesign`)
- [x] 5.1 Wire drawing tool re-enabled (live preview, 10 px snap, `JunctionRef`, `FreeEnd`, wire-on-wire notice) - *done (`70fe674`, 378 fast)*
- [x] 5.2 Rejections (same terminal, duplicate wire, zero-length wire) + waypoints/`Backspace` - *done (`258e8cd`, 379 fast)*
- [x] 5.3 Interactive wire segment dragging (`hints` parallel slide, live preview via `route()`, 1 undo step) - *done (`e594c34`, 380 fast)*
- [x] 5.4 Editing wire ends (`Acceptance S3`: handle, detach, reattach/`FreeEnd`, 1 undo step) + synthetic `QTest` mouse tests - *done (`875b03f`, 381 fast)*
- [x] 5.5 Wire test audit (`test_wire_snapping.py` [2], `test_rotation_and_moving_wires.py` [8], `test_connectivity_and_editing.py` [9] all pass unchanged on top of `WireGraph`; supplemented by `test_phase5_wire_tool.py`) + performance benchmark (`route()` analytic <= 2 ms / A* <= 15 ms, 50-part drag step median <= 30 ms, env-configurable) - *done (`4d552ec`, 382 fast)*

### Phase 6: Documentation & Report (branch `wire-redesign`)
- [x] 6.1 Update `PROGRESS.md` (built, audited tests, next tasks S4/S5 & Step 4B `D1`–`D5`) and `README.md` (`test_phase6_1_golden_roundtrips_and_s1_s5_matrix` S1–S5 + Schema v2 & bundled example round-trips; zero existing tests weakened or replaced) - *done (`c053c45`, 383 fast)*
- [x] CHECKPOINT 4A-2 / GATE C DONE - *done (384 full x3, offscreen `MainWindow` divider `V_mid=6.666667 V`; `checkpoint/C-done`)*

## Current Status: Milestone 2 & Milestone 3 Step 3A Complete, Phase 1 Quick Wins Tasks 1-10 Done, Startup & Shutdown Fixes Done, LED Lifecycle Complete, Paint Safety & Crash Handler Hardening Complete, Meter Loading & Ideal Modes Complete (289 Passing Tests)

## Stray Background Under LED, Problems Panel Close Button & Meter Loading (Items 1–3)
- **Item 1: Stray Background Under the LED**:
  - *Root Cause*: In `ComponentItem._paint_impl()`, components with `show_value=True` but empty formatted value text (like an LED when simulation is stopped) evaluated `QFontMetrics.boundingRect(val_rect.toRect(), text_align, "")` to a 0-width rectangle located 25–43 px below the component origin. Calling `.adjusted(-3, -1, 3, 1)` expanded that zero-width bounding box into an 84-pixel rectangle `QRect(-3, 27, 6, 14)` which was painted with `painter.fillRect(val_box, palette.canvas_bg)` onto transparent scene space ~20 px below the LED.
  - *Fix*: Guarded value/label plate painting to require non-empty text after stripping, and tightened `boundingRect()` to hug the visual symbol within 2 px.
- **Item 2: Problems Panel Close Button & Glyph Fixes**:
  - Replaced native Windows "Segoe UI" U+2715 (`✕`) glyphs—which rendered as empty tofu boxes ($\square$) on Windows—with crisp vector cross rendering in both `ProblemsCloseButton.paintEvent` and `make_severity_icon("error")`.
  - Added accessible close button with focus outline, Space/Enter keyboard activation, and tooltip `"Close Problems panel (F8)"`.
- **Item 3: Meter Loading (Visible and Optional, Engine Defaults Unchanged)**:
  - *Ideal & Realistic Meter Modes*: Added `is_ideal: bool` to `Voltmeter` and `Ammeter`. Ideal voltmeters draw 0 A (infinite input resistance) without generating false floating-node warnings. Ideal ammeters model a 0 V branch constraint with 0 V drop.
  - *Dialog Presets*: Added input resistance presets to `QuickValueDialog` for Voltmeter (`Ideal`, `10 Mohm (typical multimeter)`, `1 Mohm`, `Custom`) and series resistance presets for Ammeter (`Ideal (0 V branch)`, `1 mohm`, `Custom`). Presets update live during simulation and support full undo/redo.
  - *Edit > Preferences*: Added `"New meters are: Ideal / Realistic"` setting defaulting to `"Ideal"` for newly placed palette meters, while engine constructors (`Voltmeter(...)`, `Ammeter(...)`) retain realistic defaults ($10\text{ M}\Omega$, $1\text{ m}\Omega$) so existing tests and bundled examples are unaffected.
  - *Voltmeter Loading Warnings*: Added network Thévenin resistance computation to evaluate voltmeter loading. When loading changes the measured voltage by $\ge 0.1\%$, the hover tooltip and editor dialog display: `"VM1 draws 30 uA from the 10 kohm resistor; the reading is 0.1% lower than the unloaded voltage."`
  - *Educational Notes*: Added dedicated educational section in `LEARNING_NOTES.md` with hand-calculated analysis (30 mA source into 10 kΩ giving 299.70 V vs 300 V, 30 A source current divider giving 29.97 A vs 30 A, GMIN shunt analysis at high voltages) and 3 student exercises with complete step-by-step solutions.


## Theme Palette Completeness, Crash Handler Protection & Paint Safety (Items 1–7)
- **Item 1: Reproduction Test**:
  - Added offscreen Qt reproduction test (`test_reproduce_led_paint_palette_attribute_error`) verifying that selecting and hovering an LED on canvas in dark and light themes exercises terminal tags and captures the missing `symbol_label` attribute error in both themes.
- **Item 2: ThemePalette Completeness**:
  - Added missing `symbol_label` color token to `ThemePalette` definition (`#8B949E` in `DARK_PALETTE`, `#57606A` in `LIGHT_PALETTE`).
  - Added AST-based static/dynamic test `test_every_palette_attribute_in_code_exists_in_both_themes` guaranteeing that all color tokens accessed on `palette` across the codebase exist in both dark and light palettes.
- **Item 3: Exhaustive Smoke Test**:
  - Smoke test (`test_smoke_render_every_palette_entry_all_orientations_states_and_themes`) covering all 12 palette entries across all 8 orientations, selected/unselected, hovered/unhovered, running/stopped (with all 6 LED states: off, dim, on, over, burned, reverse) across both dark and light themes (1,792 permutations in ~2.3s) with 0 exceptions reaching the exception hook.
- **Item 4: Crash Handler Loop Prevention & Modal Safety**:
  - Signature-based error deduplication `(exception_type, message, top_frame)` with 60-second throttling window.
  - Re-entrancy guard preventing duplicate or nested modal dialogs.
  - Deferred dialog execution via `QTimer.singleShot(0, ...)` ensuring errors in `paint()` never open modal event loops during a paint traversal.
  - Added "Quit" button to error dialog and report repeat counts (`Occurrences: N`).
- **Item 5: Paint Safety Fallback**:
  - Wrapped `ComponentItem.paint()`, `WireItem.paint()`, `JunctionItem.paint()`, and `CircuitScene.drawForeground()` in safety guards.
  - Caught drawing exceptions are logged once per item and replaced with a red dashed bounding rectangle with a bold "!" indicator.
  - Guarantees nothing ever propagates out of a virtual `paint()` override to cause C++ Shiboken aborts or paint loops.
- **Item 6: File Storage Location & Directory Naming Investigation**:
  - *Report*: Settings, recent files, and `crash.log` are stored under `get_default_data_dir()`. On Windows, this resolves to `C:\Users\emeri\AppData\Roaming\python` (`settings.json` and `crash.log`).
  - *Why the folder is named "python"*: `QStandardPaths.writableLocation(AppDataLocation)` checks `QCoreApplication.applicationName()`. Because neither `QCoreApplication.setApplicationName("CircuitSim")` nor `QCoreApplication.setOrganizationName("Ohmbench")` is set during initialization in `main.py`, Qt falls back to the running binary name `sys.executable` (`python.exe` -> `"python"`). No working code was modified; the upcoming rename task will set the organization and application names.

## LED Component & Problems Panel (Items 1–6)
- **Item 1: LED Symbol and Palette**:
  - Added new palette category **Semiconductors** with **LED** (`80 x 40 px`, terminals `anode` at `(0, 20)` and `cathode` at `(80, 20)`).
  - Designed IEC 60617 LED symbol with distinct cathode flat cut on the case outline (Variant A: flat vertical cut on cathode rim, clear across all 8 orientations).
  - Pure-Python geometric marks in `led_marks()` with verified cathode-flat proximity tests.
  - Preview image generated and committed at `docs/design/led_variants.png`.
- **Item 2: Colour Selection & Datasheet Forward Voltages**:
  - Integrated LED color selector into `QuickValueDialog` with 7 real-world presets + Custom: Red (2.0 V), Orange (2.0 V), Yellow (2.1 V), Green (2.2 V), Pure green (3.0 V), Blue (3.2 V), White (3.2 V).
  - Swatch icons rendered beside each colour choice in the combo box.
  - Informative datasheet label ("typical datasheet values; real LEDs vary by about +-0.3 V") that auto-switches to "Custom" on manual voltage edits.
  - Configurable `rated_current` (default 20 mA), `max_current` (default 30 mA), and optional `burnout_enabled` / `burnout_current` checkbox and threshold.
  - Full undo/redo support and `.circ` document serialization.
- **Item 3: LED Drawing & Perceptual Glow**:
  - Live brightness follows perceptual human-eye response: $I_{\text{vis}} = \sqrt{I / I_{\text{rated}}}$.
  - Realistic sharp sub-threshold cutoff below $10\,\mu\text{A}$ (off).
  - Over-current warning badge: amber exclamation mark badge rendered when $I > I_{\text{max}}$.
  - Burnout rendering: darkened case body with distinct cross when burned out ($I > I_{\text{burnout}}$).
  - Diagnostic outline coloring: amber box for warnings (reversed, over-current, missing resistor) and red box for fatal errors.
  - Live tooltip shows Name, Colour, Voltage, Current, and State (`Off`, `On`, `Over-current`, `Burned out`, `Reverse-biased`).
- **Item 4: Problems Panel (`View > Problems` / `F8`)**:
  - Bottom panel dock with fixed 120 px initial height, 'X' close button, and `F8` shortcut.
  - Diagnostics restructured into clear 3-part format: What happened, Why it happened, and How to fix it (referencing component names and numbers).
  - Tree columns: Severity (icon + "Error"/"Warning"), Message, Components affected.
  - Clicking a problem selects and centers the affected component on the canvas.
  - Status bar shows short problem count ("2 problems") with clickable action.
  - Auto-opens on simulation errors unless closed by the user during the current run.
  - Strict canvas invariance guaranteed (`NoAnchor` view prevents zoom or pan shift on panel open/close).
- **Item 5: Bundled LED Examples**:
  - `File > Examples > LED with a resistor`: 5 V DC source, 150 Ω resistor, Red LED (calculating $I = 20\text{ mA}$).
  - `File > Examples > LED colours`: 12 V rail powering 7 parallel branches showcasing all standard LED colours with calculated current-limiting resistors.
- **Item 6: Integration Tests & Documentation**:
  - Offscreen Qt integration test suite with timeout verifying palette placement, snapping, properties editing with undo/redo, reversed/no-resistor diagnostics, problems panel interaction, all 8 orientations, and `.circ` save/load round-tripping.

## Startup & Shutdown Bug Fixes (Items 1–6)
- **Bug (a) Empty Start Canvas**:
  - *Origin*: In `src/circuitsim/main.py` lines 106–111 inside `run_gui()`, hardcoded placement calls (`scene.place_new_component(...)`) created a 5-component demo circuit (`dc_voltage`, `switch`, `resistor`, `lightbulb`, `ground`) on every launch.
  - *Commit Origin*: Commit `06c2d72` ("Milestone 2 Step 2A: main window, IEC symbol loader/preview, palette, and schematic canvas").
  - *Root Cause*: Placing components via `place_new_component` pushed commands onto `scene.undo_stack`, making `undo_stack.isClean() == False`, marking the document modified with an asterisk (`*`) in the title bar, and triggering the "save changes?" prompt whenever the user closed a fresh app without touching anything.
  - *Fix*: Removed the hardcoded startup placement in `src/circuitsim/main.py`. Canvas now starts completely empty, undo stack is clean, title has no asterisk, and closing immediately exits without prompting.
  - *Test Impact*: 0 existing tests depended on startup components (all existing tests construct their own isolated circuits). Added new offscreen Qt test `test_fresh_window_starts_empty_and_clean_with_no_close_prompt` in `tests/test_circ_io.py`.

- **Bug (b) Shutdown Exception (libshiboken C++ object already deleted)**:
  - *Origin & Cause*: In `MainWindow.__init__`, `self.circuit_scene.undo_stack.cleanChanged` was connected to `lambda _clean: self._update_window_title()`. When closing the application after selecting "Discard", Qt destroys child objects during teardown. When `CircuitScene` destroys its child `QUndoStack`, the C++ destructor `QUndoStack::~QUndoStack()` cleans uncommitted commands, emitting `cleanChanged(bool)`. The connected slot invoked `self._update_window_title()` -> `has_unsaved_changes()` -> `self.circuit_scene.undo_stack.isClean()`. Because the underlying C++ `QUndoStack` was already partially destroyed, Shiboken raised `RuntimeError: libshiboken: Internal C++ object (PySide6.QtGui.QUndoStack) already deleted.`. The global unhandled exception hook (`crash_handler._excepthook`) then displayed a modal critical error dialog during application exit.
  - *Fix*:
    1. Reordered shutdown sequence in `MainWindow.closeEvent`: prompt for unsaved changes first; if cancelled, ignore event; if discarded/saved, set `self._is_closing = True`.
    2. Stop and wait for the background simulation worker thread (`self.sim_controller.shutdown()`).
    3. Explicitly disconnect `self.circuit_scene.undo_stack.cleanChanged` and worker signals before widget teardown.
    4. Guard `_update_window_title()` and `has_unsaved_changes()` with `if self._is_closing: return`.
    5. Set `set_shutting_down(True)` in `crash_handler` to suppress modal `QMessageBox` popups during shutdown while still logging any shutdown exceptions to `crash.log` and stderr.
  - *Tests*: Added `test_window_shutdown_cleanliness_all_cases` (verifying all 8 shutdown scenarios: discard, save, cancel, clean start, running simulation, loaded file, new with discard, rapid double close) and `test_quiet_shutdown_error_logging_without_dialog`.

### What Is Done
- **Environment, Licensing & Editable Package Installation**:
  - Local Python 3.12 virtual environment (`.venv`) with `PySide6`, `numpy`, `scipy`, and `pytest`.
  - Project is installed in editable mode (`pip install -e .`) and includes an MIT [`LICENSE`](LICENSE) and updated [`README.md`](README.md).
- **Milestone 1 — Core Physics & MNA Engine (`src/circuitsim/components/`, `src/circuitsim/engine/`, `src/circuitsim/dialogs/value_parser.py`)**:
  - Engineering-notation parser (`parse_spice_value`, `format_engineering_value`).
  - All 11 Milestone 1 & 2 components (`Ground`, `Resistor`, `Capacitor`, `Inductor`, `DCVoltageSource`, `DCCurrentSource`, `ACVoltageSource`, `Switch` with $350\text{ V}$ contact breakdown arc clamp, `Lightbulb`, `Voltmeter`, `Ammeter`).
  - DC and Transient MNA solver (`Circuit`, `MNASolver`) with topology validation, `thevenin_resistance`, Backward Euler switching/arc settling, and Trapezoidal integration.
- **Milestone 2 — Step 2A: Main Window, IEC Symbol System, Palette & Schematic Canvas**:
  - 11 IEC symbol JSON metadata files (`src/circuitsim/resources/symbols/iec/`), pure-Python symbol geometry & tolerant SVG loader (`src/circuitsim/canvas/symbol_geometry.py`), hybrid SVG/procedural IEC renderer (`symbol_renderer.py`), symbol preview tool (`tools/symbol_preview.py`), and dockable `MainWindow`.
- **Milestone 2 — Step 2B: Connectivity Engine, Orthogonal Wiring, Full Undo/Redo & Usability**:
  - **Pure-Python Connectivity & Netlist Extractor (`src/circuitsim/canvas/connectivity.py`, zero Qt imports)**:
    - Union-Find netlist extractor (`extract_connectivity`, `CircuitScene.build_circuit()`) enforcing exact schematic connectivity rules (wire endpoints on terminals, crossing wires do not connect without a T-junction or explicit junction dot, isolated junction dots not on any wire are ignored and never create nets or false warnings, attached wire deletion/re-routing, zero-length/duplicate/collinear wire normalization).
  - **Interactive Orthogonal Wiring & Full `QUndoStack` Editing (`src/circuitsim/canvas/wire_item.py`, `commands.py`, `scene.py`, `view.py`)**:
    - Rubber-band selection, group move, Rotate 90° CW (`R`), Flip Horizontal (`F`), Delete (`Del`/`Backspace`), Cut/Copy/Paste, and full `QUndoStack` Undo/Redo (`Ctrl+Z`/`Ctrl+Y`) including parameter edits.
- **Milestone 2 — Step 2C & Post-2C Polish: Live Simulation Worker, Canvas Readouts, Interactive Switch, Bulb Glow, Problem Highlighting & Usability Fixes**:
  - **Pure-Python Simulation Session & Process-Ready Protocol (`src/circuitsim/engine/simulation_session.py`, zero Qt imports)**:
    - `SimulationSession` and `SimulationBackendProtocol` handle schematic-to-solver execution (`start`, `pause`, `stop`, `step_chunk`, `toggle_switch`, `update_component_parameter`), transport-control state mapping (`compute_transport_controls_state`), and emit immutable, picklable `SimulationFrame`, `ComponentTelemetry`, and `ProblemDiagnostic` dataclasses.
  - **Background `QThread` Worker (`src/circuitsim/app/sim_worker.py`)**:
    - `SimulationWorkerController` runs `_SessionThreadWorker` on a dedicated background `QThread` communicating strictly via queued Qt signals so the main GUI thread never freezes during simulation, and stops cleanly when `MainWindow` closes.
  - **Value Input, Live Editing, Undo/Redo, Transport States, Unconnected Parts & Selection Outline**:
    - **Value Parser (`value_parser.py` & `quick_edit_dialog.py`)**: Accepts single decimal comma (`"5,65 G"`), spaces before prefix (`"4.7 k"`), uppercase `"K"` as kilo (`"10K"`), shows live parsed preview (`= 5.65 GΩ`) beneath each input field, and displays clear plain-language error messages.
    - **Live Editing & Undo/Redo**: Double-clicking any component opens `QuickValueDialog` with a note that changes apply immediately while running (`t = 0` note on capacitor initial voltage), and pushes an undoable `EditComponentParametersCommand` (`Ctrl+Z` / `Ctrl+Y`) that updates the component model, canvas label, and live simulation both when stopped and when running.
    - **Run / Pause / Stop Visual States**: Distinct green active Run, amber active Pause (`▶ Resume`), red Error outline, and left-hand status bar indicator (`● Running`, `❚❚ Paused`, `■ Stopped`) across Dark and Light themes.
    - **Unconnected Parts & Ground Hint**: Components with no wires attached are dimmed (`opacity = 0.45`) and ignored with an amber warning (`"B1 is not connected and is ignored"`) so the rest of the circuit runs; connected islands with no path to ground still raise a floating-node error; missing ground reports `"No ground connected — Add a Ground part and wire it to the circuit"`.
    - **Selection Outline**: Component selection and diagnostic boxes stay strictly clear of upright name (`R1`) and value (`1 kΩ`) labels across all rotations (`0°`, `90°`, `180°`, `270°`).

- **Milestone 3 - Step 3A: Save / Load `.circ` Files (done)**:
  - [`src/circuitsim/io/circ_serializer.py`](src/circuitsim/io/circ_serializer.py) (no Qt): versioned JSON (`"format": "circuitsim"`, `"schema_version": 1`) storing components (type, name, label, show_value, position, rotation, flip, parameters, extra settings), wires, junction dots and view state (theme, zoom, pan, simulation speed). Strict validation with plain-language `CircuitFileError` messages (empty, truncated, wrong format, newer version, bad values).
  - Atomic save (temp file in the same folder, flush to disk, then `os.replace`) with one backup `name.circ.bak`.
  - `CircuitScene.load_document` builds everything first and swaps only if it all worked; `CircuitScene.to_document` snapshots the schematic. Loading clears the undo stack.
  - File > New / Open / Save / Save As (`Ctrl+N`, `Ctrl+O`, `Ctrl+S`, `Ctrl+Shift+S`), toolbar New/Open/Save, title shows `name.circ*` for unsaved changes, prompts before New / Open / closing.
  - New Qt-free [`components/factory.py`](src/circuitsim/components/factory.py) (the "make a default component" function moved out of the canvas so the loader can use it).

- **Milestone 3 — Design Overrides & Refinements (Items 1–4)**:
  - **Item 1: Saved Parts Bar Removed (Step 3B Cancelled)**: The 10-slot saved parts toolbar, placeholder buttons, shortcuts and package have been completely removed. Digit keys are no longer reserved for saved parts. (Design decision overrides specification and AGENT_INSTRUCTIONS.md; noted in SUGGESTIONS.md for future redesign if requested).
  - **Item 2: No Fake Oscilloscope**: The empty "Oscilloscope (4-Channel)" dock widget and unused placeholder actions/menus have been removed. Window layout restoration (`restoreState`) guarantees stale saved states cannot bring the dock back. The `circuitsim/scope/` package structure remains preserved for Milestone 4.
  - **Item 3: Lowest Free Number for Component Names**: New and pasted components now consistently receive the lowest unused number for their prefix (`R`, `C`, `L`, `V`, `VM`, `A`, `B`, `SW`, `GND`, etc.). Deleting a part frees its number for the next part added; undoing restores original names; names stay unique at all times; save/load round-trips preserve names.
  - **Item 4: Small Visual Fixes**: The internal letter in Voltmeter (`V`) and Ammeter (`A`) symbols stays upright across all rotations (`0°`, `90°`, `180°`, `270°`) and horizontal flip (`F`). Components are placed above wires (`Z = 20.0` vs `5.0`), and value readouts (e.g. `"11.11 mA"`) and name labels include a canvas background plate so wires never cut through or obscure the text.

- **Quality & Reliability Fixes (Items A–E)**:
  - **Item A (Polarity Regressions & Transform Chain)**: Added 40 pure-Python regression tests in `tests/test_polarity.py` covering all 8 orientations (`rot` in `{0, 90, 180, 270}`, `flip` in `{False, True}`) for DC Voltage Source, DC Current Source, Voltmeter, Ammeter, and Two-source series chain. Unified `SymbolLibrary._paint_procedural_iec` transform order (`scale(-1, 1)` then `rotate(rot)`) with `transform_terminal_position` so drawing and netlist extraction use identical geometry.
  - **Item B (Hit Areas & Bounding Rectangles)**: Implemented exact `shape()` on `ComponentItem` with `QPainterPath` (using `WindingFill`) matching rotated/flipped symbol body and 6 px terminal clearance. Tightened `boundingRect()`. Excluded label plates from mouse clicks so wires behind labels remain selectable. Bound `prepareGeometryChange()` to `rotation_deg` and `flipped_h` setters. Verified via `tests/test_hitboxes.py`.
  - **Item C (Wire Re-routing & Detour on Rotate/Move)**: Implemented collision-avoidance routing in `_route_wire_avoiding_body` in `src/circuitsim/canvas/connectivity.py`. Direct routes that intersect the component body are diverted via orthogonal 40 px clear bypass paths (terminal `a` on negative side, `b` on positive side), preventing tangled overlapping shorts. Added 8 regression tests in `tests/test_rotation_and_moving_wires.py` verifying full 360° cycles, dragging, group rotation, voltage preservation, and undo/redo.
  - **Item D (Wire Terminal Snapping)**: Added `TERMINAL_SNAP_RADIUS_PX = 10.0` (half grid cell) snapping in `add_wire`, hovering terminal indicator, and nearest terminal selection. Tested in `tests/test_wire_snapping.py` across 0, 1, 5, 8, 9 px (connects) and 11 px (stays off).
  - **Item E (Save / Load Round-Trip Preservation)**: Added `test_save_load_rotated_flipped_circuit_preserves_wires_polarity_and_voltages` in `tests/test_circ_io.py`. Verified that saving and reloading circuits with rotated and flipped parts preserves exact wire connections, terminal polarities, and simulation voltages.

- **Item F (Test Suite Speed Analysis)**:
  - **Profile (`pytest --durations=10`)**: 180 passed in ~43.5 s. The top 10 slowest tests account for ~23 s (~53% of total time), with the slowest being UI tests in `tests/test_simulation_session_and_step2c.py` (2.7s–3.0s each), `test_hitboxes.py` (1.95s), and `test_circ_io.py` window tests (1.85s).
  - **Root Cause**: Tests involving Qt widgets are executed in isolated subprocesses (`_run_qt` via `sys.executable -c`) to enforce the strict architectural rule that pure-Python physics, connectivity, and serializer modules have zero Qt imports in `sys.modules`. On Windows, each subprocess invocation incurs Python startup and dynamic loading of large PySide6 DLLs (~1.5–2.5 s per process). By contrast, all 130+ pure-Python unit tests complete in under 2 seconds combined (<0.02 s each).
  - **Optimization Assessment**: Running Qt tests in-process would reduce test suite duration to <10 s, but would contaminate `sys.modules` and weaken the isolation tests that prove physics and IO code remain Qt-free. The current architecture should be left alone because 43 s is fast enough for CI/local verification while providing absolute architectural guarantees.

- **Recent Quality & Usability Tasks**:
  - **Item 1 (Commit List for Items A–E and Docs)**:
    - `eee01f6`: `Item A: Fix painter transform order to match symbol terminal geometry and add polarity regression tests`
    - `9368dd9`: `Item B: Implement precision shape and tight boundingRect matching rotated symbols, ensuring label plates do not intercept clicks`
    - `95efe4f`: `Item C: Safe orthogonal wire re-routing avoiding component body penetration and tangle shorts on rotation and dragging`
    - `36b90ea`: `Item D: Precise 10 px wire terminal snapping, hover highlighting, and connection guarantees`
    - `c5f6fcc`: `Item E: Verify save and load round-trip on rotated and flipped circuits preserves wire connections, polarity, and simulation voltages`
    - `f86a89e`: `Item E: Unify symbol rendering and terminal coordinate transforms with save/load regression tests` (Item E shared two commits)
    - `f3539df`: `Item F and G: Document test suite speed profiling analysis and explicit wiring architecture plan` (Docs for F and G shared commit f3539df)
  - **Item 2 (Canvas View & Dock Stability on Selection / Clicks)**:
    - *Root Cause*: Dock width dynamically followed `info_label.sizeHint()` across selection changes (`DEFAULT_SHORTCUTS_HELP_TEXT` vs component text), resizing the viewport. Combined with `AnchorViewCenter`, this shifted the scene under the cursor by up to 58 px and triggered unintended 20 px grid-snap dragging on click.
    - *Fix*: Locked `info_dock` to fixed width (280 px), wrapped contents inside `QScrollArea`, set `info_label` size policy to `Ignored` horizontally with `wordWrap=True`, locked `palette_dock` (240 px) and `statusBar` (height 26 px, grip disabled), switched `CircuitView` to `NoAnchor`, and introduced a 4 px drag threshold suppressing mouse move jitter on clicks. Verified via `tests/test_view_stability.py`.
  - **Item 3 (Unified Drawing & Netlist Polarity via `polarity_mark_positions`)**:
    - Implemented pure-Python `polarity_mark_positions(part_type, rotation_deg, flipped_h, center_px)` in `symbol_geometry.py` using `transform_terminal_position` as the single geometric source of truth.
    - Updated `symbol_renderer.py` to paint `+`, `-`, current arrow and meter polarity in upright coordinates directly from this function with optional `polarity_color`.
    - Added `tests/test_drawing_polarity.py` (8 pure-Python hand-worked tests + offscreen Qt test asserting `+` mark pixel centroid in rendered `QImage` is nearer `pos` than `neg` across all 8 orientations).
  - **Item 4 (Revised Plan: Attachment-Based Wires & Junctions — Plan Only, No Code)**:
    ```text
    1. Data Model: Wire has two ends. Each end is an Attachment: TerminalRef(comp_name, term_name), JunctionRef(junction_id), or FreeEnd(x, y). Junction has id and pos; created only explicitly (placed by user or splitting a wire when ending on it).
    2. Electrical Connectivity: Pure Union-Find over TerminalRefs and JunctionRefs connected through Wires (circuitsim.wiring.connectivity). Coordinates play zero part; crossing/overlapping wires never connect; moving components/junctions never alters connectivity.
    3. Route Generation: Pure function route(start_pos, end_pos, obstacles) -> tuple[PointPx, ...] (no Qt, deterministic, stable while dragging); replaces the 40 px detour heuristic. Waypoint drags stored as optional routing hints only.
    4. Visual Projection: JunctionItem drawn as dot only when degree >= 3; merges into a bend when degree drops to 2. JunctionItems are selectable and draggable, with attached wires following live. FreeEnd drawn with hollow marker; snap radius attaches to terminal.
    5. Schema v2 & Migration: Schema v2 stores explicit attachments and junction list. V1 loader runs legacy geometric extraction once to compute connectivity, generates attachments, and upgrades seamlessly; verified with identical simulation voltages.
    6. Tests Affected: test_connectivity_and_editing.py adapts from raw coordinate arrays to Wire(start_ref, end_ref); crossing-wire and collinear tests become structural assertions. V1 tests in test_circ_io.py verify migration.
    7. Replaces from C & D: Replaces _route_cuts_box detour heuristic and geometric wire endpoint mutation on move (Item C); adapts 10 px snap (Item D) to bind FreeEnd to TerminalRef.
    8. Stepped Commits & Size:
       - Step 1 (Medium): Pure-Python wiring model (TerminalRef, JunctionRef, FreeEnd, Wire, Junction, Union-Find netlist builder) with graph unit tests.
       - Step 2 (Medium): Pure-Python route(start, end, obstacles) orthogonal pathfinder replacing detour heuristic.
       - Step 3 (Small): File format schema v2 serializer and v1 migration runner with voltage-equivalence tests.
       - Step 4 (Large): Canvas integration (WireItem, JunctionItem, split-on-drop, degree-2 merge, free-end hollow markers, drag-move tracking).
    9. Risks:
       - Interactive drag oscillation/hunting between symmetrical routes. Mitigation: Route caching with H-first/V-first orientation latching.
       - V1 migration edge cases with unattached floating wire clusters. Mitigation: Cluster grouping via legacy Union-Find before attachment synthesis.
    ```

- **Explicit Wiring Redesign (Steps 0 to 4 Done)**:
  - **Item 0: Single Source of Truth for Terminals & Drag vs. Jitter**:
    - Added comprehensive regression tests in [`tests/test_terminal_source_of_truth_and_drag.py`](tests/test_terminal_source_of_truth_and_drag.py).
    - Verified all 88 combinations (11 components $\times$ 4 rotations $\times$ 2 flips) prove that `item.terminal_scene_positions()`, `spec.resolve_terminals_px()`, `shape().contains()`, terminal snapping, and rendered terminal dot pixels from `item.paint()` all take their coordinates strictly from `transform_terminal_position`.
    - Tested that deliberate 40 px drag moves a part by 2 grid cells (+40 px) and creates an undo command, while 3 px mouse jitter moves by 0 px and creates zero undo commands.
  - **Item 1: Pure-Python Wiring Model & Union-Find Netlist Builder (`src/circuitsim/wiring/`)**:
    - Built [`src/circuitsim/wiring/model.py`](src/circuitsim/wiring/model.py) with zero Qt imports: `TerminalRef(component, terminal)`, `JunctionRef(id)`, `FreeEnd(x, y)`, `Wire(id, end_a, end_b, hints)`, `Junction(id, pos)`, and `WireGraph`.
    - `WireGraph` is an immutable frozen dataclass; every operation (`add_wire`, `remove_wire`, `add_junction`, `split_wire_at`, `remove_junction`, `remove_component`) returns a new graph.
    - Automatic cascading merge rules: degree 2 junction merges 2 wires into 1 (storing junction pos in hints); degree 1 junction converts wire end to `FreeEnd`; degree 0 junction is deleted; duplicate wires are deduplicated.
    - Built [`src/circuitsim/wiring/netlist.py`](src/circuitsim/wiring/netlist.py): Union-Find over explicit attachments producing solver-ready `Circuit` with deterministic node names ("0", "N1", ...).
    - Added 20 unit and equivalence tests in [`tests/test_wiring_model_and_netlist.py`](tests/test_wiring_model_and_netlist.py) verifying immutability, merge rules, crossing wires remaining separate, 200 random operations maintaining validity, diagnostics (`NoGroundError`, `FloatingNodeError`, `VoltageSourceLoopError`), and exact voltage divider, RC, 12V+12V, and switch-bulb equivalence against analytic calculations.
  - **Item 2: Pure-Python Orthogonal Wire Router with Latching (`src/circuitsim/wiring/router.py`)**:
    - Built `route(start, end, start_dir, end_dir, obstacles, previous=None)` with zero Qt imports.
    - Enforces orthogonal paths, outward departure in `start_dir`, inward arrival in `-end_dir`, obstacle collision avoidance (inflated bounding boxes, exempting attached terminals), deterministic <= 2 bend candidates, minimal-bend A* grid search fallback, and orientation latching from `previous` routes.
    - Added 10 unit tests in [`tests/test_wiring_router.py`](tests/test_wiring_router.py) covering straight, L-shape, Z-shape, facing/opposed terminals, obstacle avoidance, collinear simplification, and latching.
  - **Item 3: Schema v2 File Format & v1-to-v2 Migration (`src/circuitsim/wiring/serializer.py`)**:
    - Defined Schema v2 storing explicit attachments (`terminal`, `junction`, `free`), explicit junctions, and view state.
    - Implemented `migrate_v1_to_v2` running geometric extraction once, slicing wires at internal vertices and segments, preserving separation of crossing wires without dots, and cleaning stray dots.
    - Added 9 unit tests in [`tests/test_wiring_v2_migration.py`](tests/test_wiring_v2_migration.py) verifying voltage divider, RC circuit, 12V+12V chain, rotated/flipped parts, crossing wire separation, v2 save/load round-tripping, and error handling.
  - **Item 4: Small Usability & Geometry Fixes**:
    - Switch toggle marks file modified: Clicking a switch on the canvas marks the document modified (`has_unsaved_changes() = True`, window title asterisk `*`), but does NOT push to `QUndoStack` (not undoable). Saving clears the modified state.
    - Selection outline clearance: Adjusted Ground label positioning below the earth bars (`label_offset[1] >= 0`), providing 30 px clearance and ensuring Voltage Source selection outline never touches `GND2` label.
    - Added offscreen Qt tests in [`tests/test_switch_dirty_and_ground_label_clearance.py`](tests/test_switch_dirty_and_ground_label_clearance.py).

## STEP 4 REQUIREMENTS (not built yet)

Everything Step 4 needs to do to integrate the wire redesign with the schematic canvas:

1. **Canvas `WireGraph` Integration**:
   - Replace raw `_wire_items` list and `_explicit_junctions` coordinate set in `CircuitScene` with `WireGraph` as the single source of truth for all schematic connections.
   - Replace coordinate-coincidence netlist extraction in `CircuitScene.build_circuit()` with `build_circuit(self.wire_graph, [item.component for item in self.placed_items])`.

2. **Visual `WireItem` and `JunctionItem`**:
   - Update `WireItem` to store a `Wire` reference (`id`, `end_a`, `end_b`, `hints`).
   - Dynamically compute polyline points from `route(start_pos, end_pos, start_dir, end_dir, obstacles, previous=self.points)`.
   - Implement `JunctionItem` representing a `Junction` object:
     - Drawn as a solid dot only when degree $\ge 3$.
     - When degree drops to 2 (e.g. wire removed), automatically merges into a simple bend (stored as a hint in the combined wire) and the dot disappears.
     - `JunctionItem` is selectable, draggable, and deletable.
   - Draw `FreeEnd` attachments with a distinct hollow circular marker indicating an open-circuit wire end.

3. **Interactive Wiring & Terminal Snapping**:
   - When drawing a wire, hovering within `TERMINAL_SNAP_RADIUS_PX = 10 px` of a terminal snaps to that terminal; releasing binds that end as a `TerminalRef(comp_name, term_name)`.
   - Releasing a wire endpoint onto the interior of an existing wire automatically calls `graph.split_wire_at()`, creating a `Junction` and binding the new wire end to `JunctionRef(jid)`.
   - Releasing a wire in empty space creates a `FreeEnd(x, y)`.
   - Dragging a `FreeEnd` onto a terminal converts it to a `TerminalRef`.

4. **Component Dragging & Wire Tracking**:
   - When one or more components are moved, rotated (`R`), or flipped (`F`), attached wire endpoints follow their terminals automatically.
   - Update wire routes live using `route(...)` with orientation latching, preserving user-created waypoints where applicable.
   - Completely remove the legacy 40 px detour heuristic (`_route_wire_avoiding_body`) and coordinate-based endpoint mutation.

5. **`QUndoStack` Editing Commands with Immutability**:
   - Because `WireGraph` is immutable, each wire editing command (`AddWireCommand`, `RemoveWireCommand`, `MoveComponentCommand`, `SplitWireCommand`, `DeleteSelectionCommand`) simply stores the `old_graph` and `new_graph` instances, making undo and redo trivial and 100% reliable.

6. **Schema v2 Canvas Save / Load**:
   - Update `CircuitScene.to_document()` to output `CircuitDocumentV2` and `CircuitScene.load_document()` to accept `CircuitDocumentV2` (auto-migrating legacy v1 files on load).

### What Is Next (Milestone 3)
- **Step 4** (Canvas Integration of Explicit Wiring Model).
- **Step 3C** (Full properties dialog) follows Step 4. Step 3A is done, and Step 3B is cancelled by design decision.

## FLASH QUEUE

### Phase 1: Quick Wins
- [x] Task 1: Housekeeping (`.gitignore` `*.circ.bak`, `*.corrupt`; switch breakdown voltage contact arc hint; deterministic label collision avoidance) - *done (commit d67da36)*
- [x] Task 2: Recent Files (`File > Open Recent`, 8 items, injected path, atomic persistence, missing file handling, Clear List) - *done (commit 1710f48)*
- [x] Task 3: Example Circuits (`File > Examples` with 5 bundled circuits, package data, analytic tests) - *done (commit 7e29c9f)*
- [x] Task 4: Zoom to fit (`View > Zoom to fit` `Ctrl+0`, shortcuts `F5`/`F6`/`Shift+F5`, info panel, invariant canvas) - *done (commit fb23c5b)*
- [x] Task 5: Hover tooltips with live values (parts & terminals, stopped vs running, canvas invariance) - *done (commit 9c05b72)*
- [x] Task 6: Export schematic (`File > Export as PNG...` and `Export as SVG...`, white/theme options, tests) - *done (commit 7700a43)*
- [x] Task 7: About dialog (`Help > About`), feedback link (`Help > Report a problem / suggest a feature`), crash log exception hook - *done (commit f338754)*
- [x] Task 8: Bulb burn-out option (expose in dialog, dark+cross drawing, amber status note, open circuit) - *done (commit 47ee8e9)*
- [x] Task 9: Number format setting (`Edit > Preferences`, Auto/Dot/Comma display-only, persistence) - *done (commit 3bbc0fe)*
- [x] Task 10: CI (`.github/workflows/ci.yml`, README status badge) - *done*

### Phase 2: Step 3C Full Properties Dialog
- [ ] Step 3C: Full Properties Dialog (Identity, parameters, live readout, inductor helpers R_th, tau, 5tau, X_L, L calc) - *todo*

### Phase 3: Oscilloscope Plan
- [ ] Oscilloscope Plan (Step rate benchmarks, decoupled architecture, 3-chat split, risks; plan only, max 50 lines, await approval) - *todo*

### How to Run the App, Preview Tool, and Tests

```powershell
# Launch the CircuitSim desktop GUI
.\.venv\Scripts\python.exe -m circuitsim.main

# Open the IEC symbol & grid alignment preview tool
.\.venv\Scripts\python.exe -m circuitsim.tools.symbol_preview

# Run the headless Milestone 1 engine demo
.\.venv\Scripts\python.exe -m circuitsim.main --demo

# Run the full test suite (290 tests)
.\.venv\Scripts\python.exe -m pytest

# Run the fast test subset (excludes @pytest.mark.slow tests > 3s)
.\.venv\Scripts\python.exe -m pytest -m "not slow"
```

### Test Suite Speedup & Widget Audit (290 Tests)
- **Widget & Icon Scan**: Added `test_no_blank_buttons_or_icons_in_both_themes` in [`tests/test_problems_panel.py`](tests/test_problems_panel.py) scanning `MainWindow` buttons, `ProblemsPanel`, `make_severity_icon`, menu icons, `QuickValueDialog`, `PreferencesDialog`, and `ExportOptionsDialog` in both Dark and Light themes.
- **In-Process Shared Offscreen `QApplication` (`tests/conftest.py`)**: Replaced ~50 per-test Qt subprocess launches with one session-scoped `qapp` fixture (`QT_QPA_PLATFORM=offscreen`), per-test widget/timer/worker/`crash_handler` teardown, and a 30 s hang watchdog (`faulthandler.dump_traceback_later` + `threading.Timer`). Switched pixel-by-pixel `QImage.pixelColor()` loops to vectorized NumPy buffer checks.
- **Why Subprocesses Remain Where Used**:
  - Qt-free module import checks (`circuitsim.engine`, `circuitsim.components`, `circuitsim.canvas.connectivity`, `circuitsim.wiring`, `circuitsim.io.circ_serializer`, `circuitsim.engine.simulation_session`) run in clean subprocesses via `assert_module_has_no_qt_imports` so `PySide6` is guaranteed absent from `sys.modules`.
  - The `1024x768` toolbar `speed_combo` visibility check in `test_qt_step2c_ui_responsiveness_upright_labels_speed_dropdown_and_quick_edit` runs in a clean native-QPA subprocess because Qt's `offscreen` fallback font inflates the toolbar `sizeHint` width (~1748 px vs ~1042 px on Windows).
- **Runtime**: Full suite reduced from **150.86s** to **35.01s** (`290 passed`); fast subset `pytest -m "not slow"` runs in **23.22s** (`289 passed, 1 deselected`).

## RENAME INVENTORY (CircuitSim -> Ohmbench, Stage 1)

1. **User-visible text (Stage 1 -> `APP_NAME` = `"Ohmbench"`)**:
   - Window title (`src/circuitsim/app/main_window.py`): `"CircuitSim — Electrical Circuit Simulator"` and `f"{name}{star} — CircuitSim"`.
   - File dialogs (`src/circuitsim/app/main_window.py`): `"CircuitSim circuits (*.circ);;All files (*)"` / `"CircuitSim circuits (*.circ)"`.
   - About dialog & Help menu (`src/circuitsim/app/main_window.py`): `"&About CircuitSim"`, tooltip `"About CircuitSim"`, dialog body `f"<h3>CircuitSim v{version}</h3>"`, title `"About CircuitSim"`.
   - Crash handler & crash log (`src/circuitsim/app/crash_handler.py`): header `f"CircuitSim Crash Report: {now}"`, dialog body `f"CircuitSim encountered an unexpected error:..."`, dialog title `"CircuitSim Error"`.
   - Serializer user error messages (`src/circuitsim/io/circ_serializer.py`, `src/circuitsim/wiring/serializer.py`): `"This is not a CircuitSim file..."`, `"This file was made by a newer version of CircuitSim ... Please update CircuitSim to open it."`, `"...not a CircuitSim circuit."`.
   - CLI banner & parser (`src/circuitsim/main.py`): `"=== CircuitSim Milestone 1 Engine Demo ==="`, `argparse.ArgumentParser(description="CircuitSim Desktop Application")`.
   - Developer preview tool (`src/circuitsim/tools/symbol_preview.py`): argparse description and window title `"CircuitSim — IEC Symbol Preview (Dark & Light Themes)"`.
   - Documentation & guides (`README.md`, `PROGRESS.md`, `src/circuitsim/resources/symbols/iec/README.md`).

2. **App identity, OS storage & environment paths (Stage 1 updates + migration)**:
   - Qt identity (`QCoreApplication.setOrganizationName` / `setApplicationName`): currently unset, causing `QStandardPaths.AppDataLocation` to fall back to `AppData\Roaming\python`. Stage 1 sets both to `"Ohmbench"` and configures Windows `AppUserModelID`.
   - Data dir fallbacks (`src/circuitsim/app/settings.py`): `"CircuitSim"` / `.circuitsim` / `circuitsim` + one-time copy migration from legacy `AppData\Roaming\python` (`settings.json`, `crash.log`, recent-files data) into the new Ohmbench folder.
   - Env override: `CIRCUITSIM_DATA_DIR` (keep supported / Stage 2), `CIRCUITSIM_MAX_UI_GAP_MS` (test threshold env var, keep in Stage 1).

3. **File/folder names (Deferred to Stage 2 unless doc-only)**:
   - `src/circuitsim/` package directory, `pyproject.toml` `[project].name = "circuitsim"`, console script `circuitsim = "circuitsim.main:run_demo"`, MIME type `"application/x-circuitsim-component-id"`, `.circ` extension, `circuit_simulator_specification.txt`, and `circuitsim` module docstrings/imports — keep intact in Stage 1.

4. **Internal identifiers (Unchanged in Stage 1)**:
   - `circuitsim.*` imports, `CircuitScene`, `CircuitView`, `CircuitDocument`, `CircuitFileError`, `CIRCUITSIM_MAX_UI_GAP_MS`, test names/markers.

5. **Stored data (`.circ` format field & bundled examples)**:
   - `FORMAT_NAME = "circuitsim"` in `src/circuitsim/io/circ_serializer.py`, `src/circuitsim/wiring/serializer.py`, and all 7 files in `src/circuitsim/resources/examples/*.circ`, plus `.circ` test fixtures in `tests/test_circ_io.py`, `tests/test_led_examples.py`, `tests/test_wiring_v2_migration.py`. Stage 1 handles `"format"` decision and preserves seamless loading of legacy `"format": "circuitsim"` files.

6. **URLs (`EmerikBaltic/circuitsim` -> `EmerikBaltic/ohmbench`)**:
   - `REPO_URL` and `ISSUES_URL` in `src/circuitsim/app/crash_handler.py` (centralized into `src/circuitsim/app_info.py`), Help > Report issue action, About dialog links, and `README.md` CI badge URL (and clone note).

### `.circ` Format Identifier Decision (Phase 2.4)
- **Decision**: Keep writing `FORMAT_NAME = "circuitsim"` on save in both `io/circ_serializer.py` and `wiring/serializer.py`, while accepting `ACCEPTED_FORMAT_NAMES = ("circuitsim", "ohmbench")` on load.
- **Why**: Preserves full forward and backward compatibility with existing `.circ` project files, the 7 bundled example files (`resources/examples/*.circ`), earlier CircuitSim builds, and existing serializer contract tests (`test_file_declares_format_and_version`, `test_v2_save_and_load_round_trip`), while also accepting `"ohmbench"` cleanly if written in future builds.

### Legitimate Old-Name Strings Remaining in Tests (Phase 2.5)
- `tests/test_app_info.py`: Negative assertion `"CircuitSim" not in ...`, legacy `.circ` fixture `"format": "circuitsim"` (`test_circ_format_backward_and_forward_compatibility`).
- `tests/test_circ_io.py`, `tests/test_wiring_v2_migration.py`: Stored format token `"format": "circuitsim"` in schema round-trip / malformed header tests.
- `tests/test_simulation_session_and_step2c.py`: Benchmark env var `CIRCUITSIM_MAX_UI_GAP_MS`.
- `tests/test_deprecation_filter.py`: Test warning emitter `module="circuitsim.x"` matching `circuitsim.*` package filter.
- `tests/test_phase1_quick_wins.py`: Historical docstring reference (`'About CircuitSim'`) in `test_task7_about_feedback_and_crash_log`, and package path imports `circuitsim.*`.

### Stage 2 Rename Plan & Risks (Phase 2.6)
When ready to perform Stage 2 (internal/package rename after Stage 1 settles):
1. **Python Package Rename (`src/circuitsim/` -> `src/ohmbench/`)**:
   - Move `src/circuitsim` to `src/ohmbench` (`git mv`) and update all imports across `src/` and `tests/`.
   - Keep a temporary compatibility shim or clean deprecation period if external scripts import `circuitsim`? Better: clean atomic rename once `wire-redesign` branch is merged to avoid merge conflicts across branches.
2. **Packaging & CLI Entry Point (`pyproject.toml`)**:
   - Rename `[project].name` to `"ohmbench"`, `[tool.setuptools.package-data]` key to `"ohmbench"`, `[project.scripts]` entry point to `ohmbench = "ohmbench.main:run_demo"` (optionally keeping `circuitsim` alias temporarily), and pytest `filterwarnings = ["error::DeprecationWarning:ohmbench.*"]`.
   - Re-run `pip install -e .` in `.venv` so `importlib.metadata.version("ohmbench")` resolves cleanly and stale `circuitsim.egg-info` is removed.
3. **GitHub Repository & Remote URLs**:
   - Rename GitHub repo from `EmerikBaltic/circuitsim` to `EmerikBaltic/ohmbench` (GitHub automatically redirects old web/clone URLs) and update local `git remote set-url origin ...`.
   - Update clone instructions in `README.md`, MIME type (`application/x-ohmbench-component-id`), and env var alias (`CIRCUITSIM_DATA_DIR` -> `OHMBENCH_DATA_DIR` as primary, keeping old fallback).
4. **Risks & Mitigations**:
   - **Risk 1 — Merge Conflicts Across Active Feature Branches**: Renaming `src/circuitsim/` touches every Python file. *Mitigation*: Only perform Stage 2 when `wire-redesign` (and any other open branches) have been merged to `main`.
   - **Risk 2 — Stale `.venv` Editable Install / `.egg-info`**: Editable installs cache the old package name until `pip install -e .` is re-run, causing `ModuleNotFoundError` or stale package metadata. *Mitigation*: Include `pip uninstall -y circuitsim && pip install -e .[dev]` in the Stage 2 checklist and CI.
   - **Risk 3 — Drag-and-Drop / Saved `.circ` Compatibility**: Changing `MIME_COMPONENT_ID` mid-session isn't persistent, whereas `.circ` `"format"` IS persistent; keep `ACCEPTED_FORMAT_NAMES = ("circuitsim", "ohmbench")` indefinitely so user files never break.

## LOGIC GATES PLAN

1. **MODEL (Compare -> Recommend)**
   - *Options*: (a) **Last-step sampled inputs + RHS-only Thévenin/Norton output (`R_out` fixed e.g. `100 Ω`, `R_in = 10 MΩ`, `t_pd >= dt` delay buffer, linear ramp over `t_r/t_f`)**; (b) Switch-like `R_out` toggle forcing `lu_factor` rebuilds + Backward Euler bursts per edge; (c) LED-style same-step iteration over nonlinear curves (`solver.py` `is_conducting` loop).
   - *Comparison*: (b) destroys fixed-`dt` LU caching on clocks/ring oscillators; (c) fails or diverges on bistable latches (multiple DC roots) and astable loops. (a) decouples logic state from linear matrix structure so LU `(lu, piv)` stays cached cleanly every step (`lu_solve` only).
   - *Recommendation*: Use **(a)** with fixed `R_in = 10 MΩ` to `GND`, fixed `R_out = 100 Ω` + smoothed Norton current target `V_out(t) / R_out`, family presets (**5 V CMOS**: `V_IL=1.5, V_IH=3.5, V_OL=0.05, V_OH=4.95 V`; **5 V TTL**: `0.8, 2.0, 0.4, 3.4 V`; **3.3 V LVCMOS**: `0.8, 2.0, 0.1, 3.2 V`), default `t_pd = 10 ns` (clamped `>= dt`), and finite ramp `t_edge = max(t_r, 2*dt)` so Trapezoidal integration never sees step ringing.
   - *Loops & Latches*: Gates require causal `t_pd >= dt`. True-delay loops (SR NAND latch, odd-inverter ring oscillator) naturally settle or oscillate over time (`Q` initialized `L` at `t=0` to break symmetric latch ties deterministically); zero-delay combinational cycles (`t_pd = 0` topological cycle) are caught pre-run via Kahn/DFS topological check and reported.

2. **SUPPLY**
   - *v1*: Inputs and outputs reference circuit ground `"0"` implicitly (`Ground` still required), minimizing clutter on 2-input gates.
   - *Future explicit Vcc/GND*: Store `"supply_mode": "implicit"` (plus preset family) in `.circ` JSON; later add `"explicit"` toggle clamping `V_OH <= V_cc`, `V_OL >= V_gnd`, active only when `V_cc - V_gnd >= V_min` without breaking older saved files.

3. **DIAGNOSTICS (Problems Panel, plain 3-part What/Why/How style)**
   - **Floating input**: Unwired/high-Z input terminal (only internal `R_in` to GND) -> error naming gate & pin (`"Tie input to High/Low or wire to a driver"`).
   - **Undefined / mid-band voltage**: `V_IL < V_in < V_IH` -> amber warning reporting measured voltage and active family thresholds.
   - **Output contention**: >= 2 gate outputs tied to one net (especially driving opposite levels) -> warning/error reporting net, drivers, and contention current `I = (V_OH - V_OL) / (2 * R_out)`.
   - **Output shorted to supply/ground**: Gate output node clamped directly to ideal `DCVoltageSource` or `Ground` -> warning/error naming source & current.

4. **SYMBOLS & FOOTPRINTS (20 px grid, IEC 60617, no SVG `<text>`)**
   - *Footprints*: **NOT** (`80x40` / `4x2` grid): `in` at `[-2, 0]`, `out` at `[2, 0]`. **2-input gates (`AND, OR, NAND, NOR, XOR, XNOR`)**: `80x80` (`4x4` grid) so center & all pins sit strictly on 20 px intersections: `a` at `[-2, -1]`, `b` at `[-2, +1]`, `out` at `[2, 0]` (`40 px` pin spacing, body `x=[-20..20], y=[-30..30]`, 4 px negation circle on `NAND/NOR/XNOR/NOT`).
   - *Labels without SVG `<text>`*: Author qualifiers (`&`, `>=1`, `=1`, `1`) as vector `<path>` geometry in SVG (`stroke="currentColor"`) and mirror in `_paint_procedural_iec` (upright body rendering across rotations). Reuse `polarity_mark_positions` pattern + all-8-orientation pixel/terminal tests.

5. **TEACHING FEATURES**
   - **Logic Probe component**: Compact `1`-pin indicator displaying live `H` / `L` / `?` and node voltage.
   - **Net logic coloring** (optional canvas toggle): Green = High (`>= V_IH`), Blue/Gray = Low (`<= V_IL`), Amber = Undefined/Contention.
   - **Truth-Table Dialog**: Headless helper sweeps all `2^n` input combinations (stepping past `t_pd`) and shows measured voltages + `0/1/X` table beside ideal values.

6. **TESTS (Hand-worked & analytic)**
   - Analytic truth tables across all 7 gates and all 3 logic families.
   - Propagation delay & rise-time verification (`v_out` crosses 50% at `t = t_pd + t_edge/2`).
   - Odd-stage ring oscillator (`N = 3, 5`) matching analytic period `T = 2 * N * t_pd`.
   - Cross-coupled NAND SR latch verifying Set (`S=0,R=1`), Hold (`1,1` keeps Q), Reset (`1,0`), and recovery.
   - 5-gate XOR+AND half-adder (`S = A xor B`, `C = A & B`); contention current `I = (V_OH - V_OL) / (R_out1 + R_out2)`; zero-delay cycle error; Backward Euler / LU-reuse convergence checks.

7. **CHAT SPLIT, SPEED, RISKS & PREREQUISITES**
   - *Prerequisites*: **N-terminal support / 3-terminal pack** across `engine/netlist.py` (`validate_topology` currently inspects `nodes[0..1]` only and needs per-pin GND edges so open gate outputs work while floating inputs warn), `solver.py`, `connectivity`, and `ComponentItem`; **Wire Redesign Step 4** (explicit nets for contention checks); **Milestone 4 Oscilloscope** (needed for live timing waveforms/ring oscillators, while headless tests & truth tables work first).
   - *Splits*:
     - **Chat 1 (DONE — pure-Python engine & physics)**: Added `LogicGate` (`components/logic_gate.py`, 0 Qt imports) with 7 gates (`not/and/or/nand/nor/xor/xnor`), 3 family presets (`5V_CMOS`, `5V_TTL`, `3V3_LVCMOS` + `Custom`), fixed `R_in=10 MΩ`, `R_out=100 Ω`, `t_pd` delay queue, linear ramp smoothing, and implicit `"0"` ground; generalized `netlist.py`, `solver.py`, `errors.py` (`ZeroDelayLoopError`), and `simulation_session.py` for N-terminal gates, Gauss-Seidel DC/`t=0` equilibrium, fixed-LU transient stepping, and gate diagnostics (`floating_gate_input`, `indeterminate_logic_input`, `output_contention`, `output_shorted_to_supply`, `zero_delay_loop`); added 35 tests in `tests/test_logic_gates_engine.py`.
     - **Chat 2 (DONE — UI, symbols & `.circ` I/O)**: Added all 7 IEC 60617 JSON/vector-SVG symbols (`gate_not` `80x40`; `gate_and/or/nand/nor/xor/xnor` `80x80`, `<path>` qualifiers `1`/`&`/`≥1`/`=1`, no `<text>`), procedural counter-transformed upright qualifiers across all 8 `(rotation_deg, flipped_h)` orientations, `Logic Gates` category in `ComponentPaletteWidget`, `"U"` prefix & hover pin tags (`in`/`out`/`a`/`b`) in `ComponentItem`, dynamic `@property input_nodes/output_node` & `set_parameters_batch` in `LogicGate`, `QuickValueDialog` family preset/custom sync + `initial_state` combo + batch Undo/Redo (`commands.py`/`scene.py`/`main_window.py`/`simulation_session.py`), `.circ` v1/v2 `settings` & `include_extensions=True` in `wiring/` (`netlist.py`, `serializer.py`), per-version component bucketing in `solver.py` eliminating transient inner-loop ABC `isinstance` checks (~81 ms -> ~18 ms per 350-step ladder chunk), and 13 comprehensive tests in `tests/test_logic_gates_ui_and_io.py`.
     - **Chat 3 (NEXT — Logic Probe, net coloring & Truth-Table dialog)**: Interactive Logic Probe, live H/L net highlighting, and truth-table view; **Chat 4 (Small)**: Scope waveform demos (ring oscillator, SR latch).
   - *Solver Speed*: Fixed `R_out`/`R_in` preserves cached LU factorization (fast `lu_solve` only); ns-scale `t_pd` needs small `dt`, handled smoothly by decoupled sim speed.
   - *Risks*: Auditing `nodes[0..1]` assumptions across solver/netlist/canvas cleanly; symmetric SR latch metastability at `t=0` (mitigated via deterministic initial state); keeping `dt <= t_pd / 2` so delay & ramp stay well resolved.

---

## SCOPE CORE QUEUE

| Item | Description | Status | Commit |
| --- | --- | --- | --- |
| 1 | Orientation, step-rate benchmarks & design note | `[x]` | `b41cab4` |
| 2 | Decoupled simulation time, `SimulationSpeed.AS_FAST_AS_POSSIBLE` & step coupling (`timebase.py`, `simulation_session.py`) | `[x]` | `b60c169` |
| 3 | Qt-free sample sink, immutable `SampleBlock` & vectorized `RingBuffer` (`capture.py`, `simulation_session.py`) | `[x]` | `ba653aa` |
| 4 | Vectorized trigger engine (`trigger.py`) | `[x]` | `ec2c624` |
| 5 | 4-channel vertical scaling, 1x/10x probe & DC/AC/GND coupling (`channels.py`) | `[x]` | `c652bcf` |
| 6 | Automated waveform measurements & phase lag (`measurements.py`) | `[x]` | `bc4024c` |
| 7 | Math traces (`CH1-CH2`, `CH1+CH2`, `CH1*CH2`), periodic-Hann FFT, XY & cursors (`math_fft.py`) | `[x]` | `80ea01c` |
| 8 | `ScopeCore` state machine, Roll mode, Single & Auto Set (`core.py`, `__init__.py`) | `[x]` | `3fd9cde` |
| 9 | Comprehensive Qt-free analytical & circuit test suite (`tests/test_scope_core.py`) | `[x]` | `5a7a78e` |
| 10 | Offscreen Qt waveform debug tool & PNG exports (`tools/scope_demo.py`, `docs/design/scope_*.png`) | `[x]` | `849c7ea` |
| 11 | Performance benchmarks, sink overhead (<5%) & UI responsiveness check (`tests/test_scope_performance.py`) | `[x]` | `d2f10be` |
| 12 | Scope core API & Next-Chat UI requirements in `PROGRESS.md` | `[x]` | `2cfbf48` |
| 13 | Student `LEARNING_NOTES.md`, final fast suite check & wrap-up | `[x]` | pending |

## OSCILLOSCOPE ORIENTATION, BENCHMARKS & DESIGN NOTE

### Orientation & Spec Conflict Resolutions
- Branch: `main`; baseline `pytest -m "not slow"`: **344 passed, 1 deselected in 30.32s**.
- `docs/design/oscilloscope.md` does not exist; `circuit_simulator_specification.txt` Section 7 exists. Task rules override Section 7 on three points:
  1. **Probing**: Net-name probing via `SampleBlock.node_names` with ground `"0"` reference and `CH1 - CH2` differential math (UI chat adds the 4-terminal `Oscilloscope` canvas part).
  2. **Trigger**: Supports `rising`, `falling`, and `either` edges, explicit hysteresis (`0.05 div` default), simulated-time `holdoff`, and `DC`/`AC` trigger coupling.
  3. **Timebase Coupling**: `dt = min(dt_scope, dt_circuit)` with vectorized min/max peak-detect buckets whenever `dt_circuit < dt_scope`, plus Roll mode at `time/div >= 100 ms`.

### Baseline Transient Solver Step Rates (warmup 50 steps, 2,000 timed steps, `dt = 10 us`)
- (a) 5-part RC (`AC 5 V / 1 kHz`, `1 kΩ`, `10 kΩ`, `1 uF`): **30,593 steps/s** (`32.7 us/step`)
- (b) 30-resistor ladder + RC load: **19,534 steps/s** (`51.2 us/step`)
- (c) RL + switch (`12 V DC`, switch toggled at step 1000, `100 Ω`, `10 mH` / `1 Ω` DCR): **25,256 steps/s** (`39.6 us/step`)
- (d) Nonlinear LED + resistor (`AC 5 V / 1 kHz`, `150 Ω`, red LED): **29,494 steps/s** (`33.9 us/step`)

### Design Note (<= 30 lines)
- **Qt-free `src/circuitsim/scope/` package**: zero `PySide6` imports (`timebase.py`, `capture.py`, `trigger.py`, `channels.py`, `measurements.py`, `math_fft.py`, `core.py`), verified automatically in tests.
- **Decoupled clock & step coupling (`SimulationSession`)**: adds `SimulationSpeed.AS_FAST_AS_POSSIBLE`, injected `clock`, `SimulationFrame.overflow`, and `set_timebase()` using `compute_coupled_solver_dt(timebase, solver.choose_default_dt())` without resetting transient state or forcing repeated LU rebuilds.
- **Low-overhead capture sink**: `SimulationSession.step_chunk` keeps a zero-cost `self._scope_sink is None` check and otherwise fills one pre-allocated NumPy `(n_steps, n_nodes)` buffer into a frozen, picklable, read-only `SampleBlock` (< 5% solver overhead), stored in a fixed-capacity vectorized `RingBuffer`.
- **Signal pipeline & state machine**: `ScopeCore` ingests blocks, routes requested nets (with `1x`/`10x` scaling & integer-period-fallback AC coupling), detects sub-sample interpolated triggers with hysteresis & holdoff, emits immutable read-only `ScopeFrame` snapshots (switching cleanly to Roll mode at `>= 100 ms/div`), and computes IEEE-style 10%-90% measurements, periodic-Hann FFT, math traces, XY, and cursors in `< 2 ms` per 4ch 1000-pt frame.

### Scope Core Performance Benchmarks & Architecture Decision (Item 11)
- **4-ch 1000-pt `produce_frame()` (measurements + math + FFT)**: **2.172 ms** average (hard limit `< 8.0 ms`).
- **Active `ScopeCore` sink overhead (median of 5 paired 2,000-step trials vs no sink)**:
  - (a) 5-part RC: `30,614` -> `29,663 steps/s` (**+3.21%** overhead)
  - (b) 30-R ladder + RC: `19,469` -> `18,851 steps/s` (**+3.28%** overhead)
  - (c) RL + switch: `25,408` -> `24,856 steps/s` (**+2.22%** overhead)
  - (d) Nonlinear LED + R: `29,724` -> `28,958 steps/s` (**+2.64%** overhead)
- **30-R ladder `MainWindow` UI responsiveness with active scope sink**: `19` timer heartbeats in `300 ms`, `avg_gap = 16.53 ms`, `max_gap = 38.65 ms` (`< 50 ms` limit).
- **Decision**: Keep the single background `QThread` simulation worker (`SimulationWorkerController`); multiprocessing (`max_gap > 100 ms` threshold) is **not needed**.

---

## OSCILLOSCOPE CORE API & NEXT-CHAT UI REQUIREMENTS

### 1. Public Qt-Free Scope API (`src/circuitsim/scope/`)
- **`circuitsim.scope.ScopeCore`**: Stateful controller managing channels (`CH1`–`CH4`), `TimebaseConfig`, `TriggerConfig`, `MathConfig`, FFT, XY, cursors, and acquisition state (`ScopeRunState.RUN | STOP | SINGLE`).
  - Configuration: `assign_probe(channel, node_name, enabled=...)`, `set_channel(name, ...)`, `set_timebase(seconds_per_div, ...)`, `set_trigger(...)`, `set_math(enabled, op, ...)`, `set_fft(channel, window="hann")`, `set_xy(enabled, ...)`, `set_cursors(channel, t1_rel, t2_rel)`.
  - Controls: `run()`, `stop()`, `single()`, `auto_set()`, `clear()`, `bind_session(session)`, `unbind_session()`, `ingest_block(block)` (`on_sample_block`), `produce_frame() -> ScopeFrame`.
- **Immutable Transport Types (read-only NumPy arrays, safe before/after `pickle`)**:
  - `SampleBlock(t0, dt, node_names, data)` — 2D `(n_steps, n_nodes)` float64 snapshot (`data.flags.writeable == False`).
  - `ScopeFrame` — contains `run_state`, `status` (`TriggerStatus`), `status_message`, `sim_time`, `t_rel`, `t_abs`, `timebase`, `trigger`, `trigger_time_s`, `trigger_edge`, `traces: dict[str, ChannelFrameTrace]` (each carrying `volts`, `volts_min`, `volts_max`, `display_div_y`, `peak_detect`, `measurements: ChannelMeasurements`), `math_trace`, `fft: FFTResult`, `xy: XYTrace`, `phase_lag_ch1_ch2_deg`, `phase_reason`, `cursors: CursorReadout`, `overflow`, `missed_frames`.
- **Session Integration (`circuitsim.engine.simulation_session.SimulationSession`)**:
  - `SimulationSpeed.AS_FAST_AS_POSSIBLE` (`"max"`), `SimulationSession(..., clock=...)`, `SimulationFrame.overflow`.
  - `set_timebase(seconds_per_div, points_per_div=100) -> float`: couples solver `dt = min(dt_scope, dt_circuit)` without resetting transient state (LU rebuilds once on `dt` change).
  - `attach_scope_sink(sink)` / `detach_scope_sink(sink)` and `step_for_wall_elapsed(wall_dt, max_wall_budget_s=...)`.

### 2. Next-Chat UI & Canvas Part Requirements
- **`wire-redesign` prerequisite status**: Already merged (`git merge-base main wire-redesign == 50d4016`), so multi-terminal net extraction, explicit nets, and `.circ` persistence are live on `main` with no pending prerequisite branch merge.
- **4-Terminal `Oscilloscope` Schematic Part**:
  - New component type `"oscilloscope"` (`X` or `SCOPE` prefix) with 4 passive high-impedance probe terminals `("CH1", "CH2", "CH3", "CH4")` referenced to circuit `"0"`; high input impedance (`R_in >= 10 MOhm` or passive voltage observation) so attaching probes does not load ordinary circuits.
  - IEC Symbol (`src/circuitsim/resources/symbols/iec/oscilloscope.json` & `.svg`): **`120 x 80 px`** (`6 x 4` grid units, `#000000` strokes, `stroke-width="2"`, `fill="none"`, no `<text>`). IMPORTANT: Align all 4 left-edge pins strictly to the `20 px` grid relative to component center `(60, 40)` — e.g., `y in {0, 20, 60, 80}` (`x = 0`), avoiding half-grid `10, 30, 50, 70` offsets so wires stay 20 px grid-aligned across all 8 `(rotation, flip)` orientations.
  - Channel Net Mapping: Map canvas pins `"CH1"`..`"CH4"` connected net names dynamically into `ScopeCore.assign_probe(ch, net_name)` when circuit connectivity updates.
- **Oscilloscope Bench Window (Non-Modal, Never Shifts Canvas)**:
  - Modeless floating `QDialog` / detachable window (NOT a docked widget that resizes/shifts `CircuitScene` viewport); stays synchronized while `MainWindow` edits components/switches live.
  - Plot Widget (`QPainter`): dark `#0D1117` 10x8 graticule, fixed trace colors (`#FFD600`, `#00E5FF`, `#FF4081`, `#69F0AE`), peak-detect vertical envelope shading when `peak_detect=True`, trigger markers, interactive drag cursors, and FFT / XY / Time views.
  - Controls: CH1–CH4 enable/scale (`1 mV`–`1 kV`)/offset/`1x-10x`/DC-AC-GND/invert, Timebase (`10 ns`–`10 s`, Roll >= `100 ms`), Trigger (`source`, `edge`, `level`, `hysteresis`, `holdoff`, `DC/AC`, `Auto/Normal/Single`), `Run/Stop`, `Single`, `Auto Set`, Math (`CH1-CH2`, `CH1+CH2`, `CH1*CH2`), Measurements table & Export CSV/PNG.
- **Recommended Next-Chat Split**:
  1. **UI Chat 1 (Canvas Symbol + Connectivity + Modeless Bench Window Skeleton + Graticule Canvas)**: `oscilloscope` part & IEC SVG/JSON, palette category (`Instruments`), `.circ` save/load, live net binding into `ScopeCore`, modeless `OscilloscopeDialog`, dark 10x8 `QPainter` display + offscreen render tests in both themes.
  2. **UI Chat 2 (Controls, Cursors, Math/FFT/XY Views & CSV/PNG Export)**: Interactive controls wired to `ScopeCore` + `SimulationWorkerController.set_timebase`, drag cursors, measurements panel, FFT/XY mode toggle, Save CSV / Export PNG, and end-to-end Qt GUI workflow tests.

## KNOWN ISSUES / FIXED
- **Plain `pytest` collection error on `tests/test_app_info.py` (`ModuleNotFoundError: No module named 'tests'`)**: **Cause**: `[tool.pytest.ini_options]` in `pyproject.toml` only listed `pythonpath = ["src"]`, omitting the project root `.` when invoked as `pytest` rather than `python -m pytest`. **Fix**: Updated `pythonpath = [".", "src"]` in `pyproject.toml` so both `pytest` and `python -m pytest` resolve `tests.conftest` consistently.
- **Shared-CI timing variance in `tests/test_scope_performance.py`**: **Cause**: Strict local wall-clock thresholds (`< 8.0 ms`, `< 5.0%`, `>= 12`, `< 50.0 ms`) can flake on loaded CI runners due to noisy CPU scheduling. **Fix**: Relaxed thresholds only when `os.environ.get("CI")` is set (`MAX_FRAME_TIME_MS=25.0`, `MAX_SINK_OVERHEAD_PCT=15.0`, `MIN_UI_HEARTBEATS=6`, `MAX_UI_GAP_MS=150.0`), while keeping local thresholds strict and using repeated trials + median/GC control (`7` paired trials for sink overhead, `5` trials for 4ch `produce_frame()`, warm-up + clean signal/sink disconnect in UI loop test).
- **Intermittent Windows Qt teardown access violation (`tests/conftest.py:112`, `_qt_cleanup_and_watchdog`)**: **Cause**: Three interacting lifecycle issues: (1) `tests/conftest.py` set `widget._is_closing = True` before `widget.close()`, causing `MainWindow.closeEvent` to return early without disconnecting `undo_stack.cleanChanged`, `frame_ready`, or `circuit_scene` signals; (2) `CircuitScene` had `parent=None` in `MainWindow` plus Python reference cycles (closure lambda on `cleanChanged`, `selectionChanged` slot), so scenes with selected items survived until later cyclic GC during unrelated tests and fired `selectionChanged` during C++ destruction; (3) `SimulationWorkerController._worker` (`_SessionThreadWorker`) had no C++ parent and was moved to `self._thread` at `__init__` without returning to the main thread / `deleteLater()` before `self._thread` destruction. **Fix**: Separated `_close_cleaned_up` from `_is_closing` in `MainWindow.closeEvent` and passed `parent=self` to `CircuitScene`, replaced the `cleanChanged` lambda with `_on_undo_clean_changed`, added `CircuitScene.shutdown_scene()`, deferred `_worker.moveToThread(self._thread)` until `_ensure_thread_running()` with `cleanup_for_shutdown()` on `SimulationWorkerController.shutdown()`, and updated `_qt_cleanup_and_watchdog` in `tests/conftest.py` to shut down standalone `CircuitScene`s and run `gc.collect()` + `DeferredDelete` flush between tests.
- **`ScopeCore.auto_set` exact-value specification (A4c)**: `ScopeCore.auto_set` docstring (`src/circuitsim/scope/core.py:355`) states only `"Automatically choose vertical scale, offset, timebase, and trigger level."` and `PROGRESS.md` does not document its `Vpp / 5.0` divisor, midpoint offset rounding, or `1-2-5` snapping tie rules; exact-value Auto Set test in `tests/test_scope_core_exact.py` was skipped and reported as instructed.



