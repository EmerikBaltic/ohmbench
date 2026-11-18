"""Simulation engine: MNA matrix builder, solvers, session controller, and topology diagnostics."""

from circuitsim.engine.errors import (
    CircuitError,
    CircuitWarning,
    FloatingNodeError,
    NoGroundError,
    SingularMatrixError,
    VoltageSourceLoopError,
    ZeroDelayLoopError,
)
from circuitsim.engine.netlist import Circuit
from circuitsim.engine.simulation_session import (
    ComponentTelemetry,
    ProblemDiagnostic,
    SimulationBackendProtocol,
    SimulationFrame,
    SimulationSession,
    SimulationSpeed,
    SimulationState,
    TransportControlsViewState,
    compute_transport_controls_state,
)
from circuitsim.engine.solver import (
    IntegrationMethod,
    MNASolver,
    SolveResult,
    TransientResult,
)
from circuitsim.engine.stress import (
    StressLevel,
    StressReport,
    evaluate_component_stress,
)
from circuitsim.engine.overlay import (
    compute_current_dot_velocity,
    format_power_probe_text,
    format_watts_si,
    map_voltage_to_rgb,
)
from circuitsim.engine.ac_analysis import (
    ACAnalysisResult,
    ACFrequencySample,
    run_ac_sweep,
)
from circuitsim.engine.fft_analysis import (
    FFTSpectrumResult,
    compute_fft_spectrum,
)
from circuitsim.engine.sweep import (
    MonteCarloResult,
    MonteCarloTolerance,
    ParameterSweepResult,
    run_monte_carlo,
    run_parameter_sweep,
)

__all__ = [
    "ACAnalysisResult",
    "ACFrequencySample",
    "Circuit",
    "CircuitError",
    "CircuitWarning",
    "ComponentTelemetry",
    "FFTSpectrumResult",
    "FloatingNodeError",
    "IntegrationMethod",
    "MNASolver",
    "MonteCarloResult",
    "MonteCarloTolerance",
    "NoGroundError",
    "ParameterSweepResult",
    "ProblemDiagnostic",
    "SimulationBackendProtocol",
    "SimulationFrame",
    "SimulationSession",
    "SimulationSpeed",
    "SimulationState",
    "SingularMatrixError",
    "SolveResult",
    "StressLevel",
    "StressReport",
    "TransientResult",
    "TransportControlsViewState",
    "VoltageSourceLoopError",
    "ZeroDelayLoopError",
    "compute_fft_spectrum",
    "compute_current_dot_velocity",
    "compute_transport_controls_state",
    "evaluate_component_stress",
    "format_power_probe_text",
    "format_watts_si",
    "map_voltage_to_rgb",
    "run_ac_sweep",
    "run_monte_carlo",
    "run_parameter_sweep",
]
