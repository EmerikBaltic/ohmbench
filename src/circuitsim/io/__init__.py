"""JSON project save/load, SPICE netlist import/export, WAV I/O, and image export."""

from circuitsim.io.kicad_export import export_to_kicad_sch
from circuitsim.io.share_codec import (
    SHARE_URL_PREFIX,
    SHARE_URL_SCHEME,
    ShareUrlError,
    decode_circuit_share_url,
    encode_circuit_share_url,
)
from circuitsim.io.spice_import import (
    EncryptedModelError,
    SpiceElement,
    SpiceImportError,
    SpiceLibrary,
    SpiceModelDef,
    SpiceSubcktDef,
    detect_encrypted_spice,
    instantiate_diode_model,
    instantiate_subckt_elements,
    map_subckt_pins,
    parse_spice_file,
    parse_spice_library,
    parse_spice_number,
)
from circuitsim.io.wav_signal import (
    WavSignalData,
    export_waveform_to_wav,
    load_wav_samples,
    save_wav_samples,
)

__all__ = [
    "EncryptedModelError",
    "SHARE_URL_PREFIX",
    "SHARE_URL_SCHEME",
    "ShareUrlError",
    "SpiceElement",
    "SpiceImportError",
    "SpiceLibrary",
    "SpiceModelDef",
    "SpiceSubcktDef",
    "WavSignalData",
    "decode_circuit_share_url",
    "detect_encrypted_spice",
    "encode_circuit_share_url",
    "export_to_kicad_sch",
    "export_waveform_to_wav",
    "instantiate_diode_model",
    "instantiate_subckt_elements",
    "load_wav_samples",
    "map_subckt_pins",
    "parse_spice_file",
    "parse_spice_library",
    "parse_spice_number",
    "save_wav_samples",
]
