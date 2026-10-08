import logging
from pathlib import Path
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

# Add a standard, plain-text handler
bp_logger = logging.getLogger("basic-pitch")
# Remove its fancy emoji-printing handlers
bp_logger.handlers = []
bp_logger.addHandler(logging.StreamHandler())
bp_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

def transcribe_to_midi_with_bp(audio_file: str, 
                        final_output_path: str,
                        overwrite: bool = False, 
                        min_freq: float | None = None, 
                        max_freq: float | None = None,
                        melodia_trick: bool = False,
                        onset_threshold: float = 0.5,
                        frame_threshold: float = 0.3,
                        min_note_len_ms: float = 127.70,
                        ) -> str:
    """
    Transcribes an audio file to MIDI using Basic-Pitch.

    Args:
        audio_file: Path to the input audio file.
        overwrite: If True, will overwrite existing MIDI files.
        min_freq: The minimum frequency (in Hz) to detect. Notes whose
            fundamental is below this are removed from the model's activation
            matrices before decoding, so they can never be emitted. Use it to
            bound the output to an instrument's range (a standard-tuned guitar
            cannot sound below MIDI 40 = 82.41 Hz).
        max_freq: The maximum frequency (in Hz) to detect — same mechanism.
        melodia_trick: Basic-Pitch's residual-energy pass. After the
            onset-driven notes are decoded it keeps scavenging the remaining
            frame energy into extra notes, which on a polyphonic mix yields a
            lot of low-amplitude harmonics (a note 12/19 semitones above an
            already-ringing note). Passed through to ``predict`` (see NOTE) and
            also drives ``multiple_pitch_bends``.
        onset_threshold / frame_threshold: Basic-Pitch decode thresholds.
        min_note_len_ms: Minimum note length in **milliseconds**. Notes whose
            above-threshold span is not longer than this are dropped. This is
            the main knob against the sub-frame blips a dense mix produces.

    Returns:
        The file path to the generated .mid file.

    NOTE (fixed): ``minimum_note_length`` is documented by Basic-Pitch as
    milliseconds and its default is 127.70 — the value used to be passed as
    ``min_note_len_ms / 1000``, i.e. 0.1277 *ms*, which converts to 0 frames and
    silently disabled the minimum-length filter entirely (~45% of the emitted
    notes on a 15 s solo slice were shorter than 128 ms, ~21% were a single
    11.6 ms frame). ``melodia_trick`` was likewise never forwarded to
    ``predict``, so it was pinned to that function's default (True) no matter
    what the caller asked for.
    """
    logger.info("Transcribing audio to MIDI with Basic-Pitch...")
    logger.info("This may take a moment...")

    output_path = Path(final_output_path)
    if not output_path.exists() or overwrite:
        logger.info(f"Predicting MIDI for {audio_file}...")

        model_output, midi_data, note_events = predict(
            audio_path=audio_file,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
            minimum_note_length=min_note_len_ms,
            melodia_trick=melodia_trick,
            multiple_pitch_bends=melodia_trick
        )
        
        # --- Use our own reliable file-saving function ---
        if overwrite and output_path.exists():
            logger.info(f"Overwrite enabled. Removing existing file: {output_path}")
            output_path.unlink()
            
        # The midi_data object from basic-pitch can be written directly to a file
        midi_data.write(str(output_path))

        logger.info(f"--- MIDI transcription successfully generated: {output_path} ---")
    else:
        logger.info(f"--- MIDI file already exists, skipping transcription: {output_path} ---")

    return str(output_path)