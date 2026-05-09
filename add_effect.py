import sys
import argparse
import numpy as np
from pydub import AudioSegment
from pydub.effects import speedup
from pydub.scipy_effects import high_pass_filter, low_pass_filter


# CONSTANTS ======================================================================
FADE_MS       = 4000   # length of the outgoing fade
CROSSFADE_MS  = 2000   # overlap between song1 tail and song2 head
FADE_IN_MS    = 1500   # song2 fade-in duration


# HELPER FUNCTIONS ================================================================
def _load(path: str) -> AudioSegment:
    return AudioSegment.from_mp3(path)

# splits a song into a body and a "tail" closer to the transition
def _split(song: AudioSegment, end_ms: int, fade_ms: int):
    fade_start = max(0, end_ms - fade_ms)
    body = song[:fade_start]
    tail = song[fade_start:end_ms]
    return body, tail

# sweeps the bass in/out by applying a high-pass filter
def _apply_eq_sweep(segment: AudioSegment, direction: str) -> AudioSegment:
    chunk_ms   = 100                                # split song into chunks, apply a varying filter to each chunk
    n_chunks   = max(1, len(segment) // chunk_ms)
    cutoffs    = np.linspace(20, 300, n_chunks)     # Cutoff sweeps from 20 Hz (full bass) → 300 Hz (bass cut) or vice versa
    if direction == "in":
        cutoffs = cutoffs[::-1]   # start high-cut, end with full bass

    result = AudioSegment.empty()
    for i in range(n_chunks):
        start = i * chunk_ms
        end   = start + chunk_ms
        chunk = segment[start:end]
        if len(chunk) == 0:
            continue
        cutoff = float(cutoffs[min(i, len(cutoffs) - 1)])
        if cutoff > 30:   # only filter when cutoff is meaningfully above DC
            chunk = high_pass_filter(chunk, cutoff)
        result += chunk

    # Append any leftover ms that didn't fit into a full chunk
    remainder = segment[n_chunks * chunk_ms:]
    if len(remainder) > 0:
        result += remainder

    return result

# sweep a low-pass filter over chunks
def _apply_lpf_sweep(segment: AudioSegment, direction: str) -> AudioSegment:
    chunk_ms = 100
    n_chunks = max(1, len(segment) // chunk_ms)
    cutoffs  = np.linspace(18000, 400, n_chunks)    # Cutoff in Hz: sweep between param1 (higher, open) and param2 (lower, muffled)
    if direction == "in":
        cutoffs = cutoffs[::-1]

    result = AudioSegment.empty()
    for i in range(n_chunks):
        start = i * chunk_ms
        end   = start + chunk_ms
        chunk = segment[start:end]
        if len(chunk) == 0:
            continue
        cutoff = float(cutoffs[min(i, len(cutoffs) - 1)])
        chunk  = low_pass_filter(chunk, cutoff)
        result += chunk

    remainder = segment[n_chunks * chunk_ms:]
    if len(remainder) > 0:
        result += remainder

    return result

# adjust song's playback speed
def _beatmatch(song: AudioSegment, src_bpm: float, tgt_bpm: float) -> AudioSegment:
    ratio = tgt_bpm / src_bpm   # >1 -> speed up | <1 -> slow down
    if abs(ratio - 1.0) < 0.01:
        return song   # already close enough, skip processing

    # speed up, using pydub
    if ratio >= 1.0:
        return speedup(song, playback_speed=ratio, chunk_size=150, crossfade=25)
    # slow down requires re-sampling manually
    else:
        samples    = np.array(song.get_array_of_samples(), dtype=np.float32)
        n_channels = song.channels
        if n_channels == 2:
            samples = samples.reshape(-1, 2)

        old_len = samples.shape[0]
        new_len = int(round(old_len / ratio))
        x_old   = np.linspace(0, old_len - 1, old_len)
        x_new   = np.linspace(0, old_len - 1, new_len)

        if n_channels == 2:
            left  = np.interp(x_new, x_old, samples[:, 0]).astype(np.int16)
            right = np.interp(x_new, x_old, samples[:, 1]).astype(np.int16)
            stretched = np.column_stack([left, right]).flatten()
        else:
            stretched = np.interp(x_new, x_old, samples).astype(np.int16)

        return song._spawn(stretched.tobytes())


# Transition modes ===========================================================================

# standard, simple crossfade
def crossfade(song1: AudioSegment, song1_end_ms: int,
              song2: AudioSegment, song2_start_ms: int) -> AudioSegment:
    body, tail = _split(song1, song1_end_ms, FADE_MS)
    song1_out  = body + tail.fade_out(FADE_MS)
    song2_out  = song2[song2_start_ms:].fade_in(FADE_IN_MS)
    return song1_out.append(song2_out, crossfade=CROSSFADE_MS)

# bass swap
# song1's tail has bass gradually cut
# song2's tail has its bass gradually increased
def eq_sweep(song1: AudioSegment, song1_end_ms: int,
             song2: AudioSegment, song2_start_ms: int) -> AudioSegment:
    body, tail = _split(song1, song1_end_ms, FADE_MS)

    # Sweep bass out on song1's tail, then crossfade volume
    tail_eq   = _apply_eq_sweep(tail, direction="out")
    song1_out = body + tail_eq.fade_out(FADE_MS)

    # Song2: take a 4-second intro section to apply the bass-in sweep, then continue
    song2_clip    = song2[song2_start_ms:]
    sweep_in_len  = min(FADE_MS, len(song2_clip))
    song2_head    = _apply_eq_sweep(song2_clip[:sweep_in_len], direction="in")
    song2_out     = song2_head.fade_in(FADE_IN_MS) + song2_clip[sweep_in_len:]

    return song1_out.append(song2_out, crossfade=CROSSFADE_MS)

# match song2 tail's BPM to song1's BPM
def beatmatch_transition(song1: AudioSegment, song1_end_ms: int,
                         song2: AudioSegment, song2_start_ms: int,
                         bpm1: float, bpm2: float) -> AudioSegment:
    
    body, tail = _split(song1, song1_end_ms, FADE_MS)
    song1_out  = body + tail.fade_out(FADE_MS)

    song2_clip = song2[song2_start_ms:]

    # Only beatmatch the first few seconds of song2 (the overlap window)
    match_len     = min(FADE_MS * 2, len(song2_clip))
    song2_head    = _beatmatch(song2_clip[:match_len], src_bpm=bpm2, tgt_bpm=bpm1)
    song2_out     = song2_head.fade_in(FADE_IN_MS) + song2_clip[match_len:]

    print(f"  BPM: {bpm1:.1f} → {bpm2:.1f}  (ratio {bpm1/bpm2:.3f}x applied to song2 intro)")
    return song1_out.append(song2_out, crossfade=CROSSFADE_MS)

# low-pass sweep transition
def lpf_sweep(song1: AudioSegment, song1_end_ms: int,
              song2: AudioSegment, song2_start_ms: int) -> AudioSegment:

    body, tail = _split(song1, song1_end_ms, FADE_MS)

    tail_lpf  = _apply_lpf_sweep(tail, direction="out")
    song1_out = body + tail_lpf.fade_out(FADE_MS)

    song2_clip   = song2[song2_start_ms:]
    sweep_in_len = min(FADE_MS, len(song2_clip))
    song2_head   = _apply_lpf_sweep(song2_clip[:sweep_in_len], direction="in")
    song2_out    = song2_head.fade_in(FADE_IN_MS) + song2_clip[sweep_in_len:]

    return song1_out.append(song2_out, crossfade=CROSSFADE_MS)


# Public entry point ===================================================================

def transition(song1_path: str, song1_end_sec: float,
               song2_path: str, song2_start_sec: float,
               output_path: str = "output.mp3",
               mode: str = "crossfade",
               bpm1: float = 120.0,
               bpm2: float = 120.0):
    """
    Apply a DJ transition effect between two songs and save the result.

    Parameters
    ----------
    song1_path      : path to the outgoing song MP3
    song1_end_sec   : second in song1 where the transition OUT begins
    song2_path      : path to the incoming song MP3
    song2_start_sec : second in song2 where the transition IN begins
    output_path     : where to save the mixed MP3
    mode            : 'crossfade' | 'eq_sweep' | 'beatmatch' | 'lpf_sweep'
    bpm1            : BPM of song1 (only used when mode='beatmatch')
    bpm2            : BPM of song2 (only used when mode='beatmatch')
    """
    song1_end_ms   = int((song1_end_sec + 2) * 1000)
    song2_start_ms = int(song2_start_sec * 1000)

    print(f"Loading songs...")
    song1 = _load(song1_path)
    song2 = _load(song2_path)
    print(f"  Song1: {len(song1)/1000:.1f}s  |  Song2: {len(song2)/1000:.1f}s")
    print(f"  Mode : {mode}")

    if mode == "crossfade":
        result = crossfade(song1, song1_end_ms, song2, song2_start_ms)

    elif mode == "eq_sweep":
        result = eq_sweep(song1, song1_end_ms, song2, song2_start_ms)

    elif mode == "beatmatch":
        result = beatmatch_transition(song1, song1_end_ms, song2, song2_start_ms,
                                      bpm1=bpm1, bpm2=bpm2)

    elif mode == "lpf_sweep":
        result = lpf_sweep(song1, song1_end_ms, song2, song2_start_ms)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose: crossfade | eq_sweep | beatmatch | lpf_sweep")

    result.export(output_path, format="mp3")
    print(f"Saved to {output_path}")


# main ==============================================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DJ transition effects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  crossfade   simple fade out / fade in (default)
  eq_sweep    bass cut on exit, bass restore on entry
  beatmatch   tempo-align song2 to song1 BPM (requires --bpm1 and --bpm2)
  lpf_sweep   low-pass filter sweeps down then up (EDM style)

Examples:
  python add_effect.py s1.mp3 120 s2.mp3 8 out.mp3 crossfade
  python add_effect.py s1.mp3 120 s2.mp3 8 out.mp3 eq_sweep
  python add_effect.py s1.mp3 120 s2.mp3 8 out.mp3 beatmatch --bpm1 128 --bpm2 124
  python add_effect.py s1.mp3 120 s2.mp3 8 out.mp3 lpf_sweep
        """
    )
    parser.add_argument("song1_path")
    parser.add_argument("song1_end_sec",   type=float)
    parser.add_argument("song2_path")
    parser.add_argument("song2_start_sec", type=float)
    parser.add_argument("output_path",     nargs="?", default="output.mp3")
    parser.add_argument("mode",            nargs="?", default="crossfade",
                        choices=["crossfade", "eq_sweep", "beatmatch", "lpf_sweep"])
    parser.add_argument("--bpm1", type=float, default=120.0,
                        help="BPM of song1 (beatmatch mode only)")
    parser.add_argument("--bpm2", type=float, default=120.0,
                        help="BPM of song2 (beatmatch mode only)")

    args = parser.parse_args()
    transition(
        args.song1_path, args.song1_end_sec,
        args.song2_path, args.song2_start_sec,
        output_path=args.output_path,
        mode=args.mode,
        bpm1=args.bpm1,
        bpm2=args.bpm2,
    )
