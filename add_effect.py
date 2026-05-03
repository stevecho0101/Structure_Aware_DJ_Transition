import sys
from pydub import AudioSegment


FADE_MS = 4000


def transition(song1_path, song1_end_sec, song2_path, song2_start_sec,
               output_path="output.mp3"):

    song1_end_ms   = int((song1_end_sec+2) * 1000)
    song2_start_ms = int(song2_start_sec * 1000)

    song1 = AudioSegment.from_mp3(song1_path)
    song2 = AudioSegment.from_mp3(song2_path)

    # Song1: body plays normally, last 4 seconds fade to silence
    fade_start  = max(0, song1_end_ms - FADE_MS)
    song1_body  = song1[:fade_start]
    song1_tail  = song1[fade_start:song1_end_ms].fade_out(FADE_MS)
    song1_out   = song1_body + song1_tail

    # Song2 fades in over 2 seconds
    song2_out = song2[song2_start_ms:].fade_in(1500)

    result = song1_out.append(song2_out, crossfade=2000)
    result.export(output_path, format="mp3")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python add_effect.py <song1.mp3> <song1_end_sec> <song2.mp3> <song2_start_sec> [output.mp3]")
        sys.exit(1)

    song1_path      = sys.argv[1]
    song1_end_sec   = float(sys.argv[2])
    song2_path      = sys.argv[3]
    song2_start_sec = float(sys.argv[4])
    output_path     = sys.argv[5] if len(sys.argv) > 5 else "output.mp3"

    transition(song1_path, song1_end_sec, song2_path, song2_start_sec, output_path)
