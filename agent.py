import os
import anthropic
import json
import sys
import glob

sys.path.insert(0, os.path.dirname(__file__))
from train_cnn_lstm import predict
from add_effect import transition

SECTION_NAMES = {1: "intro", 2: "verse", 3: "chorus", 4: "outro"}


def summarize_song(labels):
    sections = []
    prev = None
    for i, lbl in enumerate(labels):
        if lbl != prev:
            sections.append({"section": SECTION_NAMES[lbl], "start_sec": i})
            prev = lbl
    return sections


def run_agent(current_song_name, current_labels, candidate_songs,
              feedback_history=[], forced_mode=None):
    current_summary  = summarize_song(current_labels)
    candidates_info  = {name: summarize_song(labels) for name, labels in candidate_songs.items()}

    # Build feedback context from previous attempts
    feedback_str = ""
    if feedback_history:
        feedback_str = "\n\nPREVIOUS ATTEMPTS AND FEEDBACK:\n"
        for i, f in enumerate(feedback_history):
            feedback_str += (
                f"Attempt {i+1}: recommended {f['song']} at "
                f"{f['transition_out']}s → {f['transition_in']}s. "
                f"User feedback: {f['feedback']}\n"
            )
        feedback_str += "\nUse this feedback to pick a different song or different transition points this time.\n"

    # Effect selection instructions
    if forced_mode:
        effect_instruction = (
            f"\n\nIMPORTANT: The user has forced the transition effect to '{forced_mode}'. "
            f"You MUST set transition_effect to '{forced_mode}' in your response regardless of your preference. "
            f"Still explain why in effect_reason."
        )
    else:
        effect_instruction = """

TRANSITION EFFECT SELECTION — pick the best one for these two songs:
- 'lpf_sweep'  : high-energy EDM/electronic transitions; dramatic build-to-drop feel
- 'eq_sweep'   : hip-hop, funk, or R&B where the bass line is important
- 'beatmatch'  : when both songs have a steady groove and BPM continuity matters
- 'crossfade'  : safe default for mismatched genres, ballads, or low-energy sections"""

    prompt = f"""You are an expert DJ assistant. A DJ is currently playing a song and needs to know which song to transition to next.

CURRENT SONG ({current_song_name}):
{json.dumps(current_summary)}

CANDIDATE SONGS TO TRANSITION TO:
{json.dumps(candidates_info, indent=2)}
{feedback_str}{effect_instruction}

Rules:
- Pick the ONE best candidate song to transition to next
- Choose a transition OUT point from the current song (end of chorus or verse)
- Choose a transition IN point for the next song (start of verse or chorus)
- Prioritize energy continuity, no big energy drops
- If previous attempts had bad feedback, try a completely different song or different section types

Respond ONLY with this JSON format, no extra text:
{{
  "recommended_next_song": "<filename>",
  "reason_for_recommendation": "<one sentence>",
  "transition_out": {{"time_sec": <float>, "section": "<string>", "reason": "<string>"}},
  "transition_in":  {{"time_sec": <float>, "section": "<string>", "reason": "<string>"}},
  "transition_effect": "<crossfade|eq_sweep|beatmatch|lpf_sweep>",
  "effect_reason": "<one sentence explaining why this effect fits>",
  "overall_reasoning": "<2 sentences>"
}}"""

    client = anthropic.Anthropic(api_key="your_key_here")  # insert API Key
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(raw)


# ── Main ──────────────────────────────────────────────────────────────────────

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "song_samples")

print("Analyzing songs...")
all_songs = {}
for mp3 in glob.glob(os.path.join(SAMPLES_DIR, "*.mp3")):
    name = os.path.basename(mp3)
    print(f"  {name}...")
    all_songs[name] = predict(mp3).tolist()

print(f"\nLoaded {len(all_songs)} songs\n")

current_song   = list(all_songs.keys())[0]
current_labels = all_songs[current_song]
print(f"Now playing: {current_song}")

candidates = {k: v for k, v in all_songs.items() if k != current_song}

# ── Optional effect override before agent runs ────────────────────────────────
print("\nForce a transition effect? (or press enter to let Claude decide)")
print("  [1] crossfade  [2] eq_sweep  [3] lpf_sweep  [4] beatmatch  [enter] auto")
forced_choice = input("Choice: ").strip()
mode_map      = {"1": "crossfade", "2": "eq_sweep", "3": "lpf_sweep", "4": "beatmatch"}
forced_mode   = mode_map.get(forced_choice, None)

if forced_mode:
    print(f"  Effect locked to: {forced_mode}")
else:
    print("  Claude will recommend an effect.")

# ── Feedback loop ─────────────────────────────────────────────────────────────
feedback_history = []
while True:
    result = run_agent(current_song, current_labels, candidates,
                       feedback_history, forced_mode=forced_mode)
    print("\n" + json.dumps(result, indent=2))

    next_song  = result["recommended_next_song"]
    song1_out  = result["transition_out"]["time_sec"]
    song2_in   = result["transition_in"]["time_sec"]
    mode       = forced_mode if forced_mode else result["transition_effect"]

    print(f"\nTransitioning : {current_song} → {next_song}")
    print(f"Cut at        : {song1_out}s → enter at {song2_in}s")
    print(f"Effect        : {mode}  ({result['effect_reason']})")

    # Collect BPMs upfront if beatmatch is selected
    bpm1, bpm2 = 120.0, 120.0
    if mode == "beatmatch":
        try:
            bpm1 = float(input("BPM of current song: ").strip())
            bpm2 = float(input("BPM of next song: ").strip())
        except ValueError:
            print("  Invalid BPM, defaulting to 120/120")

    # Ask for feedback
    print("\nRate this transition:")
    print("  [enter] accept and mix")
    print("  [1-5]   rate it (1=terrible, 5=perfect) and try again")
    print("  [q]     quit")
    rating = input("Your rating: ").strip().lower()

    if rating == "" or rating == "5":
        # Accepted — do the mix
        transition(
            os.path.join(SAMPLES_DIR, current_song), song1_out,
            os.path.join(SAMPLES_DIR, next_song),    song2_in,
            "output_mix.mp3",
            mode=mode,
            bpm1=bpm1,
            bpm2=bpm2,
        )
        print("Mix saved to output_mix.mp3")
        break
    elif rating == "q":
        print("Cancelled.")
        break
    else:
        comment      = input("Any specific feedback? (or press enter to skip): ").strip()
        feedback_text = f"rating {rating}/5" + (f" — {comment}" if comment else "")
        feedback_history.append({
            "song":           next_song,
            "transition_out": song1_out,
            "transition_in":  song2_in,
            "feedback":       feedback_text,
        })
        print("\nTrying again with your feedback...\n")
