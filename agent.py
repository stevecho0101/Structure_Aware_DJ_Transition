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

def run_agent(current_song_name, current_labels, candidate_songs):
    current_summary = summarize_song(current_labels)
    candidates_info = {name: summarize_song(labels) for name, labels in candidate_songs.items()}

    prompt = f"""You are an expert DJ assistant. A DJ is currently playing a song and needs to know which song to transition to next.

CURRENT SONG ({current_song_name}):
{json.dumps(current_summary)}

CANDIDATE SONGS TO TRANSITION TO:
{json.dumps(candidates_info, indent=2)}

Rules:
- Pick the ONE best candidate song to transition to next
- Choose a transition OUT point from the current song (end of chorus or verse)
- Choose a transition IN point for the next song (start of verse or chorus)
- Prioritize energy continuity, no big energy drops

Respond ONLY with this JSON format, no extra text:
{{
  "recommended_next_song": "<filename>",
  "reason_for_recommendation": "<one sentence>",
  "transition_out": {{"time_sec": <float>, "section": "<string>", "reason": "<string>"}},
  "transition_in": {{"time_sec": <float>, "section": "<string>", "reason": "<string>"}},
  "overall_reasoning": "<2 sentences>"
}}"""

    client = anthropic.Anthropic(api_key="your_key_here")
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(raw)


SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "song_samples")

print("Analyzing songs...")
all_songs = {}
for mp3 in glob.glob(os.path.join(SAMPLES_DIR, "*.mp3")):
    name = os.path.basename(mp3)
    print(f"  {name}...")
    all_songs[name] = predict(mp3).tolist()

print(f"\nLoaded {len(all_songs)} songs\n")

current_song = list(all_songs.keys())[0]
current_labels = all_songs[current_song]
print(f"Now playing: {current_song}")

candidates = {k: v for k, v in all_songs.items() if k != current_song}

result = run_agent(current_song, current_labels, candidates)
print(json.dumps(result, indent=2))

next_song = result["recommended_next_song"]
song1_out = result["transition_out"]["time_sec"]
song2_in  = result["transition_in"]["time_sec"]

print(f"\nTransitioning: {current_song} → {next_song}")
print(f"Cut at {song1_out}s → enter at {song2_in}s")

transition(
    os.path.join(SAMPLES_DIR, current_song), song1_out,
    os.path.join(SAMPLES_DIR, next_song), song2_in,
    "output_mix.mp3"
)
