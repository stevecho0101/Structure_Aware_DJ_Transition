'''
- given a user-selected song, will pick the best choice from the user's pool of songs and use Claude to decide the best transition type and time
- from train_cnn_lstm.py, the agent recieves the predicted section labels which are stored in a json to avoid unnecessary repeated inference
- Claude takes the json and returns a json that includes which song to play, exit and entry timestamps for the two songs, the transition type, and the reasoning
- Agent has a recent song cooldown, takes feedback from the user, and can be forced to choose certain transition types
- ran by agent.py
'''
import os
import anthropic
import json
import sys
import glob

sys.path.insert(0, os.path.dirname(__file__))
from train_cnn_lstm import predict

SECTION_NAMES = {1: "intro", 2: "verse", 3: "chorus", 4: "outro"}
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "song_samples")
CACHE_FILE = os.path.join(os.path.dirname(__file__), "labels_cache.json")

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        labels_cache = json.load(f)
else:
    labels_cache = {}


def get_labels(mp3_path):
    name = os.path.basename(mp3_path)
    if name not in labels_cache:
        print(f"  CNN: {name}...")
        labels_cache[name] = predict(mp3_path).tolist()
        with open(CACHE_FILE, "w") as f:
            json.dump(labels_cache, f)
    return labels_cache[name]


def load_all_songs():
    all_songs = {}
    for mp3 in glob.glob(os.path.join(SAMPLES_DIR, "*.mp3")):
        name = os.path.basename(mp3)
        all_songs[name] = get_labels(mp3)
    return all_songs


def summarize_song(labels):
    sections = []
    prev = None
    for i, lbl in enumerate(labels):
        if lbl != prev:
            sections.append({"section": SECTION_NAMES[lbl], "start_sec": i})
            prev = lbl
    return sections


def run_agent(current_song_name, current_labels, all_songs,
              recent_songs=[], feedback_history=[], forced_mode=None):
    current_summary = summarize_song(current_labels)
    # pass all songs as candidates — agent decides
    candidates_info = {
        name: summarize_song(labels)
        for name, labels in all_songs.items()
        if name != current_song_name
    }

    # last 3 songs to avoid repeating
    avoid_str = ", ".join(recent_songs[-3:]) if recent_songs else "none"

    feedback_str = ""
    if feedback_history:
        feedback_str = "\n\nPREVIOUS TRANSITION FEEDBACK:\n"
        for i, f in enumerate(feedback_history):
            feedback_str += f"Transition {i+1}: played {f['song']}. Rating: {f.get('rating','?')}/5. Comment: {f.get('comment') or 'none'}\n"
        feedback_str += "\nUse this feedback to improve the next recommendation.\n"

    if forced_mode:
        effect_instruction = (
            f"\n\nIMPORTANT: The user has forced the transition effect to '{forced_mode}'. "
            f"You MUST set transition_effect to '{forced_mode}'. Still explain why in effect_reason."
        )
    else:
        effect_instruction = """

TRANSITION EFFECT SELECTION — pick the best one:
- 'lpf_sweep'  : high-energy EDM/electronic; dramatic build-to-drop feel
- 'eq_sweep'   : hip-hop, funk, R&B where or other genres where bass line matters
- 'beatmatch'  : when both songs have steady rhythm and BPM continuity matters, maintaining the flow of the set
- 'crossfade'  : safe default for mismatched genres or low-energy sections"""

    prompt = f"""You are an expert DJ assistant. A DJ is currently playing a song and needs to know which song to transition to next.

CURRENT SONG ({current_song_name}):
{json.dumps(current_summary)}

ALL CANDIDATE SONGS:
{json.dumps(candidates_info, indent=2)}
{feedback_str}{effect_instruction}

Rules:
- Pick the ONE best candidate song to transition to next
- Choose a transition OUT point from the current song (end of chorus or verse)
- Choose a transition IN point for the next song (start of verse or chorus)
- Prioritize energy continuity, no big energy drops
- AVOID these recently played songs (cooldown): {avoid_str}
- Prefer songs from a DIFFERENT artist than the current song and recent songs
- If previous transitions had low ratings, try a completely different approach

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

    client = anthropic.Anthropic(api_key="your_key_here")  # insert API key
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(raw)


# ── Run this first to cache all CNN labels ────────────────────────────────────

if __name__ == "__main__":
    print("Running CNN on all songs and caching labels...")
    all_songs = load_all_songs()
    print(f"\nDone! Cached {len(all_songs)} songs to labels_cache.json")
    print("Now run: python app.py")
