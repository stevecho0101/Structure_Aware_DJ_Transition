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

def run_agent(current_song_name, current_labels, candidate_songs, feedback_history=[]):
    current_summary = summarize_song(current_labels)
    candidates_info = {name: summarize_song(labels) for name, labels in candidate_songs.items()}

    # build feedback context from previous attempts
    feedback_str = ""
    if feedback_history:
        feedback_str = "\n\nPREVIOUS ATTEMPTS AND FEEDBACK:\n"
        for i, f in enumerate(feedback_history):
            feedback_str += f"Attempt {i+1}: recommended {f['song']} at {f['transition_out']}s → {f['transition_in']}s. User feedback: {f['feedback']}\n"
        feedback_str += "\nUse this feedback to pick a different song or different transition points this time.\n"

    prompt = f"""You are an expert DJ assistant. A DJ is currently playing a song and needs to know which song to transition to next.

CURRENT SONG ({current_song_name}):
{json.dumps(current_summary)}

CANDIDATE SONGS TO TRANSITION TO:
{json.dumps(candidates_info, indent=2)}
{feedback_str}
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
  "transition_in": {{"time_sec": <float>, "section": "<string>", "reason": "<string>"}},
  "overall_reasoning": "<2 sentences>"
}}"""

    client = anthropic.Anthropic(api_key="your_key_here") #insert API Key
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

# feedback loop
feedback_history = []
while True:
    result = run_agent(current_song, current_labels, candidates, feedback_history)
    print(json.dumps(result, indent=2))

    next_song = result["recommended_next_song"]
    song1_out = result["transition_out"]["time_sec"]
    song2_in  = result["transition_in"]["time_sec"]

    print(f"\nTransitioning: {current_song} → {next_song}")
    print(f"Cut at {song1_out}s → enter at {song2_in}s")

    # ask for feedback
    print("\nRate this transition:")
    print("  [enter] accept and mix")
    print("  [1-5]   rate it (1=terrible, 5=perfect) and try again")
    print("  [q]     quit")
    rating = input("Your rating: ").strip().lower()

    if rating == "" or rating == "5":
        # accepted — do the mix
        transition(
            os.path.join(SAMPLES_DIR, current_song), song1_out,
            os.path.join(SAMPLES_DIR, next_song), song2_in,
            "output_mix.mp3"
        )
        print("Mix saved to output_mix.mp3")
        break
    elif rating == "q":
        print("Cancelled.")
        break
    else:
        # get optional written feedback
        comment = input("Any specific feedback? (or press enter to skip): ").strip()
        feedback_text = f"rating {rating}/5" + (f" — {comment}" if comment else "")
        feedback_history.append({
            "song": next_song,
            "transition_out": song1_out,
            "transition_in": song2_in,
            "feedback": feedback_text
        })
        print("\nTrying again with your feedback...\n")
