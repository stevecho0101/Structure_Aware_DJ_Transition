import os
import sys
import json
import glob

from flask import Flask, jsonify, send_from_directory, request
from train_cnn_lstm import predict
import anthropic

app = Flask(__name__, static_folder="static")

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "song_samples")
SECTION_NAMES = {1: "intro", 2: "verse", 3: "chorus", 4: "outro"}

# cache CNN labels so we don't rerun on every analyze
labels_cache = {}

# store feedback per session
feedback_store = {}


def summarize_song(labels):
    sections = []
    prev = None
    for i, lbl in enumerate(labels):
        if lbl != prev:
            sections.append({"section": SECTION_NAMES[lbl], "start_sec": i})
            prev = lbl
    return sections


def run_agent(current_song_name, current_labels, candidate_songs, played_songs=[], prior_feedback=[]):
    current_summary = summarize_song(current_labels)
    candidates_info = {name: summarize_song(labels) for name, labels in candidate_songs.items()}
    played_str = ", ".join(played_songs) if played_songs else "none"

    feedback_str = ""
    if prior_feedback:
        feedback_str = "\n\nPREVIOUS ATTEMPTS AND FEEDBACK:\n"
        for i, f in enumerate(prior_feedback):
            feedback_str += f"Attempt {i+1}: recommended {f['song']} at {f['transition_out']}s. Rating: {f['rating']}/5. Comment: {f.get('comment') or 'none'}\n"
        feedback_str += "\nUse this feedback to pick a different song or different transition points.\n"

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
- Do NOT recommend already played songs: {played_str}
- If previous attempts had low ratings, try a completely different approach

Respond ONLY with this JSON format, no extra text:
{{
  "recommended_next_song": "<filename>",
  "reason_for_recommendation": "<one sentence>",
  "transition_out": {{"time_sec": <float>, "section": "<string>", "reason": "<string>"}},
  "transition_in": {{"time_sec": <float>, "section": "<string>", "reason": "<string>"}},
  "overall_reasoning": "<2 sentences>"
}}"""

    client = anthropic.Anthropic(api_key="your_key_here") #Insert API Key
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(raw)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/songs")
def get_songs():
    songs = [os.path.basename(p) for p in glob.glob(os.path.join(SAMPLES_DIR, "*.mp3"))]
    return jsonify({"songs": songs})


@app.route("/song_samples/<filename>")
def serve_song(filename):
    return send_from_directory(SAMPLES_DIR, filename)


@app.route("/analyze_with_feedback", methods=["POST"])
def analyze_with_feedback():
    data = request.json
    current_song = data["current_song"]
    played = data.get("played_songs", [])
    session_id = data.get("session_id", "default")
    prior_feedback = feedback_store.get(session_id, [])

    all_songs = {}
    for mp3 in glob.glob(os.path.join(SAMPLES_DIR, "*.mp3")):
        name = os.path.basename(mp3)
        if name not in labels_cache:
            print(f"  CNN: {name}...")
            labels_cache[name] = predict(mp3).tolist()
        all_songs[name] = labels_cache[name]

    current_labels = all_songs[current_song]
    candidates = {k: v for k, v in all_songs.items() if k != current_song and k not in played}
    if not candidates:
        candidates = {k: v for k, v in all_songs.items() if k != current_song}

    result = run_agent(current_song, current_labels, candidates, played, prior_feedback)
    result["current_structure"] = summarize_song(current_labels)
    next_song = result["recommended_next_song"]
    if next_song in all_songs:
        result["next_structure"] = summarize_song(all_songs[next_song])

    return jsonify(result)


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    session_id = data.get("session_id", "default")

    if session_id not in feedback_store:
        feedback_store[session_id] = []

    feedback_store[session_id].append({
        "song": data["song"],
        "transition_out": data["transition_out"],
        "transition_in": data["transition_in"],
        "rating": data["rating"],
        "comment": data.get("comment", "")
    })

    return jsonify({"status": "ok", "attempts": len(feedback_store[session_id])})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
