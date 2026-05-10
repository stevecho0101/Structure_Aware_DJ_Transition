import os
import sys
import glob
from flask import Flask, jsonify, send_from_directory, request

sys.path.insert(0, os.path.dirname(__file__))
from agent import run_agent, summarize_song, load_all_songs, SAMPLES_DIR
from add_effect import transition

app = Flask(__name__, static_folder="static")

feedback_store = {}   # session_id -> list of feedback
played_store = {}     # session_id -> list of all played songs


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


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    current_song = data["current_song"]
    session_id = data.get("session_id", "default")
    forced_mode = data.get("forced_mode", None)
    prior_feedback = feedback_store.get(session_id, [])
    played = played_store.get(session_id, [])

    all_songs = load_all_songs()
    current_labels = all_songs[current_song]

    # hard exclude played songs from candidates
    candidates = {k: v for k, v in all_songs.items() if k != current_song and k not in played}

    # if all songs have been played, reset and allow all
    if not candidates:
        played_store[session_id] = [current_song]
        candidates = {k: v for k, v in all_songs.items() if k != current_song}

    result = run_agent(current_song, current_labels, candidates,
                       played, prior_feedback, forced_mode)

    result["current_structure"] = summarize_song(current_labels)
    next_song = result["recommended_next_song"]
    if next_song in all_songs:
        result["next_structure"] = summarize_song(all_songs[next_song])

    return jsonify(result)


@app.route("/played", methods=["POST"])
def mark_played():
    data = request.json
    session_id = data.get("session_id", "default")
    song = data["song"]
    if session_id not in played_store:
        played_store[session_id] = []
    if song not in played_store[session_id]:
        played_store[session_id].append(song)
    return jsonify({"status": "ok", "played_count": len(played_store[session_id])})


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    session_id = data.get("session_id", "default")
    if session_id not in feedback_store:
        feedback_store[session_id] = []
    feedback_store[session_id].append({
        "song": data["song"],
        "rating": data["rating"],
        "comment": data.get("comment", "")
    })
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
