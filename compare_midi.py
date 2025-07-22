import mido
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import logging

note_name_maps = {
    'PLASTIC GUITAR': { 127: "Trill Marker", 126: "Tremolo Marker", 116: "Overdrive", 103: "Solo Marker", 102: "EXPERT Force HOPO Off", 101: "EXPERT Force HOPO On", 100: "EXPERT Orange", 99: "EXPERT Blue", 98: "EXPERT Yellow", 97: "EXPERT Red", 96: "EXPERT Green", 90: "HARD Force HOPO Off", 89: "HARD Force HOPO On", 88: "HARD Orange", 87: "HARD Blue", 86: "HARD Yellow", 85: "HARD Red", 84: "HARD Green", 76: "MEDIUM Orange", 75: "MEDIUM Blue", 74: "MEDIUM Yellow", 73: "MEDIUM Red", 72: "MEDIUM Green", 64: "EASY Orange", 63: "EASY Blue", 62: "EASY Yellow", 61: "EASY Red", 60: "EASY Green" },
    'PLASTIC BASS': { 127: "Trill Marker", 126: "Tremolo Marker", 116: "Overdrive", 103: "Solo Marker", 102: "EXPERT Force HOPO Off", 101: "EXPERT Force HOPO On", 100: "EXPERT Orange", 99: "EXPERT Blue", 98: "EXPERT Yellow", 97: "EXPERT Red", 96: "EXPERT Green", 90: "HARD Force HOPO Off", 89: "HARD Force HOPO On", 88: "HARD Orange", 87: "HARD Blue", 86: "HARD Yellow", 85: "HARD Red", 84: "HARD Green", 76: "MEDIUM Orange", 75: "MEDIUM Blue", 74: "MEDIUM Yellow", 73: "MEDIUM Red", 72: "MEDIUM Green", 64: "EASY Orange", 63: "EASY Blue", 62: "EASY Yellow", 61: "EASY Red", 60: "EASY Green" },
    'PLASTIC DRUMS': { 127: "Cymbal Swells", 126: "Drum Roll", 124: "Drum Fill", 123: "Drum Fill", 122: "Drum Fill", 121: "Drum Fill", 120: "Drum Fill (use all 5)", 116: "Overdrive", 112: "Tom Marker 4", 111: "Tom Marker 3", 110: "Tom Marker 2", 103: "Solo Marker", 100: "EXPERT Green", 99: "EXPERT Blue", 98: "EXPERT Yellow", 97: "EXPERT Red", 96: "EXPERT Kick", 88: "HARD Green", 87: "HARD Blue", 86: "HARD Yellow", 85: "HARD Red", 84: "HARD Kick", 76: "MEDIUM Green", 75: "MEDIUM Blue", 74: "MEDIUM Yellow", 73: "MEDIUM Red", 72: "MEDIUM Kick", 64: "EASY Green", 63: "EASY Blue", 62: "EASY Yellow", 61: "EASY Red", 60: "EASY Kick" },
    'BEAT': { 13: "Measure", 12: "Beat" },
    'PRO VOCALS': { 116: "Overdrive", 105: "Phrase Marker", 84: "Pitched Vocals 48", 83: "Pitched Vocals 47", 82: "Pitched Vocals 46", 81: "Pitched Vocals 45", 80: "Pitched Vocals 44", 79: "Pitched Vocals 43", 78: "Pitched Vocals 42", 77: "Pitched Vocals 41", 76: "Pitched Vocals 40", 75: "Pitched Vocals 39", 74: "Pitched Vocals 38", 73: "Pitched Vocals 37", 72: "Pitched Vocals 36", 71: "Pitched Vocals 35", 70: "Pitched Vocals 34", 69: "Pitched Vocals 33", 68: "Pitched Vocals 32", 67: "Pitched Vocals 31", 66: "Pitched Vocals 30", 65: "Pitched Vocals 29", 64: "Pitched Vocals 28", 63: "Pitched Vocals 27", 62: "Pitched Vocals 26", 61: "Pitched Vocals 25", 60: "Pitched Vocals 24", 59: "Pitched Vocals 23", 58: "Pitched Vocals 22", 57: "Pitched Vocals 21", 56: "Pitched Vocals 20", 55: "Pitched Vocals 19", 54: "Pitched Vocals 18", 53: "Pitched Vocals 17", 52: "Pitched Vocals 16", 51: "Pitched Vocals 15", 50: "Pitched Vocals 14", 49: "Pitched Vocals 13", 48: "Pitched Vocals 12", 47: "Pitched Vocals 11", 46: "Pitched Vocals 10", 45: "Pitched Vocals 9", 44: "Pitched Vocals 8", 43: "Pitched Vocals 7", 42: "Pitched Vocals 6", 41: "Pitched Vocals 5", 40: "Pitched Vocals 4", 39: "Pitched Vocals 3", 38: "Pitched Vocals 2", 37: "Pitched Vocals 1" },
    'PART VOCALS': { 116: "Overdrive", 106: "EXPERT 5 Lift", 105: "EXPERT 4 Lift", 104: "EXPERT 3 Lift", 103: "EXPERT 2 Lift", 102: "EXPERT 1 Lift", 100: "EXPERT 5", 99: "EXPERT 4", 98: "EXPERT 3", 97: "EXPERT 2", 96: "EXPERT 1", 93: "HARD 4 Lift", 92: "HARD 3 Lift", 91: "HARD 2 Lift", 90: "HARD 1 Lift", 87: "HARD 4", 86: "HARD 3", 85: "HARD 2", 84: "HARD 1", 81: "MEDIUM 4 Lift", 80: "MEDIUM 3 Lift", 79: "MEDIUM 2 Lift", 78: "MEDIUM 1 Lift", 75: "MEDIUM 4", 74: "MEDIUM 3", 73: "MEDIUM 2", 72: "MEDIUM 1", 69: "EASY 4 Lift", 68: "EASY 3 Lift", 67: "EASY 2 Lift", 66: "EASY 1 Lift", 63: "EASY 4", 62: "EASY 3", 61: "EASY 2", 60: "EASY 1" },
    'PART BASS': { 116: "Overdrive", 106: "EXPERT 5 Lift", 105: "EXPERT 4 Lift", 104: "EXPERT 3 Lift", 103: "EXPERT 2 Lift", 102: "EXPERT 1 Lift", 100: "EXPERT 5", 99: "EXPERT 4", 98: "EXPERT 3", 97: "EXPERT 2", 96: "EXPERT 1", 93: "HARD 4 Lift", 92: "HARD 3 Lift", 91: "HARD 2 Lift", 90: "HARD 1 Lift", 87: "HARD 4", 86: "HARD 3", 85: "HARD 2", 84: "HARD 1", 81: "MEDIUM 4 Lift", 80: "MEDIUM 3 Lift", 79: "MEDIUM 2 Lift", 78: "MEDIUM 1 Lift", 75: "MEDIUM 4", 74: "MEDIUM 3", 73: "MEDIUM 2", 72: "MEDIUM 1", 69: "EASY 4 Lift", 68: "EASY 3 Lift", 67: "EASY 2 Lift", 66: "EASY 1 Lift", 63: "EASY 4", 62: "EASY 3", 61: "EASY 2", 60: "EASY 1", 59: "Fret 12", 57: "Fret 11", 56: "Fret 10", 55: "Fret 9", 53: "Fret 8", 52: "Fret 7", 50: "Fret 6", 49: "Fret 5", 47: "Fret 4", 45: "Fret 3", 43: "Fret 2", 40: "Fret 1" },
    'PART DRUMS': { 116: "Overdrive", 106: "EXPERT 5 Lift", 105: "EXPERT 4 Lift", 104: "EXPERT 3 Lift", 103: "EXPERT 2 Lift", 102: "EXPERT 1 Lift", 100: "EXPERT 5", 99: "EXPERT 4", 98: "EXPERT 3", 97: "EXPERT 2", 96: "EXPERT 1", 93: "HARD 4 Lift", 92: "HARD 3 Lift", 91: "HARD 2 Lift", 90: "HARD 1 Lift", 87: "HARD 4", 86: "HARD 3", 85: "HARD 2", 84: "HARD 1", 81: "MEDIUM 4 Lift", 80: "MEDIUM 3 Lift", 79: "MEDIUM 2 Lift", 78: "MEDIUM 1 Lift", 75: "MEDIUM 4", 74: "MEDIUM 3", 73: "MEDIUM 2", 72: "MEDIUM 1", 69: "EASY 4 Lift", 68: "EASY 3 Lift", 67: "EASY 2 Lift", 66: "EASY 1 Lift", 63: "EASY 4", 62: "EASY 3", 61: "EASY 2", 60: "EASY 1", 51: "Floor Tom hit w/RH", 50: "Floor Tom hit w/LH", 49: "Tom2 hit w/RH", 48: "Tom2 hit w/LH", 47: "Tom1 hit w/RH", 46: "Tom1 hit w/LH", 45: "A soft hit on crash 2 with the left hand", 44: "A hit on crash 2 with the left hand", 43: "A ride hit with the left hand", 42: "Ride Cym hit w/RH", 41: "Crash2 Choke (hit w/RH, choke w/LH)", 40: "Crash1 Choke (hit w/RH, choke w/LH)", 39: "Crash2 (near Ride Cym) soft hit w/RH", 38: "Crash2 hard hit w/RH", 37: "Crash1 (near Hi-Hat) soft hit w/RH", 36: "Crash1 hard hit w/RH", 35: "Crash1 soft hit w/LH", 34: "Crash1 hard hit w/LH", 32: "Percussion w/ RH", 31: "Hi-Hat hit w/RH", 30: "Hi-Hat hit w/LH", 29: "A soft snare hit with the right hand", 28: "A soft snare hit with the left hand", 27: "Snare hit w/RH", 26: "Snare hit w/LH", 25: "Hi-Hat pedal up (hat open) w/LF", 24: "Kick hit w/RF" },
    'PART GUITAR': { 116: "Overdrive", 106: "EXPERT 5 Lift", 105: "EXPERT 4 Lift", 104: "EXPERT 3 Lift", 103: "EXPERT 2 Lift", 102: "EXPERT 1 Lift", 100: "EXPERT 5", 99: "EXPERT 4", 98: "EXPERT 3", 97: "EXPERT 2", 96: "EXPERT 1", 93: "HARD 4 Lift", 92: "HARD 3 Lift", 91: "HARD 2 Lift", 90: "HARD 1 Lift", 87: "HARD 4", 86: "HARD 3", 85: "HARD 2", 84: "HARD 1", 81: "MEDIUM 4 Lift", 80: "MEDIUM 3 Lift", 79: "MEDIUM 2 Lift", 78: "MEDIUM 1 Lift", 75: "MEDIUM 4", 74: "MEDIUM 3", 73: "MEDIUM 2", 72: "MEDIUM 1", 69: "EASY 4 Lift", 68: "EASY 3 Lift", 67: "EASY 2 Lift", 66: "EASY 1 Lift", 63: "EASY 4", 62: "EASY 3", 61: "EASY 2", 60: "EASY 1", 59: "Fret 12", 57: "Fret 11", 56: "Fret 10", 55: "Fret 9", 53: "Fret 8", 52: "Fret 7", 50: "Fret 6", 49: "Fret 5", 47: "Fret 4", 45: "Fret 3", 43: "Fret 2", 40: "Fret 1" },
}

TIME_WINDOW = 10
TIME_THRESHOLD = 10

def load_midi_tracks(file_path):
    try:
        mid = mido.MidiFile(file_path)
    except Exception as e:
        logging.error(f"Error loading MIDI file {file_path}: {e}")
        return None
    tracks = {track.name: track for track in mid.tracks if hasattr(track, 'name')}
    return tracks

def extract_note_events(track, note_range):
    note_events = defaultdict(list)
    current_time = 0
    for msg in track:
        current_time += msg.time
        if msg.type in {'note_on', 'note_off'} and msg.note in note_range:
            note_type = 'note_off' if msg.type == 'note_off' or msg.velocity == 0 else 'note_on'
            note_events[current_time].append((msg.note, note_type, msg.velocity))
    return note_events

def extract_text_events(track):
    text_events = []
    current_time = 0
    for msg in track:
        current_time += msg.time
        if msg.type in ('text', 'lyrics'):
            text_events.append((current_time, msg.text))
    return text_events

def group_events_by_time_window(events, time_window):
    grouped_events = defaultdict(list)
    sorted_times = sorted(events.keys())
    
    for time in sorted_times:
        notes = events[time]
        found_group = False
        for group_time in grouped_events:
            if abs(time - group_time) <= time_window:
                grouped_events[group_time].extend(notes)
                found_group = True
                break
        if not found_group:
            grouped_events[time].extend(notes)
    return grouped_events

def compare_tracks(track1_events, track2_events, time_window, time_threshold):
    differences = []
    grouped1 = group_events_by_time_window(track1_events, time_window)
    grouped2 = group_events_by_time_window(track2_events, time_window)
    
    all_times = sorted(set(grouped1.keys()) | set(grouped2.keys()))
    
    for time in all_times:
        events1 = set(tuple(e) for e in grouped1.get(time, []))
        events2 = set(tuple(e) for e in grouped2.get(time, []))
        
        if events1 != events2:
            is_timing_shift = False
            for other_time in all_times:
                if abs(time - other_time) <= time_threshold and time != other_time:
                    other_events1 = set(tuple(e) for e in grouped1.get(other_time, []))
                    other_events2 = set(tuple(e) for e in grouped2.get(other_time, []))
                    if events1 == other_events2 or events2 == other_events1:
                        is_timing_shift = True
                        break
            if not is_timing_shift:
                added = events2 - events1
                removed = events1 - events2
                if added or removed:
                    differences.append((time, list(removed), list(added)))
    return differences

def compare_text_events(text_events1, text_events2):
    diffs = []
    events_map1 = {time: text for time, text in text_events1}
    events_map2 = {time: text for time, text in text_events2}
    all_times = sorted(set(events_map1.keys()) | set(events_map2.keys()))
    
    for time in all_times:
        text1 = events_map1.get(time, "[No Event]")
        text2 = events_map2.get(time, "[No Event]")
        if text1 != text2:
            diffs.append((time, text1, text2))
    return diffs

def visualize_midi_changes(differences, text_differences, note_name_map, track_name, output_folder, session_id):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_labels = {}
    y_pos_counter = 0

    all_notes_in_diff = set()
    for _, removed, added in differences:
        all_notes_in_diff.update(note[0] for note in removed)
        all_notes_in_diff.update(note[0] for note in added)

    for note in sorted(list(all_notes_in_diff)):
        y_labels[note] = y_pos_counter
        y_pos_counter += 1

    for time, removed, added in differences:
        for note_val, _, _ in removed:
            ax.scatter(time, y_labels[note_val], c='red', marker='s', s=100, edgecolors='black', label='Removed' if 'Removed' not in plt.gca().get_legend_handles_labels()[1] else "")
        for note_val, _, _ in added:
            ax.scatter(time, y_labels[note_val], c='green', marker='s', s=100, edgecolors='black', label='Added' if 'Added' not in plt.gca().get_legend_handles_labels()[1] else "")

    if text_differences:
        if 'text' not in y_labels:
            y_labels['text'] = y_pos_counter
        for time, _, _ in text_differences:
            ax.scatter(time, y_labels['text'], c='blue', marker='^', s=100, label='Text Change' if 'Text Change' not in plt.gca().get_legend_handles_labels()[1] else "")

    ax.set_xlabel('Time (MIDI Ticks)')
    ax.set_ylabel('Notes / Events')
    ax.set_title(f'MIDI Changes for {track_name}')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    sorted_y_labels = sorted(y_labels.items(), key=lambda item: item[1])
    ax.set_yticks([pos for _, pos in sorted_y_labels])
    ax.set_yticklabels([note_name_map.get(note, f'Note {note}') if isinstance(note, int) else "Text Events" for note, _ in sorted_y_labels])
    
    ax.legend()
    plt.tight_layout()
    
    image_path = os.path.join(output_folder, f"{track_name.replace(' ', '_')}_changes_{session_id}.png")
    plt.savefig(image_path)
    plt.close()
    return image_path

def run_comparison(midi_file1_path, midi_file2_path, session_id, output_folder='out', format="json"):
    os.makedirs(output_folder, exist_ok=True)

    tracks1 = load_midi_tracks(midi_file1_path)
    if not tracks1: return []
    tracks2 = load_midi_tracks(midi_file2_path)
    if not tracks2: return []
        
    generated_results = []
    
    if format == "ini":
        tracks_to_compare = [
            'PART DRUMS', 'PART BASS', 'PART GUITAR', 
            'PAD VOCALS', 'PAD BASS', 'PAD DRUMS', 'PAD GUITAR', 
            'PRO VOCALS', 'BEAT', 'EVENTS', 'SECTION'
        ]
    else: 
        tracks_to_compare = [
            'PART BASS', 'PART GUITAR', 'PART DRUMS', 'PART VOCALS', "PRO VOCALS", 
            "PLASTIC GUITAR", "PLASTIC DRUMS", "PLASTIC BASS", 'BEAT', 'EVENTS', 'SECTION'
        ]

    all_present_track_names = sorted(list(set(tracks1.keys()) | set(tracks2.keys())))

    tracks_to_actually_compare = [name for name in all_present_track_names if name in tracks_to_compare]

    for track_name in tracks_to_actually_compare:
        track1 = tracks1.get(track_name)
        track2 = tracks2.get(track_name)
        
        note_events1 = extract_note_events(track1, range(128)) if track1 else {}
        note_events2 = extract_note_events(track2, range(128)) if track2 else {}
        
        text_events1 = extract_text_events(track1) if track1 else []
        text_events2 = extract_text_events(track2) if track2 else []

        note_diffs = compare_tracks(note_events1, note_events2, TIME_WINDOW, TIME_THRESHOLD)
        text_diffs = compare_text_events(text_events1, text_events2)

        if note_diffs or text_diffs:
            note_map_key = track_name
            if track_name.startswith("PAD"):
                note_map_key = track_name.replace("PAD", "PART")

            note_map = note_name_maps.get(note_map_key, {})
            image_path = visualize_midi_changes(note_diffs, text_diffs, note_map, track_name, output_folder, session_id)
            
            if image_path:
                generated_results.append((track_name, image_path))
                logging.info(f"Differences found in '{track_name}'. Image saved to {image_path}")
        else:
            logging.info(f"'{track_name}' has no significant changes.")
            
    return generated_results