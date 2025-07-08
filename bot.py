import discord
from discord import app_commands
from discord.ext import tasks
import aiohttp
import json
import asyncio
import re
import string
import io
from difflib import get_close_matches
from datetime import datetime
import statistics
import os
import random
from pydub import AudioSegment

BOT_TOKEN = ""
JSON_DATA_URL = "https://raw.githubusercontent.com/JaydenzKoci/EncoreCustoms/refs/heads/main/data/tracks.json"
ASSET_BASE_URL = "https://jaydenzkoci.github.io/EncoreCustoms/"
CONFIG_FILE = "config.json"
TRACK_CACHE_FILE = "tracks_cache.json"
TRACK_HISTORY_FILE = "track_history.json"

intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

def load_json_file(filename: str, default_data: dict | list = None):
    """Loads a JSON file, creating it with default data if it doesn't exist."""
    if default_data is None:
        default_data = {}
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            json.dump(default_data, f, indent=4)
        return default_data
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return default_data

def save_json_file(filename: str, data: dict | list):
    """Saves data to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

async def update_bot_status():
    """Updates the bot's presence to show the current track count."""
    tracks = load_json_file(TRACK_CACHE_FILE, {"tracks": []}).get("tracks", [])
    track_count = len(tracks)
    activity = discord.Activity(type=discord.ActivityType.watching, name=f"{track_count} Tracks")
    await client.change_presence(activity=activity)
    print(f"Updated bot status: Watching {track_count} Tracks")

async def get_live_track_data() -> list | None:
    """Fetches the latest track data from the JSON URL."""
    print("Attempting to fetch live track data from source...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(JSON_DATA_URL, timeout=10) as response:
                if response.status == 200:
                    data = await response.json(content_type=None)
                    tracks_list = []
                    for track_id, track_info in data.items():
                        track_info['id'] = track_id
                        tracks_list.append(track_info)
                    print(f"Successfully fetched {len(tracks_list)} live tracks.")
                    return tracks_list
                else:
                    print(f"Error: Failed to fetch live data. Status code: {response.status}")
                    return None
    except (aiohttp.ClientError, json.JSONDecodeError, asyncio.TimeoutError) as e:
        print(f"An error occurred during live data fetching or parsing: {e}")
        return None

def get_cached_track_data() -> list:
    """Gets track data from the local cache."""
    return load_json_file(TRACK_CACHE_FILE, {"tracks": []}).get("tracks", [])

def parse_duration_to_seconds(duration_str: str) -> int:
    """Converts a duration string like '2m 49s' into total seconds."""
    if not isinstance(duration_str, str): return 0
    seconds = 0
    if (minutes_match := re.search(r'(\d+)m', duration_str)):
        seconds += int(minutes_match.group(1)) * 60
    if (seconds_match := re.search(r'(\d+)s', duration_str)):
        seconds += int(seconds_match.group(1))
    return seconds

def remove_punctuation(text: str) -> str:
    """Removes punctuation from a string for easier searching."""
    return text.translate(str.maketrans('', '', string.punctuation))

def create_difficulty_bar(level: int, max_level: int = 7) -> str:
    """Creates a visual difficulty bar."""
    if not isinstance(level, int) or level < 0: return ""
    level = min(level, max_level)
    return f"{'■' * level}{'□' * (max_level - level)}"

def fuzzy_search_tracks(tracks: list, query: str) -> list:
    """Performs an advanced, multi-faceted search on the track list."""
    search_term = remove_punctuation(query.lower())
    
    if search_term in {'saf': 'spotafake', 'lyf': 'lostyourfaith'}:
        search_term = {'saf': 'spotafake', 'lyf': 'lostyourfaith'}[search_term]
    
    sort_map = {
        'latest': ('createdAt', True, 10), 'longest': ('duration', True, 10),
        'shortest': ('duration', False, 10), 'fastest': ('bpm', True, 10),
        'slowest': ('bpm', False, 10), 'newest': ('releaseYear', True, 10),
        'oldest': ('releaseYear', False, 10)
    }
    if search_term in sort_map:
        key, reverse, limit = sort_map[search_term]
        sort_key_func = (lambda t: parse_duration_to_seconds(t.get(key, '0s'))) if key == 'duration' else (lambda t: t.get(key, 0))
        return sorted([t for t in tracks if t.get(key) is not None], key=sort_key_func, reverse=reverse)[:limit]

    exact_matches, fuzzy_matches = [], []
    for track in tracks:
        title = remove_punctuation(track.get('title', '').lower())
        artist = remove_punctuation(track.get('artist', '').lower())
        track_id = track.get('id', '').lower()

        if search_term == track_id or search_term in title or search_term in artist:
            exact_matches.append(track)
        elif get_close_matches(search_term, [title, artist], n=1, cutoff=0.7):
            fuzzy_matches.append(track)
    
    unique_results, seen_ids = [], set()
    for track in exact_matches + fuzzy_matches:
        if (track_id := track.get('id')) not in seen_ids:
            unique_results.append(track)
            seen_ids.add(track_id)
    return unique_results

def format_key(key_str: str) -> str:
    """Formats the musical key to show both flat and sharp equivalents."""
    if not key_str or not isinstance(key_str, str):
        return "N/A"
        
    key_map = {
        "A♭": "G♯", "B♭": "A♯", "D♭": "C♯", "E♭": "D♯", "G♭": "F♯",
    }

    found_flat = None
    for flat in key_map.keys():
        if flat in key_str:
            found_flat = flat
            break
            
    if found_flat:
        sharp_equivalent = key_map[found_flat]
        return f"{sharp_equivalent} / {key_str}"

    return key_str

def create_track_embed_and_view(track: dict, author_id: int, is_log: bool = False):
    """Creates the embed and view for a given track."""
    embed_title = "Track Added" if is_log else None
    color = discord.Color.green() if is_log else discord.Color.purple()
    description = f"## {track.get('title', 'N/A')} - {track.get('artist', 'N/A')}"
    
    embed = discord.Embed(
        title=embed_title,
        description=description,
        color=color
    )
    if track.get('cover'):
        embed.set_thumbnail(url=f"{ASSET_BASE_URL}/assets/covers/{track.get('cover')}")

    difficulties = track.get('difficulties', {})
    valid_diffs = [d for d in difficulties.values() if isinstance(d, int) and d != -1]
    avg_difficulty = statistics.mean(valid_diffs) if valid_diffs else 0
    
    embed.add_field(name="Release Year", value=str(track.get('releaseYear', 'N/A')))
    embed.add_field(name="Album", value=track.get('album', 'N/A'))
    embed.add_field(name="Genre", value=track.get('genre', 'N/A'))
    embed.add_field(name="Duration", value=track.get('duration', 'N/A'))
    embed.add_field(name="BPM", value=str(track.get('bpm', 'N/A')))
    embed.add_field(name="Key", value=format_key(track.get('key', 'N/A')))
    embed.add_field(name="Charter", value=track.get('charter', 'N/A'))
    embed.add_field(name="Rating", value=track.get('rating', 'N/A'))
    embed.add_field(name="Avg. Difficulty", value=f"`{create_difficulty_bar(round(avg_difficulty))}`")
    embed.add_field(name="Shortname", value=f"`{track.get('id', 'N/A')}`")
    if (loading_phrase := track.get('loading_phrase')):
        embed.add_field(name="Loading Phrase", value=f"\"{loading_phrase}\"")
    
    inst_map = {
        'vocals': 'Vocals', 'lead': 'Lead', 'bass': 'Bass', 'drums': 'Drums',
        'plastic-bass': 'Pro Bass', 'plastic-drums': 'Pro Drums',
        'plastic-guitar': 'Pro Lead', 'plastic-keys': 'Pro Keys'
    }
    diff_text = "\n".join(
        f"{name:<12}: {create_difficulty_bar(lvl)}"
        for inst, name in inst_map.items()
        if (lvl := difficulties.get(inst)) is not None and lvl != -1
    )
    if diff_text:
        embed.add_field(name="Instrument Difficulties", value=f"```\n{diff_text}```", inline=False)

    if (created_at := track.get('createdAt')):
        ts = int(datetime.fromisoformat(created_at.replace('Z', '+00:00')).timestamp())
        embed.add_field(name="Date Added", value=f"<t:{ts}:F>", inline=False)

    return embed, TrackInfoView(track=track, author_id=author_id)

def create_update_log_embed(old_track: dict, new_track: dict) -> tuple[discord.Embed | None, dict]:
    """Creates a detailed embed for a modified track and a dictionary of changes."""
    
    embed = discord.Embed(
        title="Track Modified",
        description=f"## {new_track.get('title', 'N/A')} - {new_track.get('artist', 'N/A')}",
        color=discord.Color.orange(),
        timestamp=datetime.now()
    )
    if new_track.get('cover'):
        embed.set_thumbnail(url=f"{ASSET_BASE_URL}/assets/covers/{new_track.get('cover')}")

    changes_dict = {}
    has_changes = False

    def flatten(d, parent_key='', sep='.'):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten(v, new_key))
            else:
                items[new_key] = v
        return items

    flat_old, flat_new = flatten(old_track), flatten(new_track)
    all_keys = sorted(list(set(flat_old.keys()) | set(flat_new.keys())))
    ignored_keys = ['id', 'rotated', 'glowTimes', 'modalShadowColors', 'title', 'artist', 'cover']

    key_name_map = {
        "createdAt": "Creation Date",
        "releaseYear": "Release Year",
        "songlink": "Song Link",
        "preview_time": "Preview Start Time",
        "preview_end_time": "Preview End Time",
        "difficulties.vocals": "Vocals Difficulty",
        "difficulties.lead": "Lead Difficulty",
        "difficulties.bass": "Bass Difficulty",
        "difficulties.drums": "Drum Difficulty",
        "difficulties.keys": "Keys Difficulty",
        "difficulties.plastic-vocals": "Pro Vocals Difficulty",
        "difficulties.plastic-guitar": "Pro Lead Difficulty",
        "difficulties.plastic-bass": "Pro Bass Difficulty",
        "difficulties.plastic-drums": "Pro Drum Difficulty",
        "difficulties.plastic-keys": "Pro Keys Difficulty",
    }

    change_strings = []
    for key in all_keys:
        if any(key.startswith(ignored) for ignored in ignored_keys): continue
        
        old_val, new_val = flat_old.get(key), flat_new.get(key)
        if old_val != new_val:
            has_changes = True
            key_title = key_name_map.get(key, key.replace('.', ' ').title())
            changes_dict[key] = {'old': old_val, 'new': new_val}
            
            change_str = (
                f"**{key_title} | Changed**\n"
                f"`Old: {old_val or 'N/A'}`\n"
                f"`New: {new_val or 'N/A'}`"
            )
            change_strings.append(change_str)
    
    if not has_changes: return None, {}
    
    final_description = embed.description + "\n\n" + "\n\n".join(change_strings)
    
    if len(final_description) > 4096:
        final_description = final_description[:4093] + "..."
    
    embed.description = final_description

    return embed, changes_dict

class TrackInfoView(discord.ui.View):
    """A view with buttons for a single track."""
    def __init__(self, track: dict, author_id: int):
        super().__init__(timeout=300.0)
        self.track = track
        self.author_id = author_id

        # Row 0: Previews
        if track.get('id'):
            self.add_item(self.PreviewAudioButton(track=track))
        if track.get('videoUrl'):
            self.add_item(self.PreviewVideoButton(track=track))
        
        # Row 1: Links
        if track.get('songlink'):
            self.add_item(discord.ui.Button(label="Stream Song", url=track.get('songlink'), row=1))
        if track.get('download'):
            self.add_item(discord.ui.Button(label="Download Chart", url=track.get('download'), row=1))

        # Row 2: Instrument Videos
        youtube_links = track.get('youtubeLinks', {})
        for part, name in {'vocals': 'Vocals', 'lead': 'Lead', 'drums': 'Drums', 'bass': 'Bass'}.items():
            link = youtube_links.get(part) or (youtube_links.get('guitar') if part == 'lead' else None)
            if link:
                self.add_item(self.InstrumentVideoButton(part_name=name, link=link))

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        # This view is now public, so no author check needed.
        return True

    class PreviewAudioButton(discord.ui.Button):
        def __init__(self, track: dict):
            super().__init__(label="Preview Audio", style=discord.ButtonStyle.green, row=0)
            self.track = track

        async def callback(self, interaction: discord.Interaction):
            preview_url = f"{ASSET_BASE_URL}/assets/audio/{self.track['id']}.mp3"
            await interaction.response.defer(ephemeral=True, thinking=True)
            try:
                async with aiohttp.ClientSession() as s, s.get(preview_url) as r:
                    if r.status != 200:
                        await interaction.followup.send("Could not download the audio preview.", ephemeral=True)
                        return
                    audio = AudioSegment.from_file(io.BytesIO(await r.read()))
                    start, end = self.track.get('preview_time'), self.track.get('preview_end_time')
                    trimmed = audio[start:end] if start is not None and end is not None else audio
                    
                    buffer = io.BytesIO()
                    trimmed.export(buffer, format="mp3")
                    buffer.seek(0)
                    await interaction.followup.send(file=discord.File(buffer, "preview.mp3"), ephemeral=True)
            except Exception as e:
                print(f"Error fetching audio preview: {e}")
                await interaction.followup.send("An error occurred while fetching the audio preview.", ephemeral=True)

    class PreviewVideoButton(discord.ui.Button):
        def __init__(self, track: dict):
            super().__init__(label="Preview Video", style=discord.ButtonStyle.primary, row=0)
            self.track = track
        
        async def callback(self, interaction: discord.Interaction):
            video_url = f"{ASSET_BASE_URL}/assets/preview/{self.track['videoUrl']}"
            await interaction.response.send_message(f"Here is the video preview link:\n{video_url}", ephemeral=True)

    class InstrumentVideoButton(discord.ui.Button):
        def __init__(self, part_name: str, link: str):
            super().__init__(label=f"{part_name} Video", row=2)
            self.link = link
            self.part_name = part_name
        
        async def callback(self, interaction: discord.Interaction):
            await interaction.response.send_message(f"**{self.part_name} Video:**\n{self.link}", ephemeral=True)

class TrackSelectDropdown(discord.ui.Select):
    """Dropdown for selecting a track from search results."""
    def __init__(self, tracks: list, command_type: str):
        self.tracks_map = {t['id']: t for t in tracks[:25]}
        options = [discord.SelectOption(label=t['title'], value=t['id'], description=t['artist']) for t in self.tracks_map.values()]
        super().__init__(placeholder=f"Select from {len(tracks)} results...", options=options)
        self.command_type = command_type

    async def callback(self, interaction: discord.Interaction):
        track = self.tracks_map.get(self.values[0])
        if not track: return
        
        self.view.stop()
        if self.command_type == 'info':
            embed, view = create_track_embed_and_view(track, interaction.user.id)
            await interaction.response.edit_message(content=None, embed=embed, view=view)
        elif self.command_type == 'history':
            view = HistoryPaginatorView(track, author_id=interaction.user.id)
            await interaction.response.edit_message(content=None, embed=view.create_embed(), view=view)

class TrackSelectionView(discord.ui.View):
    """View containing the TrackSelectDropdown."""
    def __init__(self, tracks: list, author_id: int, command_type: str):
        super().__init__(timeout=60.0)
        self.author_id = author_id
        self.add_item(TrackSelectDropdown(tracks, command_type))
        self.message: discord.InteractionMessage = None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.author_id:
            await interaction.response.send_message("This isn't your search session!", ephemeral=True)
            return False
        return True

    async def on_timeout(self):
        if self.message:
            for item in self.children: item.disabled = True
            await self.message.edit(content="Search timed out.", view=self)

class HistoryPaginatorView(discord.ui.View):
    """Paginator for viewing a track's update history."""
    def __init__(self, track: dict, author_id: int):
        super().__init__(timeout=120.0)
        self.track = track
        self.author_id = author_id
        self.history = load_json_file(TRACK_HISTORY_FILE, {}).get(track['id'], [])
        self.current_page = 0
        self.page_size = 3  # Number of history entries per page
        self.total_pages = (len(self.history) + self.page_size - 1) // self.page_size
        self.message: discord.InteractionMessage = None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.author_id:
            await interaction.response.send_message("This isn't your command!", ephemeral=True)
            return False
        return True

    def create_embed(self) -> discord.Embed:
        embed = discord.Embed(title=f"Update History for {self.track['title']}", color=discord.Color.blue())
        if not self.history:
            embed.description = "No update history found for this track."
            return embed
            
        start_index = self.current_page * self.page_size
        page_entries = self.history[start_index : start_index + self.page_size]
        
        desc = ""
        for entry in page_entries:
            ts = int(datetime.fromisoformat(entry['timestamp']).timestamp())
            desc += f"**<t:{ts}:F>**\n"
            for key, values in entry['changes'].items():
                key_title = "BPM" if key == "bpm" else key.replace('.', ' ').title()
                desc += f"• **{key_title}**: `{values['old'] or 'N/A'} → {values['new'] or 'N/A'}`\n"
            desc += "\n"

        embed.description = desc
        embed.set_footer(text=f"Page {self.current_page + 1}/{self.total_pages}")
        return embed

    async def update_message(self, interaction: discord.Interaction):
        self.prev_button.disabled = self.current_page == 0
        self.next_button.disabled = self.current_page >= self.total_pages - 1
        await interaction.response.edit_message(embed=self.create_embed(), view=self)

    @discord.ui.button(label="◀", style=discord.ButtonStyle.grey)
    async def prev_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.current_page > 0:
            self.current_page -= 1
            await self.update_message(interaction)

    @discord.ui.button(label="▶", style=discord.ButtonStyle.grey)
    async def next_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            await self.update_message(interaction)

# --- Background Task ---
@tasks.loop(seconds=10)
async def check_for_updates():
    """Checks for track updates, logs them, and saves history."""
    config = load_json_file(CONFIG_FILE)
    log_channels = config.get('log_channels', {})
    if not log_channels: return

    print("Checking for track updates...")
    live_tracks = await get_live_track_data()
    if live_tracks is None:
        print("Update check failed: Could not fetch live data.")
        return

    cached_data = load_json_file(TRACK_CACHE_FILE, {"tracks": []})
    cached_tracks = cached_data.get("tracks", [])
    
    old_tracks_by_id = {t['id']: t for t in cached_tracks}
    new_tracks_by_id = {t['id']: t for t in live_tracks}
    
    added_ids = new_tracks_by_id.keys() - old_tracks_by_id.keys()
    removed_ids = old_tracks_by_id.keys() - new_tracks_by_id.keys()
    modified_tracks = [
        {'old': old_tracks_by_id[tid], 'new': new_tracks_by_id[tid]}
        for tid in new_tracks_by_id.keys() & old_tracks_by_id.keys()
        if old_tracks_by_id[tid] != new_tracks_by_id[tid]
    ]

    if not (added_ids or removed_ids or modified_tracks):
        print("No track updates found.")
        return

    print("Changes detected! Processing logs and history.")
    history_data = load_json_file(TRACK_HISTORY_FILE, {})
    embeds_to_send = {cid: [] for cid in log_channels.values()}

    for tid in added_ids:
        track = new_tracks_by_id[tid]
        embed, _ = create_track_embed_and_view(track, client.user.id, is_log=True)
        for cid in log_channels.values(): embeds_to_send[cid].append(embed)

    if removed_ids:
        embed = discord.Embed(title="Tracks Removed", color=discord.Color.red(), description="\n".join(f"• **{old_tracks_by_id[tid]['title']}**" for tid in removed_ids))
        for cid in log_channels.values(): embeds_to_send[cid].append(embed)
    
    for mod_info in modified_tracks:
        embed, changes = create_update_log_embed(mod_info['old'], mod_info['new'])
        if embed:
            for cid in log_channels.values(): embeds_to_send[cid].append(embed)
            
            track_id = mod_info['new']['id']
            history_data.setdefault(track_id, []).insert(0, {'timestamp': datetime.now().isoformat(), 'changes': changes})

    for cid, embeds in embeds_to_send.items():
        if (channel := client.get_channel(int(cid))):
            try:
                for i in range(0, len(embeds), 10): await channel.send(embeds=embeds[i:i+10])
            except discord.Forbidden: print(f"Failed to send log to channel {cid}: Missing permissions.")
            except Exception as e: print(f"Failed to send log message to {cid}: {e}")

    save_json_file(TRACK_HISTORY_FILE, history_data)
    save_json_file(TRACK_CACHE_FILE, {"tracks": live_tracks})
    await update_bot_status()

# --- Bot Events ---
@client.event
async def on_ready():
    """Called when the bot is ready."""
    # Initial data fetch and cache creation
    live_tracks = await get_live_track_data()
    if live_tracks:
        save_json_file(TRACK_CACHE_FILE, {"tracks": live_tracks})
    
    await tree.sync()
    await update_bot_status()
    check_for_updates.start()
    print(f'Logged in as {client.user} (ID: {client.user.id})')
    print('Commands synced and bot is ready.')

# --- Autocomplete ---
async def track_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    """Autocompletes track names for commands."""
    tracks = get_cached_track_data()
    if not current: return [app_commands.Choice(name=t['title'], value=t['title']) for t in tracks[:25]]
    
    choices = []
    for track in tracks:
        if current.lower() in track.get('title', '').lower():
            if track['title'] not in [c.name for c in choices]:
                choices.append(app_commands.Choice(name=track['title'], value=track['title']))
    return choices[:25]

# --- Commands ---
@tree.command(name="trackinfo", description="Get detailed information about a specific track.")
@app_commands.autocomplete(track_name=track_autocomplete)
@app_commands.describe(track_name="Search by title, artist, ID, or keywords like 'newest', 'fastest', etc.")
async def trackinfo(interaction: discord.Interaction, track_name: str):
    await interaction.response.defer()
    matched_tracks = fuzzy_search_tracks(get_cached_track_data(), track_name)
    
    if not matched_tracks:
        await interaction.followup.send(f"Sorry, no tracks were found matching your query: '{track_name}'.")
        return
    
    if len(matched_tracks) == 1:
        embed, view = create_track_embed_and_view(matched_tracks[0], interaction.user.id)
        await interaction.followup.send(embed=embed, view=view)
    else:
        view = TrackSelectionView(matched_tracks, interaction.user.id, 'info')
        view.message = await interaction.followup.send(f"Found {len(matched_tracks)} results. Please select one:", view=view, ephemeral=True)

@tree.command(name="trackhistory", description="Check the update history of a specific track.")
@app_commands.autocomplete(track_name=track_autocomplete)
@app_commands.describe(track_name="The name of the track to check the history for.")
async def trackhistory(interaction: discord.Interaction, track_name: str):
    await interaction.response.defer(ephemeral=True)
    matched_tracks = fuzzy_search_tracks(get_cached_track_data(), track_name)

    if not matched_tracks:
        await interaction.followup.send(f"Sorry, no tracks were found matching your query: '{track_name}'.", ephemeral=True)
        return

    if len(matched_tracks) == 1:
        track = matched_tracks[0]
        view = HistoryPaginatorView(track, author_id=interaction.user.id)
        view.message = await interaction.followup.send(embed=view.create_embed(), view=view, ephemeral=True)
    else:
        view = TrackSelectionView(matched_tracks, interaction.user.id, 'history')
        view.message = await interaction.followup.send(f"Found {len(matched_tracks)} results. Please select one:", view=view, ephemeral=True)


@tree.command(name="setlogchannel", description="Sets this channel for track update notifications.")
@app_commands.default_permissions(administrator=True)
async def setlogchannel(interaction: discord.Interaction):
    if not interaction.guild:
        await interaction.response.send_message("This command can only be used in a server.", ephemeral=True)
        return
    config = load_json_file(CONFIG_FILE, {"log_channels": {}})
    config['log_channels'][str(interaction.guild.id)] = interaction.channel.id
    save_json_file(CONFIG_FILE, config)
    await interaction.response.send_message(f"✅ Update log channel has been set to {interaction.channel.mention}.", ephemeral=True)

# --- Debug Commands ---
test_group = app_commands.Group(name="testlog", description="Commands to test logging functions.", default_permissions=discord.Permissions(administrator=True))

@test_group.command(name="add", description="Tests the 'Track Added' log message.")
@app_commands.describe(track_id="Optional ID of a specific track to use for the test.")
async def test_add(interaction: discord.Interaction, track_id: str = None):
    await interaction.response.defer(ephemeral=True)
    all_tracks = get_cached_track_data()
    if not all_tracks:
        await interaction.followup.send("Could not fetch track data for the test.", ephemeral=True)
        return

    test_track = None
    if track_id:
        test_track = discord.utils.get(all_tracks, id=track_id)
        if not test_track:
            await interaction.followup.send(f"Could not find a track with ID '{track_id}'.", ephemeral=True)
            return
    else:
        test_track = random.choice(all_tracks)

    embed, _ = create_track_embed_and_view(test_track, interaction.user.id, is_log=True)
    await interaction.followup.send("Here is a preview of the 'Track Added' log embed:", embed=embed, ephemeral=True)

@test_group.command(name="remove", description="Tests the 'Track Removed' log message.")
@app_commands.describe(track_id="Optional ID of a specific track to use for the test.")
async def test_remove(interaction: discord.Interaction, track_id: str = None):
    await interaction.response.defer(ephemeral=True)
    all_tracks = get_cached_track_data()
    if not all_tracks:
        await interaction.followup.send("Could not fetch track data for the test.", ephemeral=True)
        return

    test_track = None
    if track_id:
        test_track = discord.utils.get(all_tracks, id=track_id)
        if not test_track:
            await interaction.followup.send(f"Could not find a track with ID '{track_id}'.", ephemeral=True)
            return
    else:
        test_track = random.choice(all_tracks)
    
    embed = discord.Embed(title="Tracks Removed", color=discord.Color.red(), description=f"• **{test_track['title']}**")
    await interaction.followup.send("Here is a preview of the 'Track Removed' log embed:", embed=embed, ephemeral=True)

@test_group.command(name="modify", description="Tests the 'Track Modified' log message.")
@app_commands.describe(track_id="Optional ID of a specific track to use for the test.")
async def test_modify(interaction: discord.Interaction, track_id: str = None):
    await interaction.response.defer(ephemeral=True)
    all_tracks = get_cached_track_data()
    if not all_tracks:
        await interaction.followup.send("Could not fetch track data for the test.", ephemeral=True)
        return

    test_track_new = None
    if track_id:
        test_track_new = discord.utils.get(all_tracks, id=track_id)
        if not test_track_new:
            await interaction.followup.send(f"Could not find a track with ID '{track_id}'.", ephemeral=True)
            return
    else:
        test_track_new = random.choice(all_tracks)

    test_track_old = test_track_new.copy()
    
    # Modify all fields for a comprehensive test
    test_track_old['title'] = "Old Test Title"
    test_track_old['artist'] = "Old Test Artist"
    test_track_old['album'] = "Old Test Album"
    test_track_old['charter'] = "Old Test Charter"
    test_track_old['releaseYear'] = 1999
    test_track_old['genre'] = "Old Genre"
    test_track_old['cover'] = "old_cover.png"
    test_track_old['bpm'] = (test_track_new.get('bpm') or 120) - 20
    test_track_old['duration'] = "1m 1s"
    test_track_old['key'] = "C Minor"
    test_track_old['songlink'] = "http://example.com/old"
    test_track_old['download'] = "http://example.com/old_download"
    test_track_old['preview_time'] = 1000
    test_track_old['preview_end_time'] = 2000
    test_track_old['createdAt'] = '2000-01-01T00:00:00.000Z'
    
    test_track_old['difficulties'] = test_track_old.get('difficulties', {}).copy()
    test_track_old['difficulties']['lead'] = 1
    test_track_old['difficulties']['drums'] = 1


    embed, _ = create_update_log_embed(test_track_old, test_track_new)
    if embed:
        await interaction.followup.send("Here is a preview of the 'Track Modified' log embed:", embed=embed, ephemeral=True)
    else:
        await interaction.followup.send("Test failed: No changes were generated for the test track.", ephemeral=True)

tree.add_command(test_group)

# --- Main Execution ---
if __name__ == "__main__":
    try:
        client.run(BOT_TOKEN)
    except discord.errors.LoginFailure:
        print("Login failed. Check your bot token and intents.")
    except Exception as e:
        print(f"An error occurred while running the bot: {e}")
