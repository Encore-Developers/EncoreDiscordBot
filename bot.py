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
from datetime import datetime, timedelta
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
SUGGESTIONS_FILE = "suggestions.json"
CHANGELOG_FILE = "changelog.json"

intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

KEY_NAME_MAP = { # I added this for /trackhistory
    "album": "Album",
    "ageRating": "Age Rating",
    "bpm": "BPM",
    "charter": "Charter",
    "coverArist": "Cover | Artist",
    "createdAt": "Creation Date",
    "download": "Download",
    "doubleBass": "Double Bass",
    "duration": "Duration",
    "genre": "Genre",
    "is_cover": "Is Cover",
    "is_verified": "Is Verified",
    "key": "Key",
    "loading_phrase": "Loading Phrase",
    "newYear": "Cover | Release Year",
    "preview_time": "Preview Start Time",
    "preview_end_time": "Preview End Time",
    "proVoxHarmonies": "Pro Vox Harmonies",
    "releaseYear": "Release Year",
    "songlink": "Song Link",
    "source": "Source",
    "difficulties.vocals": "Vocals Difficulty",
    "difficulties.lead": "Lead Difficulty",
    "difficulties.rhythm": "Rhythm Difficulty",
    "difficulties.bass": "Bass Difficulty",
    "difficulties.drums": "Drums Difficulty",
    "difficulties.keys": "Keys Difficulty",
    "difficulties.pro-vocals": "Pro Vocals Difficulty",
    "difficulties.plastic-guitar": "Pro Lead Difficulty",
    "difficulties.plastic-rhythm": "Pro Rhythm Difficulty",
    "difficulties.plastic-bass": "Pro Bass Difficulty",
    "difficulties.plastic-drums": "Pro Drums Difficulty",
    "difficulties.plastic-keys": "Pro Keys Difficulty",
    "difficulties.real-guitar": "Real Guitar Difficulty",
    "difficulties.real-keys": "Real Keys Difficulty",
    "difficulties.real-bass": "Real Bass Difficulty",
    "difficulties.real-drums": "Real Drums Difficulty",
}

def load_json_file(filename: str, default_data: dict | list = None):
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
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

async def log_error_to_channel(error_message: str):
    config = load_json_file(CONFIG_FILE)
    error_channel_id = config.get('error_log_channels', {}).get('default')
    if error_channel_id:
        channel = client.get_channel(int(error_channel_id))
        if channel:
            try:
                embed = discord.Embed(
                    title="Bot Error",
                    description=error_message[:4000],
                    color=discord.Color.red(),
                    timestamp=datetime.now()
                )
                await channel.send(embed=embed)
            except discord.Forbidden:
                print(f"Failed to send error log to channel {error_channel_id}: Missing permissions.")
            except Exception as e:
                print(f"Failed to send error log message: {e}")

async def update_bot_status():
    try:
        tracks = load_json_file(TRACK_CACHE_FILE, {"tracks": []}).get("tracks", [])
        track_count = len(tracks)
        activity = discord.Activity(type=discord.ActivityType.playing, name=f"{track_count} Tracks")
        await client.change_presence(activity=activity)
        print(f"Updated bot status: Playing {track_count} Tracks")
    except Exception as e:
        await log_error_to_channel(f"Error updating bot status: {str(e)}")

async def get_live_track_data() -> list | None:
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
                    await log_error_to_channel(f"Failed to fetch live data. Status code: {response.status}")
                    return None
    except (aiohttp.ClientError, json.JSONDecodeError, asyncio.TimeoutError) as e:
        await log_error_to_channel(f"Error during live data fetching or parsing: {str(e)}")
        return None

def get_cached_track_data() -> list:
    try:
        return load_json_file(TRACK_CACHE_FILE, {"tracks": []}).get("tracks", [])
    except Exception as e:
        asyncio.create_task(log_error_to_channel(f"Error reading track cache: {str(e)}"))
        return []

def parse_duration_to_seconds(duration_str: str) -> int:
    try:
        if not isinstance(duration_str, str): return 0
        seconds = 0
        if (minutes_match := re.search(r'(\d+)m', duration_str)):
            seconds += int(minutes_match.group(1)) * 60
        if (seconds_match := re.search(r'(\d+)s', duration_str)):
            seconds += int(seconds_match.group(1))
        return seconds
    except Exception as e:
        asyncio.create_task(log_error_to_channel(f"Error parsing duration: {str(e)}"))
        return 0

def remove_punctuation(text: str) -> str:
    try:
        return text.translate(str.maketrans('', '', string.punctuation))
    except Exception as e:
        asyncio.create_task(log_error_to_channel(f"Error removing punctuation: {str(e)}"))
        return text

def create_difficulty_bar(level: int, max_level: int = 7) -> str:
    try:
        if not isinstance(level, int) or level < 0: return ""
        level = min(level, max_level)
        return f"{'■' * level}{'□' * (max_level - level)}"
    except Exception as e:
        asyncio.create_task(log_error_to_channel(f"Error creating difficulty bar: {str(e)}"))
        return ""

def calculate_average_difficulty(track: dict) -> float:
    try:
        difficulties = track.get('difficulties', {})
        valid_diffs = [d for d in difficulties.values() if isinstance(d, int) and d != -1]
        if not valid_diffs:
            return 0.0
        return statistics.mean(valid_diffs)
    except Exception:
        return 0.0

def fuzzy_search_tracks(tracks: list, query: str, sort_method: str = None) -> list:
    try:
        sort_map = {
            'latest': ('createdAt', True, 25),
            'earliest': ('createdAt', False, 25),
            'longest': ('duration', True, 25),
            'shortest': ('duration', False, 25),
            'fastest': ('bpm', True, 25),
            'slowest': ('bpm', False, 25),
            'newest': ('releaseYear', True, 25),
            'oldest': ('releaseYear', False, 25),
            'charter': ('charter', False, 25),
            'charter_za': ('charter', True, 25),
            'hardest': ('avg_difficulty', True, 25),
            'easiest': ('avg_difficulty', False, 25)
        }
        if sort_method and sort_method.lower() in sort_map:
            key, reverse, limit = sort_map[sort_method.lower()]
            
            if key == 'duration':
                sort_key_func = lambda t: parse_duration_to_seconds(t.get(key, '0s'))
            elif key == 'createdAt':
                sort_key_func = lambda t: datetime.fromisoformat(t.get(key, '1970-01-01T00:00:00Z').replace('Z', '+00:00')).timestamp()
            elif key == 'charter':
                sort_key_func = lambda t: t.get(key, '').lower() 
            elif key == 'avg_difficulty':
                sort_key_func = calculate_average_difficulty
            else: 
                sort_key_func = lambda t: t.get(key, 0) if isinstance(t.get(key, 0), (int, float)) else 0

            if key == 'avg_difficulty':
                sortable_tracks = tracks
            else:
                sortable_tracks = [t for t in tracks if t.get(key) is not None and t.get(key) != '']
            
            sorted_tracks = sorted(sortable_tracks, key=sort_key_func, reverse=reverse)
            return sorted_tracks[:limit]

        if not query:
            return []

        search_term = remove_punctuation(query.lower())
        search_term_map = {'jjmt': 'jetpackjoyridetheme', 'peakmobilegame': 'jetpackjoyridetheme'}
        search_term = search_term_map.get(search_term, search_term)
        
        exact_matches, fuzzy_matches = [], []
        for track in tracks:
            title = remove_punctuation(track.get('title', '').lower())
            artist = remove_punctuation(track.get('artist', '').lower())
            track_id = track.get('id', '').lower()

            if search_term == track_id or search_term in title or search_term in artist:
                exact_matches.append(track)
            elif get_close_matches(search_term, [title, artist], n=1, cutoff=0.7):
                fuzzy_matches.append(track)
        
        filtered_tracks, seen_ids = [], set()
        for track in exact_matches + fuzzy_matches:
            if (track_id := track.get('id')) not in seen_ids:
                filtered_tracks.append(track)
                seen_ids.add(track_id)
        
        return filtered_tracks

    except Exception as e:
        asyncio.create_task(log_error_to_channel(f"Error in fuzzy search/sort: {str(e)}"))
        return []

def format_key(key_str: str) -> str:
    try:
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
    except Exception as e:
        asyncio.create_task(log_error_to_channel(f"Error formatting key: {str(e)}"))
        return "N/A"

def create_track_embed_and_view(track: dict, author_id: int, is_log: bool = False):
    try:
        embed_title = "Track Added" if is_log else None
        
        if is_log:
            color = discord.Color.green()
        else:
            source = track.get('source', '').lower()
            if source in ['custom', 'encore']:
                color = discord.Color(0x7d1f6e)
            elif 'rb' in source:
                color = discord.Color.blue()
            elif 'gh' in source:
                color = discord.Color.orange()
            else:
                color = discord.Color.purple()

        description = f"## {track.get('title', 'N/A')} - {track.get('artist', 'N/A')}"
        
        embed = discord.Embed(
            title=embed_title,
            description=description,
            color=color
        )
        if (is_verified := track.get('is_verified')) is not None:
            if is_verified is True or str(is_verified).lower() == 'true':
                embed.add_field(name="", value="✅ ***Verified Track***", inline=False)
            else:
                embed.add_field(name="", value="***Unverified Track***", inline=False)

        if track.get('cover'):
            embed.set_thumbnail(url=f"{ASSET_BASE_URL}/assets/covers/{track.get('cover')}")

        avg_difficulty = calculate_average_difficulty(track)

        embed.add_field(name="Release Year", value=str(track.get('releaseYear', 'N/A')))
        embed.add_field(name="Album", value=track.get('album', 'N/A'))
        embed.add_field(name="Genre", value=track.get('genre', 'N/A'))
        embed.add_field(name="Duration", value=track.get('duration', 'N/A'))
        embed.add_field(name="BPM", value=str(track.get('bpm', 'N/A')))
        embed.add_field(name="Key", value=format_key(track.get('key', 'N/A')))
        embed.add_field(name="Charter", value=track.get('charter', 'N/A'))
        embed.add_field(name="Rating", value=track.get('ageRating', 'N/A'))
        embed.add_field(name="Avg. Difficulty", value=f"`{create_difficulty_bar(round(avg_difficulty))}`")
        embed.add_field(name="Shortname", value=f"`{track.get('id', 'N/A')}`")
        
        if (loading_phrase := track.get('loading_phrase')):
            embed.add_field(name="Loading Phrase", value=f"\"{loading_phrase}\"")
        
        instrument_display_order = [
            ('vocals', 'Vocals'),
            ('lead', 'Lead'),
            ('keys', 'Keys'),
            ('bass', 'Bass'),
            ('drums', 'Drums'),
            ('pro-vocals', 'Pro Vocals'),
            ('plastic-guitar', 'Pro Lead'),
            ('plastic-keys', 'Pro Keys'),
            ('plastic-bass', 'Pro Bass'),
            ('plastic-drums', 'Pro Drums'),
            ('real-guitar', 'Real Guitar'),
            ('real-keys', 'Real Keys'),
            ('real-bass', 'Real Bass'),
            ('real-drums', 'Real Drums'),
        ]

        difficulties = track.get('difficulties', {})
        diff_lines = []
        
        for key, name in instrument_display_order:
            if (lvl := difficulties.get(key)) is not None and lvl != -1:
                diff_lines.append(f"{name:<12}: {create_difficulty_bar(lvl)}")

        diff_text = "\n".join(diff_lines)

        if diff_text:
            embed.add_field(name="Instrument Difficulties", value=f"```\n{diff_text}```", inline=False)

        compatibility_text = "N/A"
        track_format = track.get('format')
        if track_format == 'json':
            compatibility_text = "json - (Only Compatible with Encore)"
        elif track_format == 'ini':
            compatibility_text = "ini - (Compatible with Clone Hero, YARG and Encore)"
        
        embed.add_field(name="Compatibility", value=compatibility_text, inline=True)

        if (created_at := track.get('createdAt')):
            ts = int(datetime.fromisoformat(created_at.replace('Z', '+00:00')).timestamp())
            embed.add_field(name="Date Added", value=f"<t:{ts}:F>", inline=False)
        
        return embed, TrackInfoView(track=track, author_id=author_id)
    except Exception as e:
        asyncio.create_task(log_error_to_channel(f"Error creating track embed: {str(e)}"))
        return None, None

def create_update_log_embed(old_track: dict, new_track: dict) -> tuple[discord.Embed | None, dict]:
    try:
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
            "album": "Album",
            "ageRating": "Age Rating",
            "bpm": "BPM",
            "charter": "Charter",
            "coverArist": "Cover | Artist",
            "createdAt": "Creation Date",
            "download": "Download",
            "doubleBass": "Double Bass",
            "duration": "Duration",
            "genre": "Genre",
            "is_cover": "Is Cover",
            "is_verified": "Is Verified",
            "key": "Key",
            "loading_phrase": "Loading Phrase",
            "newYear": "Cover | Release Year",
            "preview_time": "Preview Start Time",
            "preview_end_time": "Preview End Time",
            "proVoxHarmonies": "Pro Vox Harmonies",
            "releaseYear": "Release Year",
            "songlink": "Song Link",
            "source": "Source",
            "difficulties.vocals": "Vocals Difficulty",
            "difficulties.lead": "Lead Difficulty",
            "difficulties.rhythm": "Rhythm Difficulty",
            "difficulties.bass": "Bass Difficulty",
            "difficulties.drums": "Drums Difficulty",
            "difficulties.keys": "Keys Difficulty",
            "difficulties.pro-vocals": "Pro Vocals Difficulty",
            "difficulties.plastic-guitar": "Pro Lead Difficulty",
            "difficulties.plastic-rhythm": "Pro Rhythm Difficulty",
            "difficulties.plastic-bass": "Pro Bass Difficulty",
            "difficulties.plastic-drums": "Pro Drums Difficulty",
            "difficulties.plastic-keys": "Pro Keys Difficulty",
            "difficulties.real-guitar": "Real Guitar Difficulty",
            "difficulties.real-keys": "Real Keys Difficulty",
            "difficulties.real-bass": "Real Bass Difficulty",
            "difficulties.real-drums": "Real Drums Difficulty",
        }

        change_strings = []
        for key in all_keys:
            if any(key.startswith(ignored) for ignored in ignored_keys): continue
            
            old_val, new_val = flat_old.get(key), flat_new.get(key)
            if old_val != new_val:
                has_changes = True
                key_title = key_name_map.get(key, key.replace('.', ' ').title())
                changes_dict[key] = {'old': old_val, 'new': new_val}
                
                change_str = f"**{key_title}**\n```\nOld: {old_val or 'N/A'}\nNew: {new_val or 'N/A'}\n```"
                change_strings.append(change_str)
        
        if not has_changes: return None, {}
        
        final_description = embed.description + "\n\n" + "\n\n".join(change_strings)
        
        if len(final_description) > 4096:
            final_description = final_description[:4093] + "..."
        
        embed.description = final_description

        return embed, changes_dict
    except Exception as e:
        asyncio.create_task(log_error_to_channel(f"Error creating update log embed: {str(e)}"))
        return None, {}

class TrackInfoView(discord.ui.View):
    def __init__(self, track: dict, author_id: int):
        super().__init__(timeout=300.0)
        self.track = track
        self.author_id = author_id

        if track.get('id'):
            self.add_item(self.PreviewAudioButton(track=track))
        if track.get('videoUrl'):
            self.add_item(self.PreviewVideoButton(track=track))
        
        if track.get('songlink'):
            self.add_item(discord.ui.Button(label="Stream Song", url=track.get('songlink'), row=1))
        if track.get('download'):
            self.add_item(discord.ui.Button(label="Download Chart", url=track.get('download'), row=1))

        # Leftover from when the bot was for my tracks lol
        youtube_links = track.get('youtubeLinks', {})
        for part, name in {'vocals': 'Vocals', 'lead': 'Lead', 'drums': 'Drums', 'bass': 'Bass'}.items():
            link = youtube_links.get(part) or (youtube_links.get('guitar') if part == 'lead' else None)
            if link:
                self.add_item(self.InstrumentVideoButton(part_name=name, link=link))

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
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
                await log_error_to_channel(f"Error fetching audio preview: {str(e)}")
                await interaction.followup.send("An error occurred while fetching the audio preview.", ephemeral=True)

    class PreviewVideoButton(discord.ui.Button):
        def __init__(self, track: dict):
            super().__init__(label="Preview Video", style=discord.ButtonStyle.primary, row=0)
            self.track = track
        
        async def callback(self, interaction: discord.Interaction):
            try:
                video_url = f"{ASSET_BASE_URL}/assets/preview/{self.track['videoUrl']}"
                await interaction.response.send_message(f"Here is the video preview link:\n{video_url}", ephemeral=True)
            except Exception as e:
                await log_error_to_channel(f"Error in preview video button: {str(e)}")
                await interaction.response.send_message("An error occurred while processing the video preview.", ephemeral=True)

    class InstrumentVideoButton(discord.ui.Button):
        def __init__(self, part_name: str, link: str):
            super().__init__(label=f"{part_name} Video", row=2)
            self.link = link
            self.part_name = part_name
        
        async def callback(self, interaction: discord.Interaction):
            try:
                await interaction.response.send_message(f"**{self.part_name} Video:**\n{self.link}", ephemeral=True)
            except Exception as e:
                await log_error_to_channel(f"Error in instrument video button: {str(e)}")
                await interaction.response.send_message("An error occurred while processing the video link.", ephemeral=True)

class TrackSelectDropdown(discord.ui.Select):
    def __init__(self, tracks: list, command_type: str, sort: str = None):
        self.tracks_map = {t['id']: t for t in tracks[:25]}
        options = []
        sort_lower = sort.lower() if sort else ''

        for t in self.tracks_map.values():
            description = t.get('artist', 'N/A')
            
            if sort_lower in ['fastest', 'slowest']:
                description += f" | BPM: {t.get('bpm', 'N/A')}"
            elif sort_lower in ['newest', 'oldest']:
                description += f" | Year: {t.get('releaseYear', 'N/A')}"
            elif sort_lower in ['longest', 'shortest']:
                description += f" | Duration: {t.get('duration', 'N/A')}"
            elif sort_lower in ['latest', 'earliest']:
                created_at = t.get('createdAt')
                if created_at:
                    date_str = datetime.fromisoformat(created_at.replace('Z', '+00:00')).strftime('%Y-%m-%d')
                    description += f" | Added: {date_str}"
                else:
                    description += " | Added: N/A"
            elif sort_lower in ['charter', 'charter_za']:
                description += f" | Charter: {t.get('charter', 'N/A')}"
            elif sort_lower in ['hardest', 'easiest']:
                avg_diff = calculate_average_difficulty(t)
                description += f" | Avg. Diff: {round(avg_diff)}/7"


            options.append(discord.SelectOption(label=t['title'], value=t['id'], description=description))

        placeholder_text = f"Select from {len(self.tracks_map)} sorted results..." if sort else f"Select from {len(tracks)} results..."
        super().__init__(placeholder=placeholder_text, options=options)
        self.command_type = command_type

    async def callback(self, interaction: discord.Interaction):
        try:
            track = self.tracks_map.get(self.values[0])
            if not track: return
            
            self.view.stop()
            if self.command_type == 'info':
                embed, view = create_track_embed_and_view(track, interaction.user.id)
                await interaction.response.edit_message(content=None, embed=embed, view=view)
            elif self.command_type == 'history':
                view = HistoryPaginatorView(track, author_id=interaction.user.id)
                await interaction.response.edit_message(content=None, embed=view.create_embed(), view=view)
        except Exception as e:
            await log_error_to_channel(f"Error in track select dropdown: {str(e)}")
            await interaction.response.send_message("An error occurred while processing your selection.", ephemeral=True)

class TrackSelectionView(discord.ui.View):
    def __init__(self, tracks: list, author_id: int, command_type: str, sort: str = None):
        super().__init__(timeout=60.0)
        self.author_id = author_id
        self.add_item(TrackSelectDropdown(tracks, command_type, sort))
        self.message: discord.InteractionMessage = None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        try:
            if interaction.user.id != self.author_id:
                await interaction.response.send_message("This isn't your session!", ephemeral=True)
                return False
            return True
        except Exception as e:
            await log_error_to_channel(f"Error in track selection view: {str(e)}")
            return False

    async def on_timeout(self):
        try:
            if self.message:
                for item in self.children: item.disabled = True
                await self.message.edit(content="Search timed out.", view=self)
        except Exception as e:
            await log_error_to_channel(f"Error in track selection view timeout: {str(e)}")

class HistoryPaginatorView(discord.ui.View):
    def __init__(self, track: dict, author_id: int):
        super().__init__(timeout=120.0)
        self.track = track
        self.author_id = author_id
        self.history = load_json_file(TRACK_HISTORY_FILE, {}).get(track['id'], [])
        self.current_page = 0
        self.page_size = 3
        self.total_pages = (len(self.history) + self.page_size - 1) // self.page_size
        self.message: discord.InteractionMessage = None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        try:
            if interaction.user.id != self.author_id:
                await interaction.response.send_message("This isn't your command!", ephemeral=True)
                return False
            return True
        except Exception as e:
            await log_error_to_channel(f"Error in history paginator view: {str(e)}")
            return False

    def create_embed(self) -> discord.Embed:
        try:
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
                    # This line is updated to use the global KEY_NAME_MAP for consistency
                    key_title = KEY_NAME_MAP.get(key, key.replace('.', ' ').title())
                    desc += f"• **{key_title}**: `{values['old'] or 'N/A'}` → `{values['new'] or 'N/A'}`\n"
                desc += "\n"

            embed.description = desc
            embed.set_footer(text=f"Page {self.current_page + 1}/{self.total_pages}")
            return embed
        except Exception as e:
            asyncio.create_task(log_error_to_channel(f"Error creating history embed: {str(e)}"))
            return discord.Embed(title="Error", description="Failed to create history embed", color=discord.Color.red())

    async def update_message(self, interaction: discord.Interaction):
        try:
            self.prev_button.disabled = self.current_page == 0
            self.next_button.disabled = self.current_page >= self.total_pages - 1
            await interaction.response.edit_message(embed=self.create_embed(), view=self)
        except Exception as e:
            await log_error_to_channel(f"Error updating history message: {str(e)}")

    @discord.ui.button(label="◀", style=discord.ButtonStyle.grey)
    async def prev_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            if self.current_page > 0:
                self.current_page -= 1
                await self.update_message(interaction)
        except Exception as e:
            await log_error_to_channel(f"Error in history prev button: {str(e)}")

    @discord.ui.button(label="▶", style=discord.ButtonStyle.grey)
    async def next_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            if self.current_page < self.total_pages - 1:
                self.current_page += 1
                await self.update_message(interaction)
        except Exception as e:
            await log_error_to_channel(f"Error in history next button: {str(e)}")

@tasks.loop(seconds=10)
async def check_for_updates():
    try:
        config = load_json_file(CONFIG_FILE)
        log_channels = config.get('update_log_channels', {})
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
                except discord.Forbidden:
                    await log_error_to_channel(f"Failed to send update log to channel {cid}: Missing permissions.")
                except Exception as e:
                    await log_error_to_channel(f"Failed to send update log message to {cid}: {str(e)}")

        save_json_file(TRACK_HISTORY_FILE, history_data)
        save_json_file(TRACK_CACHE_FILE, {"tracks": live_tracks})
        await update_bot_status()
    except Exception as e:
        await log_error_to_channel(f"Error in check_for_updates task: {str(e)}")

@client.event
async def on_ready():
    try:
        print("Starting on_ready event...")
        live_tracks = await get_live_track_data()
        print(f"Live tracks fetched: {len(live_tracks or [])}")
        if live_tracks:
            save_json_file(TRACK_CACHE_FILE, {"tracks": live_tracks})
        
        print(f"Bot logged in as {client.user} (ID: {client.user.id})")
        print(f"Found {len(client.guilds)} guilds: {[guild.name + ' (' + str(guild.id) + ')' for guild in client.guilds]}")
        
        print("Attempting to sync commands globally...")
        try:
            await tree.sync()
            sync_summary = "Global command sync successful."
            print(sync_summary)
            await log_error_to_channel(sync_summary)
        except Exception as e:
            sync_summary = f"Global command sync failed: {str(e)}"
            print(sync_summary)
            await log_error_to_channel(sync_summary)

        await update_bot_status()
        check_for_updates.start()
        print("Bot is ready.")
    except Exception as e:
        error_msg = f"Error in on_ready event: {str(e)}"
        print(error_msg)
        await log_error_to_channel(error_msg)
        raise

async def track_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    try:
        tracks = get_cached_track_data()
        if not current:
            return [app_commands.Choice(name=t['title'], value=t['title']) for t in tracks[:25]]
        
        choices = []
        for track in tracks:
            if current.lower() in track.get('title', '').lower():
                if track['title'] not in [c.name for c in choices]:
                    choices.append(app_commands.Choice(name=track['title'], value=track['title']))
        return choices[:25]
    except Exception as e:
        await log_error_to_channel(f"Error in track_autocomplete: {str(e)}")
        return []

@tree.command(name="trackinfo", description="Get detailed information about a specific track.")
@app_commands.autocomplete(track_name=track_autocomplete)
@app_commands.describe(track_name="Search by title, artist, or ID.")
async def trackinfo(interaction: discord.Interaction, track_name: str):
    try:
        await interaction.response.defer()
        matched_tracks = fuzzy_search_tracks(get_cached_track_data(), track_name)
        
        if not matched_tracks:
            await interaction.followup.send(f"Sorry, no tracks were found matching your query: '{track_name}'")
            return
        
        if len(matched_tracks) == 1:
            embed, view = create_track_embed_and_view(matched_tracks[0], interaction.user.id)
            await interaction.followup.send(embed=embed, view=view)
        else:
            view = TrackSelectionView(matched_tracks, interaction.user.id, 'info')
            view.message = await interaction.followup.send(f"Found {len(matched_tracks)} results. Please select one:", view=view, ephemeral=True)
    except Exception as e:
        await log_error_to_channel(f"Error in trackinfo command: {str(e)}")
        await interaction.followup.send("An error occurred while processing your request.", ephemeral=True)

@tree.command(name="tracksort", description="Sorts all tracks by a specific criterion.")
@app_commands.describe(sort_by="The criterion to sort tracks by.")
@app_commands.choices(sort_by=[
    app_commands.Choice(name="Charter (A-Z)", value="charter"),
    app_commands.Choice(name="Charter (Z-A)", value="charter_za"),
    app_commands.Choice(name="Hardest (Avg. Difficulty)", value="hardest"),
    app_commands.Choice(name="Easiest (Avg. Difficulty)", value="easiest"),
    app_commands.Choice(name="Fastest (Highest BPM)", value="fastest"),
    app_commands.Choice(name="Slowest (Lowest BPM)", value="slowest"),
    app_commands.Choice(name="Newest (Recent Release Year)", value="newest"),
    app_commands.Choice(name="Oldest (Oldest Release Year)", value="oldest"),
    app_commands.Choice(name="Shortest (Shortest Length)", value="shortest"),
    app_commands.Choice(name="Longest (Longest Length)", value="longest"),
    app_commands.Choice(name="Latest (Recent Creation Date)", value="latest"),
    app_commands.Choice(name="Earliest (Oldest Creation Date)", value="earliest")
])
async def tracksort(interaction: discord.Interaction, sort_by: str):
    try:
        await interaction.response.defer()
        all_tracks = get_cached_track_data()
        sorted_tracks = fuzzy_search_tracks(all_tracks, query="", sort_method=sort_by)
        
        if not sorted_tracks:
            await interaction.followup.send("Could not find any tracks to sort.", ephemeral=True)
            return
        
        view = TrackSelectionView(sorted_tracks, interaction.user.id, 'info', sort=sort_by)
        view.message = await interaction.followup.send(f"Showing top results for tracks sorted by **{sort_by.replace('_', '-').title()}**:", view=view)
    except Exception as e:
        await log_error_to_channel(f"Error in tracksort command: {str(e)}")
        await interaction.followup.send("An error occurred while processing your request.", ephemeral=True)

@tree.command(name="trackhistory", description="Check the update history of a specific track.")
@app_commands.autocomplete(track_name=track_autocomplete)
@app_commands.describe(track_name="The name of the track to check the history for.")
async def trackhistory(interaction: discord.Interaction, track_name: str):
    try:
        await interaction.response.defer()
        matched_tracks = fuzzy_search_tracks(get_cached_track_data(), track_name)

        if not matched_tracks:
            await interaction.followup.send(f"Sorry, no tracks were found matching your query: '{track_name}'.")
            return

        if len(matched_tracks) == 1:
            track = matched_tracks[0]
            view = HistoryPaginatorView(track, author_id=interaction.user.id)
            view.message = await interaction.followup.send(embed=view.create_embed(), view=view)
        else:
            view = TrackSelectionView(matched_tracks, interaction.user.id, 'history')
            view.message = await interaction.followup.send(f"Found {len(matched_tracks)} results. Please select one:", view=view)
    except Exception as e:
        await log_error_to_channel(f"Error in trackhistory command: {str(e)}")
        await interaction.followup.send("An error occurred while processing your request.", ephemeral=True)

class SuggestionModal(discord.ui.Modal, title="Suggest a Feature"):
    suggestion_input = discord.ui.TextInput(
        label="Your Suggestion",
        style=discord.TextStyle.long,
        placeholder="Type your feature suggestion here...",
        required=True,
        max_length=1000,
    )

    async def on_submit(self, interaction: discord.Interaction):
        try:
            user_id = str(interaction.user.id)
            suggestion_data = load_json_file(SUGGESTIONS_FILE, default_data={"user_timestamps": {}, "suggestions": []})
            
            now = datetime.now()
            one_hour_ago = now - timedelta(hours=1)
            
            user_timestamps = suggestion_data["user_timestamps"].get(user_id, [])
            recent_timestamps = [ts for ts in user_timestamps if datetime.fromisoformat(ts) > one_hour_ago]
            
            if len(recent_timestamps) >= 2:
                await interaction.response.send_message("You have already made 2 suggestions in the last hour. Please try again later.", ephemeral=True)
                return

            new_suggestion = {
                "username": str(interaction.user),
                "user_id": user_id,
                "suggestion": self.suggestion_input.value,
                "timestamp": now.isoformat()
            }
            suggestion_data["suggestions"].append(new_suggestion)
            
            recent_timestamps.append(now.isoformat())
            suggestion_data["user_timestamps"][user_id] = recent_timestamps
            
            save_json_file(SUGGESTIONS_FILE, suggestion_data)
            await interaction.response.send_message("✅ Thank you! Your suggestion has been submitted.", ephemeral=True)

        except Exception as e:
            await log_error_to_channel(f"Error processing suggestion: {e}")
            await interaction.response.send_message("An error occurred while submitting your suggestion.", ephemeral=True)

class BotInfoView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        self.add_item(discord.ui.Button(label="Report a Bug", style=discord.ButtonStyle.link, url="https://github.com/JaydenzKoci/EncoreDiscordBot/issues/new"))
        self.add_item(discord.ui.Button(label="Encore Discord", style=discord.ButtonStyle.link, url="https://discord.gg/FmF8DpZVrx"))

    @discord.ui.button(label="Suggest a Feature", style=discord.ButtonStyle.green)
    async def suggest_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(SuggestionModal())

    @discord.ui.button(label="Changelog", style=discord.ButtonStyle.secondary)
    async def changelog_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            await interaction.response.defer(ephemeral=True)
            changelog_data = load_json_file(CHANGELOG_FILE)
            if not changelog_data:
                await interaction.followup.send("Could not load the changelog file.", ephemeral=True)
                return

            version = changelog_data.get("version", "N/A")
            changes = changelog_data.get("changes", ["No changes listed."])

            embed = discord.Embed(
                title=f"Changelog - Version {version}",
                description="\n".join(f"• {change}" for change in changes),
                color=discord.Color.blurple()
            )
            await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception as e:
            await log_error_to_channel(f"Error in changelog button: {str(e)}")
            await interaction.followup.send("An error occurred while fetching the changelog.", ephemeral=True)


@tree.command(name="bot-info", description="Get information about the bot.")
async def bot_info(interaction: discord.Interaction):
    try:
        await interaction.response.defer()

        changelog_data = load_json_file(CHANGELOG_FILE, {})
        version = changelog_data.get("version", "N/A")
        date_str = changelog_data.get("date")
        
        bot_update_timestamp_str = "N/A"
        if date_str:
            try:
                pdt_time = datetime.strptime(date_str, "%m-%d-%Y--%I:%M%p")
                
                utc_time = pdt_time + timedelta(hours=7)
                
                timestamp = int((utc_time - datetime(1970, 1, 1)).total_seconds())
                
                bot_update_timestamp_str = f"<t:{timestamp}:f>"
            except (ValueError, TypeError) as e:
                await log_error_to_channel(f"Could not parse date from changelog.json: '{date_str}'. Error: {e}")
                bot_update_timestamp_str = date_str 

        tracks = get_cached_track_data()
        total_tracks = len(tracks)
        verified_tracks = sum(1 for t in tracks if t.get('is_verified') is True or str(t.get('is_verified')).lower() == 'true')
        unverified_tracks = total_tracks - verified_tracks

        track_history = load_json_file(TRACK_HISTORY_FILE, {})
        total_track_updates = 0
        latest_track_update_timestamp = None
        if track_history:
            all_timestamps = []
            for track_id in track_history:
                updates = track_history[track_id]
                total_track_updates += len(updates)
                for update in updates:
                    if 'timestamp' in update and isinstance(update['timestamp'], str):
                        try:
                            all_timestamps.append(datetime.fromisoformat(update['timestamp']))
                        except ValueError:
                            print(f"Could not parse timestamp: {update['timestamp']}")
            if all_timestamps:
                latest_track_update_timestamp = max(all_timestamps)

        embed = discord.Embed(
            title="Encore Bot Information",
            description="This Bot Is Very WIP. If you find any bugs please report them.",
            color=discord.Color.purple()
        )
        
        embed.add_field(name="📊 Track Statistics", value=(
            f"**Total Tracks:** {total_tracks}\n"
            f"**Verified Tracks:** {verified_tracks}\n"
            f"**Unverified Tracks:** {unverified_tracks}"
        ), inline=True)

        last_track_update_str = f"<t:{int(latest_track_update_timestamp.timestamp())}:R>" if latest_track_update_timestamp else "N/A"
        embed.add_field(name="🔄 Track Update History", value=(
            f"**Total Updates:** {total_track_updates}\n"
            f"**Last Update:** {last_track_update_str}"
        ), inline=True)
        
        embed.add_field(name="\u200b", value="\u200b", inline=False)

        embed.add_field(name="🗓️ Last Bot Update", value=bot_update_timestamp_str, inline=False)
        
        embed.set_footer(text=f"Version {version}")
        
        await interaction.followup.send(embed=embed, view=BotInfoView())
    except Exception as e:
        await log_error_to_channel(f"Error in bot-info command: {str(e)}")
        await interaction.followup.send("An error occurred while fetching bot info.", ephemeral=True)

if __name__ == "__main__":
    try:
        client.run(BOT_TOKEN)
    except discord.errors.LoginFailure:
        print("Login failed. Check your bot token and intents.")
        asyncio.create_task(log_error_to_channel(f"Login failed: Check bot token and intents."))
    except Exception as e:
        print(f"An error occurred while running the bot: {e}")
        asyncio.create_task(log_error_to_channel(f"Bot startup error: {str(e)}"))
