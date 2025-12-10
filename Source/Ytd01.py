
import os
import asyncio
import gc
import time
import librosa
import numpy as np
from shazamio import Shazam
import yt_dlp
import soundfile as sf
from datetime import datetime
import json
import tempfile
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
AUDIO_FILENAME = "audio.wav"
RESULTS_FILENAME = "songs.json"
CHUNK_SIZE = 60 #120  # 2-minute chunks for balance of performance/accuracy

class SongRecognizer:
    def __init__(self):
        self.work_dir = os.path.join(os.getcwd(), "shazam_workspace")
        os.makedirs(self.work_dir, exist_ok=True)
        self.recognized_songs = []
        self.active = True
        self.current_url = ""
        self._load_recognized_songs()
        
    def _load_recognized_songs(self):
        """Load existing recognized songs from JSON database"""
        results_file = os.path.join(self.work_dir, RESULTS_FILENAME)
        try:
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    self.recognized_songs = json.load(f)
                print(f"Loaded {len(self.recognized_songs)} previously recognized songs")
        except Exception as e:
            print(f"Warning: Could not load song database: {e}")
            self.recognized_songs = []

    def _save_recognized_songs(self):
        """Save recognized songs to JSON database"""
        if not self.active:
            return
            
        results_file = os.path.join(self.work_dir, RESULTS_FILENAME)
        try:
            with open(results_file, 'w') as f:
                json.dump(self.recognized_songs, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save song database: {e}")

    def _is_new_song_for_url(self, artist, title, url):
        """Check if song is new for this specific URL (case-insensitive)"""
        artist_lower = artist.lower()
        title_lower = title.lower()
        
        return not any(
            song['artist'].lower() == artist_lower and 
            song['title'].lower() == title_lower and
            song['source_url'] == url
            for song in self.recognized_songs
        )

    async def download_youtube_audio(self, url):
        """Download YouTube audio as WAV file and return path"""
        self.current_url = url
        audio_file = os.path.join(self.work_dir, AUDIO_FILENAME)
        
        # Clean up existing file if it exists
        if os.path.exists(audio_file):
            try:
                os.remove(audio_file)
            except:
                pass
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': audio_file.replace('.wav', '.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': False,
            'ignoreerrors': True,
            'retries': 3,
            'no-playlist': True,  # Force single video even if playlist URL
        }
        
        try:
            print(f"⏬ Downloading: {url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: ydl.download([url])
                )
            
            if not os.path.exists(audio_file):
                return None
                
            return audio_file
        except Exception as e:
            print(f"❌ Download error: {e}")
            return None

    async def process_playlist(self, playlist_url):
        """Process each video in a playlist separately"""
        print(f"🎵 Processing playlist: {playlist_url}")
        
        # Get playlist video URLs
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'force-ipv4': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: ydl.extract_info(playlist_url, download=False)
                )
                
                if 'entries' not in info:
                    print("❌ Not a playlist or no videos found")
                    return
                
                videos = info['entries']
                print(f"📋 Found {len(videos)} videos in playlist")
                
                for i, video in enumerate(videos, 1):
                    if not video or not video.get('url'):
                        continue
                        
                    video_url = video['url']
                    print(f"\n🎬 Processing video {i}/{len(videos)}: {video.get('title', 'Unknown')}")
                    
                    # Download and process this video
                    audio_file = await self.download_youtube_audio(video_url)
                    if audio_file:
                        await self.process_audio(audio_file)
                        
                        # Clean up audio file immediately
                        try:
                            os.remove(audio_file)
                            print("♻️ Cleaned up audio file")
                        except:
                            pass
                    else:
                        print("⏭️ Skipping video due to download error")
                    
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(2)
                
        except Exception as e:
            print(f"❌ Playlist processing error: {e}")

    async def shazam_recognize(self, audio_path):
        """Recognize song using shazamio"""
        if not self.active or not os.path.exists(audio_path):
            return None
            
        shazam = Shazam()
        try:
            result = await shazam.recognize(audio_path)
            if result and result.get("matches"):
                title = result["track"]["title"]
                artist = result["track"]["subtitle"]
                
                # Only return if it's a new song FOR THIS URL
                if self._is_new_song_for_url(artist, title, self.current_url):
                    return {
                        "title": title,
                        "artist": artist,
                        "time": result.get("timestamp", 0) / 1000,
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source_url": self.current_url
                    }
        except Exception as e:
            if self.active:
                print(f"Recognition error: {e}")
        return None

    async def _process_chunk(self, chunk, sr, start_time):
        """Safe chunk processing with retries and proper cleanup"""
        max_retries = 2
        chunk_file = os.path.join(self.work_dir, f"chunk_{start_time}.wav")
        
        for attempt in range(max_retries):
            try:
                # Write chunk to file
                sf.write(chunk_file, chunk, sr)
                
                # Verify file was written
                if os.path.getsize(chunk_file) == 0:
                    raise ValueError("Empty chunk file")
                
                # Recognize song
                result = await self.shazam_recognize(chunk_file)
                return result
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"⚠️ Chunk processing failed: {e}")
                await asyncio.sleep(1)  # Wait before retry
            finally:
                # Ensure cleanup happens
                if os.path.exists(chunk_file):
                    try:
                        os.remove(chunk_file)
                    except:
                        pass
        return None

    async def process_audio(self, audio_path):
        """Main audio processing with robust error handling"""
        if not self.active or not os.path.exists(audio_path):
            return None
            
        try:
            # Load audio file with verification
            try:
                y, sr = librosa.load(audio_path, sr=44100, mono=True)
                if len(y) == 0:
                    raise ValueError("Empty audio data")
            except Exception as e:
                print(f"❌ Error loading audio: {e}")
                return None
                
            duration = librosa.get_duration(y=y, sr=sr)
            num_chunks = int(np.ceil(duration / CHUNK_SIZE))
            
            print(f"🔊 Processing {duration//60:.0f}m {duration%60:.0f}s audio in {num_chunks} chunks")
            
            results = []
            last_optimize = time.time()
            
            for i in range(num_chunks):
                if not self.active:
                    break
                    
                # Periodic optimization
                if i % 20 == 0 or time.time() - last_optimize > 300:
                    gc.collect()
                    await asyncio.sleep(0.5)
                    last_optimize = time.time()
                    print("⚙️ Optimizing resources...", end=" ", flush=True)
                
                start = i * CHUNK_SIZE
                end = min((i + 1) * CHUNK_SIZE, duration)
                
                print(f"\n🔍 {start//60:.0f}:{start%60:02.0f}-{end//60:.0f}:{end%60:02.0f}...", end=" ")
                
                # Process chunk
                chunk = y[int(start * sr):int(end * sr)]
                match = await self._process_chunk(chunk, sr, start)
                
                if match:
                    match.update({
                        "start_time": start,
                        "end_time": end
                    })
                    results.append(match)
                    
                    # Add to database (with unique check already done)
                    self.recognized_songs.append({
                        "artist": match["artist"],
                        "title": match["title"],
                        "date": match["date"],
                        "source_url": match["source_url"],
                        "timestamp": match["time"],
                        "chunk_start": match["start_time"],
                        "chunk_end": match["end_time"]
                    })
                    print(f"✅ {match['artist']} - {match['title']}")
                else:
                    print("⏩ Already recognized or no match")
            
            # Save results to file
            self._save_recognized_songs()
            return results
            
        except Exception as e:
            print(f"\n❌ Processing error: {e}")
            return None
        finally:
            gc.collect()

    async def shutdown(self):
        """Graceful shutdown procedure"""
        self.active = False
        self._save_recognized_songs()
        gc.collect()

async def main():
    recognizer = SongRecognizer()
    
    try:
        url = input("Enter YouTube URL (video or playlist): ").strip()
        
        if "list=" in url.lower():
            # Process as playlist
            await recognizer.process_playlist(url)
        else:
            # Process as single video
            audio_file = await recognizer.download_youtube_audio(url)
            if audio_file:
                await recognizer.process_audio(audio_file)
                # Clean up audio file
                try:
                    os.remove(audio_file)
                    print("♻️ Cleaned up audio file")
                except:
                    pass
            else:
                print("❌ Failed to download audio")
        
        print(f"\n🎉 Processing complete! Total songs in database: {len(recognizer.recognized_songs)}")
        
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        await recognizer.shutdown()

if __name__ == "__main__":
    print("=== YouTube Song Recognition ===")
    print(f"Chunk size: {CHUNK_SIZE}s | Database: {RESULTS_FILENAME}")
    
    try:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
    finally:
        input("\nPress Enter to exit...")
