#!/usr/bin/env python3
"""
Audio Focus System - Automatic audio track focusing based on highest dB levels
Mutes all but the loudest track at any given moment with intelligent overlap detection
"""

import os
import sys
import argparse
import numpy as np
import librosa
import soundfile as sf
import ffmpeg
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt

class AudioAnalyzer:
    """Analyzes audio files to determine dB levels and track activity"""
    
    def __init__(self, frame_rate: int = 30):
        self.frame_rate = frame_rate
        self.hop_length = None
        self.sample_rate = None
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return audio data with sample rate"""
        try:
            audio, sr = librosa.load(file_path, sr=None, mono=False)
            self.sample_rate = sr
            # Calculate hop length for frame rate analysis
            self.hop_length = sr // self.frame_rate
            return audio, sr
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None, None
    
    def extract_video_audio_tracks(self, video_path: str) -> List[str]:
        """Extract all audio tracks from video file"""
        try:
            # Get video info to determine number of audio tracks
            print(f"Probing video file: {video_path}")
            probe = ffmpeg.probe(video_path)
            audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
            
            print(f"Found {len(audio_streams)} audio tracks in video")
            
            # Print stream info for debugging
            for i, stream in enumerate(audio_streams):
                print(f"Stream {i}: {stream.get('codec_name', 'unknown')} - "
                      f"{stream.get('channels', 'unknown')} channels")
            
            audio_files = []
            for i, stream in enumerate(audio_streams):
                output_file = f"temp_track_{i+1}.wav"
                print(f"Extracting track {i+1} to {output_file}")
                
                try:
                    # Extract single audio stream - note the stream index includes video stream
                    # Stream #0:1 is first audio, Stream #0:2 is second audio based on the output
                    stream_index = i + 1  # Skip video stream at index 0
                    (
                        ffmpeg
                        .input(video_path)
                        .output(output_file, map=f'0:{stream_index}', acodec='pcm_s16le')
                        .overwrite_output()
                        .run(capture_stdout=True, capture_stderr=True)
                    )
                    audio_files.append(output_file)
                    print(f"Successfully extracted track {i+1}")
                    
                except ffmpeg.Error as e:
                    print(f"FFmpeg error extracting track {i+1}:")
                    print(f"stdout: {e.stdout.decode()}")
                    print(f"stderr: {e.stderr.decode()}")
                    
            return audio_files
            
        except Exception as e:
            print(f"Error extracting audio tracks: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def calculate_db_levels(self, audio: np.ndarray) -> np.ndarray:
        """Calculate dB levels for audio frames"""
        # Ensure mono for analysis
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=0)
            
        # Calculate RMS energy in overlapping frames
        frames = librosa.util.frame(audio, frame_length=self.hop_length, 
                                  hop_length=self.hop_length//2, axis=0)
        
        # Calculate RMS for each frame
        rms = np.sqrt(np.mean(frames**2, axis=0))
        
        # Convert to dB (with small epsilon to avoid log(0))
        db_levels = 20 * np.log10(rms + 1e-10)
        
        return db_levels
    
    def analyze_sample_file(self, video_path: str):
        """Analyze the Dialog.mp4 sample file to understand structure"""
        print("=== Analyzing Dialog.mp4 Sample ===")
        
        # Extract audio tracks
        audio_files = self.extract_video_audio_tracks(video_path)
        
        if len(audio_files) != 2:
            print(f"Expected 2 audio tracks, found {len(audio_files)}")
            return
        
        track_data = []
        for i, audio_file in enumerate(audio_files):
            audio, sr = self.load_audio(audio_file)
            if audio is not None:
                db_levels = self.calculate_db_levels(audio)
                track_data.append({
                    'track_num': i + 1,
                    'audio': audio,
                    'db_levels': db_levels,
                    'file': audio_file
                })
                print(f"Track {i+1}: {len(audio)/sr:.2f}s duration, "
                      f"avg dB: {np.mean(db_levels):.2f}, "
                      f"max dB: {np.max(db_levels):.2f}")
        
        # Analyze the landmarks described in requirements
        self._analyze_landmarks(track_data)
        
        # Clean up temp files
        for audio_file in audio_files:
            try:
                os.remove(audio_file)
            except:
                pass
                
        return track_data
    
    def _analyze_landmarks(self, track_data: List[Dict]):
        """Analyze known landmarks in Dialog.mp4 for validation"""
        print("\n=== Analyzing Known Landmarks ===")
        
        if len(track_data) != 2:
            return
            
        track1_db = track_data[0]['db_levels']
        track2_db = track_data[1]['db_levels']
        
        # Time axis (approximate, based on frame rate)
        time_axis = np.linspace(0, len(track1_db) / self.frame_rate, len(track1_db))
        
        # Find dominant track at each moment
        dominant_track = np.where(track1_db > track2_db, 1, 2)
        
        # Analyze specific time periods mentioned in requirements
        landmarks = [
            (0, 8, 1, "Track 1 talking first 8 seconds"),
            (8.2, 8.3, 2, "Track 2 blip around 8.2s"),
            (8.3, 60, 2, "Track 2 talking 8.3s to 1 minute"),
            (60, 102, 1, "Track 1 talking 1min to 1:42"),
            (102, 103, 2, "Track 2 says 'yeah' at 1:42"),
            (103, 105, 1, "Track 1 final word")
        ]
        
        print("Landmark Analysis:")
        for start_time, end_time, expected_track, description in landmarks:
            # Find frame indices for time range
            start_idx = int(start_time * self.frame_rate)
            end_idx = int(end_time * self.frame_rate)
            
            if start_idx < len(dominant_track) and end_idx <= len(dominant_track):
                segment = dominant_track[start_idx:end_idx]
                actual_track = np.bincount(segment).argmax() + 1
                confidence = np.bincount(segment).max() / len(segment)
                
                status = "✓" if actual_track == expected_track else "✗"
                print(f"{status} {description}")
                print(f"    Expected: Track {expected_track}, "
                      f"Detected: Track {actual_track} ({confidence:.1%} confidence)")
        
        # Look for overlaps (both tracks active)
        print(f"\nOverlap Detection:")
        overlap_threshold = 5  # dB difference threshold for considering both active
        
        for i in range(len(track1_db)):
            db_diff = abs(track1_db[i] - track2_db[i])
            if db_diff < overlap_threshold and max(track1_db[i], track2_db[i]) > -30:
                time_sec = i / self.frame_rate
                print(f"Potential overlap at {time_sec:.2f}s "
                      f"(Track1: {track1_db[i]:.1f}dB, Track2: {track2_db[i]:.1f}dB)")

def main():
    """Main function for initial testing"""
    print("Starting audio focus analysis...")
    
    analyzer = AudioAnalyzer()
    
    # Test with Dialog.mp4
    if os.path.exists("Dialog.mp4"):
        print("Found Dialog.mp4, analyzing...")
        try:
            analyzer.analyze_sample_file("Dialog.mp4")
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Dialog.mp4 not found in current directory")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")

if __name__ == "__main__":
    main()
