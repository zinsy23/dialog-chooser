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
from scipy.io import wavfile
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
            # Load with librosa, keep original sample rate and channels
            audio, sr = librosa.load(file_path, sr=None, mono=False)
            
            self.sample_rate = sr
            # Calculate hop length for frame rate analysis
            self.hop_length = sr // self.frame_rate
            
            print(f"Loaded {file_path}: {audio.shape} at {sr}Hz")
            return audio, sr
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            import traceback
            traceback.print_exc()
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
        # Convert to mono for analysis if multi-channel
        if len(audio.shape) > 1:
            # librosa.load with mono=False returns (channels, samples) format consistently
            # So audio.shape[0] = channels, audio.shape[1] = samples
            audio_mono = np.mean(audio, axis=0)  # Average across channels, keep all samples
        else:
            audio_mono = audio
            
        # Use librosa to calculate RMS with proper framing
        # Use hop_length (not hop_length//2) to match frame rate exactly
        rms = librosa.feature.rms(y=audio_mono, frame_length=self.hop_length, 
                                 hop_length=self.hop_length)[0]
        
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


class AudioFocusProcessor:
    """Main processor that implements the audio focus algorithm"""
    
    def __init__(self, frame_rate: int = 30, overlap_threshold: float = 5.0):
        self.analyzer = AudioAnalyzer(frame_rate)
        self.frame_rate = frame_rate
        self.overlap_threshold = overlap_threshold  # dB difference for overlap detection
        
    def process_tracks(self, audio_files: List[str], output_prefix: str = "focused") -> List[str]:
        """Process multiple audio tracks to focus on loudest at each moment"""
        
        if len(audio_files) < 2:
            print("Need at least 2 audio tracks for focus processing")
            return []
        
        print(f"\n=== Processing {len(audio_files)} audio tracks ===")
        
        # Load all tracks
        tracks_data = []
        for i, audio_file in enumerate(audio_files):
            audio, sr = self.analyzer.load_audio(audio_file)
            if audio is not None:
                db_levels = self.analyzer.calculate_db_levels(audio)
                tracks_data.append({
                    'audio': audio,
                    'db_levels': db_levels,
                    'sample_rate': sr,
                    'track_num': i + 1,
                    'output_prefix': output_prefix
                })
                duration = len(audio) / sr if len(audio.shape) == 1 else audio.shape[1] / sr
                print(f"Loaded track {i+1}: {duration:.2f}s")
        
        if len(tracks_data) < 2:
            print("Failed to load sufficient tracks")
            return []
        
        # Determine focus decisions for each frame
        focus_decisions = self._calculate_focus_decisions(tracks_data)
        
        # Show focus summary
        total_frames = len(focus_decisions)
        track_active_counts = [0] * len(tracks_data)
        overlap_count = 0
        
        for decision in focus_decisions:
            if decision['is_overlap']:
                overlap_count += 1
            for track_idx in decision['active_tracks']:
                track_active_counts[track_idx] += 1
        
        print(f"\nFocus Summary:")
        for i, count in enumerate(track_active_counts):
            percentage = (count / total_frames) * 100
            print(f"Track {i+1}: {count}/{total_frames} frames ({percentage:.1f}%)")
        print(f"Overlap frames: {overlap_count}/{total_frames} ({(overlap_count/total_frames)*100:.1f}%)")
        
        # Debug: Show some sample decisions to understand what's happening
        print(f"\nSample Focus Decisions (first 10 frames):")
        for i in range(min(10, len(focus_decisions))):
            decision = focus_decisions[i]
            loudest = decision['loudest_track']
            active = decision['active_tracks']
            db_levels = [f"{db:.1f}" for db in decision['db_levels']]
            print(f"Frame {i}: Loudest=Track{loudest+1}, Active={[t+1 for t in active]}, dB=[{','.join(db_levels)}]")
        
        # Debug: Show transitions where focus changes
        print(f"\nFocus Transitions (first 20):")
        prev_loudest = -1
        transition_count = 0
        for i, decision in enumerate(focus_decisions):
            current_loudest = decision['loudest_track']
            if current_loudest != prev_loudest and i > 0:
                time_sec = decision['time']
                db_levels = [f"{db:.1f}" for db in decision['db_levels']]
                print(f"  {time_sec:.2f}s: Track{prev_loudest+1} -> Track{current_loudest+1}, dB=[{','.join(db_levels)}]")
                transition_count += 1
                if transition_count >= 20:
                    break
            prev_loudest = current_loudest
        
        # Apply focus to each track and save
        # IMPORTANT: Each output track contains ONLY its original input audio,
        # muted when that specific track is not the focused one
        output_files = []
        for i, track_data in enumerate(tracks_data):
            print(f"\nProcessing Track {i+1} (output will contain ONLY Track {i+1}'s original audio)")
            focused_result = self._apply_focus(track_data, focus_decisions, i)
            
            # Handle streaming vs in-memory result
            if isinstance(focused_result, str):
                # Streaming case - result is final filename
                output_file = focused_result
                
                # Calculate active percentage from file
                info = sf.info(output_file)
                print(f"Track {i+1}: Streaming write complete - {info.frames} samples, {info.duration:.2f}s")
                
            else:
                # In-memory case - result is audio array
                focused_audio = focused_result
                
                # Show active audio percentage
                total_samples = len(focused_audio) if len(focused_audio.shape) == 1 else focused_audio.shape[1]
                if len(focused_audio.shape) > 1:
                    non_zero_samples = np.count_nonzero(np.any(focused_audio != 0, axis=0))
                else:
                    non_zero_samples = np.count_nonzero(focused_audio)
                
                active_percentage = (non_zero_samples / total_samples) * 100
                print(f"Track {i+1}: {active_percentage:.1f}% active audio")
                
                # Save focused track
                output_file = f"{output_prefix}_track_{i+1}.wav"
                
                # Ensure proper format for soundfile
                # soundfile expects (samples, channels) format, but we have (channels, samples)
                if len(focused_audio.shape) > 1:
                    # Multi-channel - transpose to (samples, channels) format
                    focused_audio_out = focused_audio.T
                else:
                    # Mono
                    focused_audio_out = focused_audio
                
                sf.write(output_file, focused_audio_out, track_data['sample_rate'], subtype='PCM_16')
                print(f"Saved focused track {i+1} to {output_file}")
            
            output_files.append(output_file)
        
        # Generate focus report
        self._generate_focus_report(focus_decisions, tracks_data, f"{output_prefix}_report.txt")
        
        return output_files
    
    def _calculate_focus_decisions(self, tracks_data: List[Dict]) -> List[Dict]:
        """Calculate which track should be focused at each frame"""
        
        # Find the shortest track to avoid index errors
        min_frames = min(len(track['db_levels']) for track in tracks_data)
        
        focus_decisions = []
        
        for frame_idx in range(min_frames):
            frame_db_levels = [track['db_levels'][frame_idx] for track in tracks_data]
            max_db = max(frame_db_levels)
            loudest_track = frame_db_levels.index(max_db)
            
            # Check for overlaps (multiple tracks with similar levels)
            overlapping_tracks = []
            for i, db_level in enumerate(frame_db_levels):
                if abs(db_level - max_db) <= self.overlap_threshold and db_level > -30:
                    overlapping_tracks.append(i)
            
            # Decide if this is genuine overlap vs mic bleed
            is_genuine_overlap = self._detect_genuine_overlap(
                frame_idx, frame_db_levels, tracks_data, overlapping_tracks
            )
            
            decision = {
                'frame': frame_idx,
                'time': frame_idx / self.frame_rate,
                'db_levels': frame_db_levels.copy(),
                'loudest_track': loudest_track,
                'active_tracks': overlapping_tracks if is_genuine_overlap else [loudest_track],
                'is_overlap': is_genuine_overlap and len(overlapping_tracks) > 1
            }
            
            focus_decisions.append(decision)
        
        return focus_decisions
    
    def _detect_genuine_overlap(self, frame_idx: int, frame_db_levels: List[float], 
                               tracks_data: List[Dict], overlapping_tracks: List[int]) -> bool:
        """Detect if overlapping audio is genuine conversation vs mic bleed"""
        
        if len(overlapping_tracks) <= 1:
            return False
        
        # Simple heuristic: if multiple tracks are significantly above noise floor
        # and close in level, it's likely genuine overlap
        significant_tracks = [i for i, db in enumerate(frame_db_levels) 
                             if db > -25 and i in overlapping_tracks]
        
        # Additional check: look at recent history to see if it's a transition
        # vs sustained overlap (genuine conversation)
        window_size = min(10, frame_idx)  # Look back up to 10 frames
        if window_size > 3:
            recent_leaders = []
            for i in range(max(0, frame_idx - window_size), frame_idx):
                if i < len(tracks_data[0]['db_levels']):
                    recent_db = [track['db_levels'][i] for track in tracks_data]
                    recent_leaders.append(recent_db.index(max(recent_db)))
            
            # If leadership has been changing recently, more likely genuine overlap
            leadership_changes = len(set(recent_leaders))
            
            return len(significant_tracks) >= 2 and leadership_changes > 1
        
        return len(significant_tracks) >= 2
    
    def _apply_focus(self, track_data: Dict, focus_decisions: List[Dict], track_index: int) -> np.ndarray:
        """Apply focus decisions to mute/unmute audio track
        
        CRITICAL: This always processes the SAME track's audio (track_index)
        - audio = original audio from Track N (track_index)  
        - focus_decisions = when Track N should be active vs muted
        - output = Track N's audio, muted when Track N is not focused
        """
        
        audio = track_data['audio']  # Don't copy immediately to save memory
        sample_rate = track_data['sample_rate']
        
        print(f"   _apply_focus: Processing track_index={track_index} (Track {track_index+1})")
        print(f"   Input audio shape: {audio.shape}")
        
        # For very large arrays, avoid creating zeros_like which may hit memory limits
        if len(audio.shape) > 1:
            total_elements = audio.shape[0] * audio.shape[1]
            if total_elements > 2**28:  # Lower threshold to catch the issue earlier
                print(f"   Large array detected ({total_elements} elements), using direct file processing")
                # We need to get the output_prefix from the calling context
                # For now, we'll pass it through the track_data
                output_prefix = track_data.get('output_prefix', 'focused')
                return self._apply_focus_direct_write(audio, focus_decisions, track_index, sample_rate, output_prefix)
        
        # Standard processing for smaller arrays
        # Start with completely muted track
        if len(audio.shape) > 1:  # Multi-channel
            focused_audio = np.zeros_like(audio)
        else:  # Mono
            focused_audio = np.zeros_like(audio)
        
        # Convert frame decisions to sample-level decisions
        samples_per_frame = sample_rate // self.frame_rate
        
        # Calculate samples per frame for processing
        # Note: track_index is 0-based here, but displayed as 1-based elsewhere
        
        active_frame_count = 0
        total_frame_count = len(focus_decisions)
        
        for decision in focus_decisions:
            start_sample = decision['frame'] * samples_per_frame
            end_sample = min(start_sample + samples_per_frame, 
                           len(audio) if len(audio.shape) == 1 else audio.shape[1])
            
            if track_index in decision['active_tracks']:
                # This track is focused - copy THIS track's original audio
                # IMPORTANT: We always copy from the SAME track's audio, never cross-contaminate
                fade_samples = min(100, (end_sample - start_sample) // 4)  # Quick fade
                active_frame_count += 1
                
                # Debug for first few frames
                if decision['frame'] < 5:
                    print(f"      Frame {decision['frame']}: Track {track_index+1} is ACTIVE, copying audio")
                
                if len(audio.shape) > 1:  # Stereo (channels, samples)
                    # Copy the audio segment from THIS track only
                    focused_audio[:, start_sample:end_sample] = audio[:, start_sample:end_sample]
                    
                    # Apply fade in if this is a transition
                    if fade_samples > 0 and start_sample > 0:
                        # Check if previous frame was muted
                        prev_frame = max(0, decision['frame'] - 1)
                        if prev_frame < len(focus_decisions):
                            prev_decision = focus_decisions[prev_frame]
                            if track_index not in prev_decision['active_tracks']:
                                # Was muted before, apply fade in
                                fade_in = np.linspace(0.0, 1.0, fade_samples)
                                for ch in range(audio.shape[0]):
                                    if start_sample + fade_samples <= focused_audio.shape[1]:
                                        focused_audio[ch, start_sample:start_sample + fade_samples] *= fade_in
                else:  # Mono
                    # Copy the audio segment
                    focused_audio[start_sample:end_sample] = audio[start_sample:end_sample]
                    
                    # Apply fade in if this is a transition
                    if fade_samples > 0 and start_sample > 0:
                        # Check if previous frame was muted
                        prev_frame = max(0, decision['frame'] - 1)
                        if prev_frame < len(focus_decisions):
                            prev_decision = focus_decisions[prev_frame]
                            if track_index not in prev_decision['active_tracks']:
                                # Was muted before, apply fade in
                                fade_in = np.linspace(0.0, 1.0, fade_samples)
                                if start_sample + fade_samples <= len(focused_audio):
                                    focused_audio[start_sample:start_sample + fade_samples] *= fade_in
            
            # If not in active tracks, segment remains muted (zeros)
            else:
                # Debug for first few frames  
                if decision['frame'] < 5:
                    print(f"      Frame {decision['frame']}: Track {track_index+1} is MUTED")
        
        # Debug: Check frame count consistency
        expected_active_frames = sum(1 for d in focus_decisions if track_index in d['active_tracks'])
        print(f"   Track {track_index+1}: Applied {active_frame_count} active frames, expected {expected_active_frames}")
        frame_percentage = (active_frame_count / total_frame_count) * 100
        print(f"   Track {track_index+1}: {frame_percentage:.1f}% active frames")
        
        # Debug: Check samples calculation
        audio_samples = len(audio) if len(audio.shape) == 1 else audio.shape[1]
        calculated_samples = total_frame_count * samples_per_frame
        print(f"   Audio samples: {audio_samples}, Frame samples: {calculated_samples}")
        print(f"   Samples per frame: {samples_per_frame}, Total decisions: {total_frame_count}")
        
        return focused_audio
    
    def _apply_focus_chunked(self, audio: np.ndarray, focus_decisions: List[Dict], track_index: int, sample_rate: int) -> np.ndarray:
        """Memory-efficient chunked processing for large audio files"""
        
        print(f"   Using chunked processing for track {track_index+1}")
        
        # Process in chunks to avoid memory issues
        chunk_size_frames = 10000  # Process 10k frames at a time
        samples_per_frame = sample_rate // self.frame_rate
        chunk_size_samples = chunk_size_frames * samples_per_frame
        
        # Initialize output array with the same shape as input
        focused_audio = np.zeros_like(audio)
        
        total_frames = len(focus_decisions)
        frames_processed = 0
        
        for chunk_start_frame in range(0, total_frames, chunk_size_frames):
            chunk_end_frame = min(chunk_start_frame + chunk_size_frames, total_frames)
            
            # Calculate sample range for this chunk
            start_sample = chunk_start_frame * samples_per_frame
            end_sample = min(chunk_end_frame * samples_per_frame, audio.shape[1])
            
            print(f"   Processing chunk: frames {chunk_start_frame}-{chunk_end_frame}, samples {start_sample}-{end_sample}")
            
            # Process decisions for this chunk
            for frame_idx in range(chunk_start_frame, chunk_end_frame):
                if frame_idx >= len(focus_decisions):
                    break
                    
                decision = focus_decisions[frame_idx]
                frame_start_sample = decision['frame'] * samples_per_frame
                frame_end_sample = min(frame_start_sample + samples_per_frame, audio.shape[1])
                
                if track_index in decision['active_tracks']:
                    # Copy audio for this frame
                    focused_audio[:, frame_start_sample:frame_end_sample] = audio[:, frame_start_sample:frame_end_sample]
            
            frames_processed = chunk_end_frame
            if frames_processed % 50000 == 0:
                print(f"   Processed {frames_processed}/{total_frames} frames ({frames_processed/total_frames*100:.1f}%)")
        
        print(f"   Chunked processing complete: {frames_processed} frames processed")
        return focused_audio
    
    def _apply_focus_direct_write(self, audio: np.ndarray, focus_decisions: List[Dict], track_index: int, sample_rate: int, output_prefix: str) -> str:
        """Stream directly to final output file using ffmpeg for large WAV files"""
        
        print(f"   Using streaming write for track {track_index+1} to avoid memory limits")
        
        samples_per_frame = sample_rate // self.frame_rate
        total_frames = len(focus_decisions)
        total_samples = audio.shape[1] if len(audio.shape) > 1 else len(audio)
        
        # For large files, use scipy.io.wavfile to avoid soundfile WAV limitations
        if total_samples > 536870000:
            print(f"   Large file detected - using scipy.io.wavfile for WAV output")
            return self._write_large_wav_scipy(audio, focus_decisions, track_index, sample_rate, output_prefix)
        
        # Standard soundfile approach for smaller files
        final_output_file = f"{output_prefix}_track_{track_index+1}.wav"
        channels = audio.shape[0] if len(audio.shape) > 1 else 1
        
        with sf.SoundFile(final_output_file, 'w', samplerate=sample_rate, 
                         channels=channels, subtype='PCM_16') as output_file:
            
            # Process in chunks and write directly to file
            chunk_size_frames = 5000  # Process 5k frames at a time
            
            for chunk_start_frame in range(0, total_frames, chunk_size_frames):
                chunk_end_frame = min(chunk_start_frame + chunk_size_frames, total_frames)
                
                # Calculate sample range for this chunk
                chunk_start_sample = chunk_start_frame * samples_per_frame
                chunk_end_sample = min(chunk_end_frame * samples_per_frame, audio.shape[1])
                chunk_length = chunk_end_sample - chunk_start_sample
                
                # Create chunk buffer (much smaller)
                if len(audio.shape) > 1:
                    chunk_buffer = np.zeros((audio.shape[0], chunk_length), dtype=audio.dtype)
                else:
                    chunk_buffer = np.zeros(chunk_length, dtype=audio.dtype)
                
                # Fill in active parts of this chunk
                for frame_idx in range(chunk_start_frame, chunk_end_frame):
                    if frame_idx >= len(focus_decisions):
                        break
                        
                    decision = focus_decisions[frame_idx]
                    if track_index in decision['active_tracks']:
                        # Calculate sample positions within this chunk
                        frame_start_sample = decision['frame'] * samples_per_frame
                        frame_end_sample = min(frame_start_sample + samples_per_frame, audio.shape[1])
                        
                        # Convert to chunk-relative positions
                        chunk_rel_start = frame_start_sample - chunk_start_sample
                        chunk_rel_end = frame_end_sample - chunk_start_sample
                        
                        # Copy audio data to buffer
                        if len(audio.shape) > 1:
                            chunk_buffer[:, chunk_rel_start:chunk_rel_end] = audio[:, frame_start_sample:frame_end_sample]
                        else:
                            chunk_buffer[chunk_rel_start:chunk_rel_end] = audio[frame_start_sample:frame_end_sample]
                
                # Write chunk to file (transpose if multi-channel for soundfile format)
                if len(audio.shape) > 1:
                    output_file.write(chunk_buffer.T)  # soundfile expects (samples, channels)
                else:
                    output_file.write(chunk_buffer)
                
                if chunk_end_frame % 25000 == 0:
                    print(f"   Streamed {chunk_end_frame}/{total_frames} frames ({chunk_end_frame/total_frames*100:.1f}%)")
        
        print(f"   Streaming write complete: {final_output_file}")
        return final_output_file
    
    def _write_large_wav_scipy(self, audio: np.ndarray, focus_decisions: List[Dict], track_index: int, sample_rate: int, output_prefix: str) -> str:
        """Write large WAV files using scipy.io.wavfile (no size limitations)"""
        
        print(f"   Using scipy.io.wavfile for large track {track_index+1}")
        
        samples_per_frame = sample_rate // self.frame_rate
        total_frames = len(focus_decisions)
        final_output_file = f"{output_prefix}_track_{track_index+1}.wav"
        
        # Build the focused audio array in manageable chunks
        output_chunks = []
        chunk_size_frames = 5000  # Process 5k frames at a time
        
        for chunk_start_frame in range(0, total_frames, chunk_size_frames):
            chunk_end_frame = min(chunk_start_frame + chunk_size_frames, total_frames)
            
            # Calculate sample range for this chunk
            chunk_start_sample = chunk_start_frame * samples_per_frame
            chunk_end_sample = min(chunk_end_frame * samples_per_frame, audio.shape[1])
            chunk_length = chunk_end_sample - chunk_start_sample
            
            # Create chunk buffer
            if len(audio.shape) > 1:
                chunk_buffer = np.zeros((audio.shape[0], chunk_length), dtype=audio.dtype)
            else:
                chunk_buffer = np.zeros(chunk_length, dtype=audio.dtype)
            
            # Fill in active parts of this chunk
            for frame_idx in range(chunk_start_frame, chunk_end_frame):
                if frame_idx >= len(focus_decisions):
                    break
                    
                decision = focus_decisions[frame_idx]
                if track_index in decision['active_tracks']:
                    # Calculate sample positions within this chunk
                    frame_start_sample = decision['frame'] * samples_per_frame
                    frame_end_sample = min(frame_start_sample + samples_per_frame, audio.shape[1])
                    
                    # Convert to chunk-relative positions
                    chunk_rel_start = frame_start_sample - chunk_start_sample
                    chunk_rel_end = frame_end_sample - chunk_start_sample
                    
                    # Copy audio data to buffer
                    if len(audio.shape) > 1:
                        chunk_buffer[:, chunk_rel_start:chunk_rel_end] = audio[:, frame_start_sample:frame_end_sample]
                    else:
                        chunk_buffer[chunk_rel_start:chunk_rel_end] = audio[frame_start_sample:frame_end_sample]
            
            output_chunks.append(chunk_buffer)
            
            if chunk_end_frame % 25000 == 0:
                print(f"   Processed {chunk_end_frame}/{total_frames} frames ({chunk_end_frame/total_frames*100:.1f}%)")
        
        # Concatenate all chunks
        print(f"   Concatenating {len(output_chunks)} chunks for scipy write...")
        if len(audio.shape) > 1:
            focused_audio = np.concatenate(output_chunks, axis=1)
            # scipy expects (samples, channels) format
            focused_audio_out = focused_audio.T
        else:
            focused_audio = np.concatenate(output_chunks)
            focused_audio_out = focused_audio
        
        # Convert to int16 for WAV compatibility
        if focused_audio_out.dtype != np.int16:
            # Normalize and convert to int16
            if focused_audio_out.dtype == np.float32 or focused_audio_out.dtype == np.float64:
                # Assume float data is in [-1, 1] range
                focused_audio_out = (focused_audio_out * 32767).astype(np.int16)
            else:
                focused_audio_out = focused_audio_out.astype(np.int16)
        
        # Write using scipy.io.wavfile
        print(f"   Writing large WAV file with scipy: {focused_audio_out.shape}")
        wavfile.write(final_output_file, sample_rate, focused_audio_out)
        
        print(f"   Scipy write complete: {final_output_file}")
        return final_output_file
    
    def _generate_focus_report(self, focus_decisions: List[Dict], tracks_data: List[Dict], filename: str):
        """Generate a report of focus decisions for validation"""
        
        with open(filename, 'w') as f:
            f.write("Audio Focus Report\n")
            f.write("==================\n\n")
            
            f.write(f"Processed {len(tracks_data)} tracks\n")
            f.write(f"Total frames: {len(focus_decisions)}\n")
            f.write(f"Frame rate: {self.frame_rate} fps\n")
            f.write(f"Duration: {len(focus_decisions) / self.frame_rate:.2f} seconds\n\n")
            
            # Summary statistics
            track_focus_time = [0] * len(tracks_data)
            overlap_count = 0
            
            for decision in focus_decisions:
                if decision['is_overlap']:
                    overlap_count += 1
                for track_idx in decision['active_tracks']:
                    track_focus_time[track_idx] += 1
            
            f.write("Focus Time Summary:\n")
            for i, focus_time in enumerate(track_focus_time):
                percentage = (focus_time / len(focus_decisions)) * 100
                f.write(f"Track {i+1}: {focus_time} frames ({percentage:.1f}%)\n")
            
            f.write(f"\nOverlap frames: {overlap_count} ({(overlap_count/len(focus_decisions)*100):.1f}%)\n\n")
            
            # Detailed timeline (sample every 5 seconds for readability)
            f.write("Detailed Timeline (every 5 seconds):\n")
            f.write("Time\tActive Track(s)\tdB Levels\tNotes\n")
            
            for i, decision in enumerate(focus_decisions):
                if i % (self.frame_rate * 5) == 0:  # Every 5 seconds
                    time_str = f"{decision['time']:.1f}s"
                    if decision['is_overlap']:
                        active_str = f"Tracks {','.join(map(str, [t+1 for t in decision['active_tracks']]))}"
                        notes = "OVERLAP"
                    else:
                        active_str = f"Track {decision['loudest_track']+1}"
                        notes = "FOCUSED"
                    
                    db_str = " ".join([f"{db:.1f}" for db in decision['db_levels']])
                    f.write(f"{time_str}\t{active_str}\t{db_str}\t{notes}\n")
        
        print(f"Focus report saved to {filename}")

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description="Audio Focus System - Focus on loudest track at each moment")
    
    parser.add_argument('inputs', nargs='+', 
                       help='Input files: either video file (extracts all audio tracks) or multiple audio files')
    parser.add_argument('--output', '-o', default='focused',
                       help='Output prefix for generated files (default: focused)')
    parser.add_argument('--frame-rate', '-f', type=int, default=30,
                       help='Frame rate for analysis (default: 30)')
    parser.add_argument('--overlap-threshold', '-t', type=float, default=5.0,
                       help='dB threshold for overlap detection (default: 5.0)')
    parser.add_argument('--audio-only', '-a', action='store_true',
                       help='Output audio files only, even from video input')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze input, don\'t process (for testing)')
    
    args = parser.parse_args()
    
    print("=== Audio Focus System ===")
    print(f"Frame rate: {args.frame_rate} fps")
    print(f"Overlap threshold: {args.overlap_threshold} dB")
    
    # Initialize components
    analyzer = AudioAnalyzer(args.frame_rate)
    processor = AudioFocusProcessor(args.frame_rate, args.overlap_threshold)
    
    try:
        # Determine input type and extract audio tracks
        if len(args.inputs) == 1 and args.inputs[0].lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            # Single video file - extract all audio tracks
            video_file = args.inputs[0]
            print(f"Processing video file: {video_file}")
            
            if args.analyze_only:
                analyzer.analyze_sample_file(video_file)
                return
            
            audio_files = analyzer.extract_video_audio_tracks(video_file)
            if not audio_files:
                print("Failed to extract audio tracks from video")
                return
                
            is_video_input = True
            
        else:
            # Multiple audio files
            audio_files = args.inputs
            print(f"Processing {len(audio_files)} audio files")
            
            # Verify all files exist
            for audio_file in audio_files:
                if not os.path.exists(audio_file):
                    print(f"Error: Audio file not found: {audio_file}")
                    return
            
            is_video_input = False
        
        # Process the audio tracks
        output_files = processor.process_tracks(audio_files, args.output)
        
        if output_files:
            print(f"\n=== Processing Complete ===")
            print(f"Generated {len(output_files)} focused audio tracks:")
            for output_file in output_files:
                print(f"  - {output_file}")
            
            print(f"Focus report: {args.output}_report.txt")
            
            # If input was video and user wants video output (not audio-only)
            if is_video_input and not args.audio_only:
                print("\n=== Creating Focused Video ===")
                video_output = create_focused_video(args.inputs[0], output_files, args.output)
                if video_output:
                    print(f"Focused video saved as: {video_output}")
        
        # Clean up temporary files
        if is_video_input:
            for temp_file in audio_files:
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def create_focused_video(input_video: str, focused_audio_files: List[str], output_prefix: str) -> Optional[str]:
    """Create a video with focused audio tracks"""
    
    try:
        output_video = f"{output_prefix}_video.mp4"
        
        print(f"Creating video with {len(focused_audio_files)} focused audio tracks...")
        
        # Build ffmpeg command more explicitly
        inputs = [ffmpeg.input(input_video)]
        
        # Add each focused audio file as input
        for audio_file in focused_audio_files:
            inputs.append(ffmpeg.input(audio_file))
        
        # Create output with explicit mapping
        # Map video from input 0, and audio tracks from subsequent inputs
        args = {
            'vcodec': 'copy',  # Copy video without re-encoding
            'acodec': 'aac',   # Re-encode audio as AAC
            'audio_bitrate': '320k',
            'ar': 48000,       # Set audio sample rate
            'ac': 2            # Set to stereo
        }
        
        # Map streams explicitly
        stream_spec = [inputs[0]['v:0']]  # Video from first input
        
        # Add audio streams from focused audio files
        for i in range(len(focused_audio_files)):
            stream_spec.append(inputs[i + 1]['a:0'])
        
        # Create output
        output = ffmpeg.output(*stream_spec, output_video, **args)
        
        # Run ffmpeg with verbose output for debugging
        try:
            ffmpeg.run(output, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            print(f"Successfully created {output_video}")
            return output_video
            
        except ffmpeg.Error as e:
            print(f"FFmpeg error creating video:")
            print(f"stdout: {e.stdout.decode()}")
            print(f"stderr: {e.stderr.decode()}")
            
            # Try alternative approach with simpler command
            print("Trying alternative ffmpeg approach...")
            return create_focused_video_alternative(input_video, focused_audio_files, output_prefix)
        
    except Exception as e:
        print(f"Error creating focused video: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_focused_video_alternative(input_video: str, focused_audio_files: List[str], output_prefix: str) -> Optional[str]:
    """Alternative video creation method using direct ffmpeg command"""
    
    try:
        output_video = f"{output_prefix}_video.mp4"
        
        # Build command manually for more control
        cmd = ['ffmpeg', '-y']  # -y to overwrite output
        
        # Add input video
        cmd.extend(['-i', input_video])
        
        # Add input audio files
        for audio_file in focused_audio_files:
            cmd.extend(['-i', audio_file])
        
        # Map video (copy without re-encoding)
        cmd.extend(['-map', '0:v:0', '-c:v', 'copy'])
        
        # Map and re-encode audio tracks
        for i in range(len(focused_audio_files)):
            cmd.extend(['-map', f'{i+1}:a:0'])
        
        # Audio encoding settings
        cmd.extend([
            '-c:a', 'aac',
            '-b:a', '320k',
            '-ar', '48000',
            '-ac', '2'
        ])
        
        # Output file
        cmd.append(output_video)
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run command
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully created {output_video}")
            return output_video
        else:
            print(f"FFmpeg command failed:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"Error in alternative video creation: {e}")
        return None


if __name__ == "__main__":
    main()
