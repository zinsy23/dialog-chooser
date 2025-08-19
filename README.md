# Audio Focus System

An intelligent audio processing tool that automatically focuses on the loudest speaker at any given moment, eliminating microphone bleed and creating clean, focused audio tracks. No need to dial in on thresholds as it takes the loudest audio track and focuses that. It has some level of detecting when there's overlap from multiple speakers, which aren't microphone bleed related, and tries keeping those in.

## Overview

This system analyzes multiple audio tracks (from video files or separate audio files) and automatically mutes all but the loudest track at each moment in time. It features intelligent overlap detection to preserve genuine conversation interruptions while eliminating microphone bleed.

## Features

- **Automatic Audio Focusing**: Keeps only the loudest track active at each moment
- **Intelligent Overlap Detection**: Distinguishes between genuine conversation overlap and mic bleed
- **Video Processing**: Extract audio tracks from video files and create focused video output
- **Smooth Transitions**: Implements fade-out effects for clean audio switching
- **Comprehensive Reporting**: Detailed analysis and timeline of focus decisions
- **Flexible I/O**: Support for both video and audio file inputs with multiple output options

## Installation

1. Install required dependencies:
```bash
pip install ffmpeg-python librosa numpy scipy matplotlib soundfile
```

2. Ensure FFmpeg is installed on your system:
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg` (Ubuntu/Debian)

## Usage

### Video File Processing
Process a video file with multiple audio tracks:
```bash
python audio_focus.py Dialog.mp4
```

This will:
- Extract all audio tracks from the video
- Create focused audio tracks
- Generate a new video with focused audio
- Create a detailed focus report

### Audio-Only Output
To get only the focused audio tracks (no video):
```bash
python audio_focus.py Dialog.mp4 --audio-only
```

### Multiple Audio Files
Process separate audio files:
```bash
python audio_focus.py track1.wav track2.wav track3.wav
```

### Advanced Options
```bash
python audio_focus.py Dialog.mp4 \
    --output my_focused \
    --frame-rate 30 \
    --overlap-threshold 3.0 \
    --audio-only
```

## Command Line Options

- `inputs`: Video file (extracts all audio tracks) or multiple audio files
- `--output, -o`: Output prefix for generated files (default: "focused")
- `--frame-rate, -f`: Frame rate for analysis in fps (default: 30)
- `--overlap-threshold, -t`: dB threshold for overlap detection (default: 5.0)
- `--audio-only, -a`: Output audio files only, even from video input
- `--analyze-only`: Only analyze input without processing (for testing)

## Output Files

The system generates several output files:

1. **Focused Audio Tracks**: `{prefix}_track_1.wav`, `{prefix}_track_2.wav`, etc.
2. **Focus Report**: `{prefix}_report.txt` - Detailed analysis of focus decisions
3. **Focused Video**: `{prefix}_video.mp4` - Video with focused audio tracks (if input was video)

## How It Works

### Frame-by-Frame Analysis
The system analyzes audio in frames (default 30fps) to determine the loudest track at each moment.

### Focus Algorithm
1. **dB Level Calculation**: Computes RMS energy and converts to dB for each track
2. **Loudest Track Selection**: Identifies the track with the highest dB level
3. **Overlap Detection**: Uses intelligent heuristics to detect genuine conversation vs mic bleed
4. **Audio Muting**: Applies focus decisions with smooth fade transitions

### Overlap Detection
The system distinguishes between:
- **Microphone Bleed**: Background noise from other mics (muted)
- **Genuine Overlap**: Actual conversation interruptions (preserved)

Factors considered:
- Signal strength above noise floor
- Recent leadership changes between tracks
- dB level proximity between tracks

## Example Results

Using the provided Dialog.mp4 sample:
- **Duration**: 104.53 seconds
- **Track 1 Focus Time**: 48.2% (first speaker)
- **Track 2 Focus Time**: 52.8% (second speaker)
- **Overlap Frames**: 0.9% (genuine conversation overlap preserved)

The system correctly identified:
- Track 1 speaking in first ~8 seconds
- Track 2 blip around 8.2 seconds
- Track 2 dominant from ~8.3s to 60s
- Track 1 return after 1 minute
- Minimal overlap preservation for natural conversation flow

## Configuration

### Frame Rate
Higher frame rates provide more granular control but increase processing time:
- 30fps: Good balance for most use cases
- 60fps: Higher precision for fast speech transitions
- 15fps: Faster processing for longer content

### Overlap Threshold
Controls sensitivity to overlapping speech:
- Lower values (2-3 dB): More sensitive, preserves more overlaps
- Higher values (5-7 dB): Less sensitive, focuses more aggressively
- Default 5.0 dB works well for most podcast/interview scenarios

## Technical Details

- **Audio Analysis**: Uses librosa for robust audio processing
- **Video Processing**: Leverages FFmpeg for video/audio extraction and encoding
- **Format Support**: MP4, MOV, AVI, MKV (video), WAV, MP3, FLAC (audio)
- **Output Quality**: 16-bit PCM WAV for audio, AAC 320kbps for video

## Limitations

- Requires at least 2 audio tracks for processing
- Performance depends on track separation quality
- Overlap detection heuristics may need tuning for specific content types
- Processing time increases with file duration and frame rate

## Troubleshooting

### Common Issues

1. **"Format not recognised" error**: Ensure FFmpeg is properly installed and in PATH
2. **Short duration or 0.00s**: Check input file integrity and audio track presence
3. **Poor focus decisions**: Adjust overlap threshold or frame rate settings

### Debug Mode
Use `--analyze-only` to inspect the system's analysis without processing:
```bash
python audio_focus.py Dialog.mp4 --analyze-only
```

## Contributing

This is a focused tool designed for podcast and interview post-production. Contributions welcome for:
- Additional audio format support
- Enhanced overlap detection algorithms
- Performance optimizations
- GUI interface development

## License

Open source - feel free to use and modify for your projects.
