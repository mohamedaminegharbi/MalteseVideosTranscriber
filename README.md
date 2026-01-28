# Maltese Video Transcriber

A Jupyter notebook that transcribes Maltese-language YouTube videos using OpenAI's Whisper model, optimized to prevent repetition issues common in low-resource languages.

## Features

- **Maltese-specific transcription** using Whisper's large-v3 model
- **Anti-repetition settings** to prevent common looping issues
- **Multiple output formats**: plain text transcript and SRT subtitles
- **Timestamped segments** for easy navigation
- **Kaggle-optimized** for use with GPU acceleration

## What It Does

This notebook downloads audio from YouTube videos and transcribes them to Maltese text. It produces:

1. **Full transcript** (`maltese_transcript.txt`) - Complete transcription with video metadata and timestamped segments
2. **SRT subtitles** (`maltese_subtitles.srt`) - Standard subtitle format for video players

## Installation

The notebook automatically installs required packages:
- `openai-whisper` - Whisper transcription model
- `yt-dlp` - YouTube video/audio downloader

## OpenAI-Whisper Aechitecture

1. **Input Processing**
 
- `Log-mel spectrogram` : The audio is converted into a visual representation showing frequency content over time 
- `2x Conv1D + GELU`: Two 1D convolutional layers with GELU activation functions process the spectrogram
- `Sinusoidal Positional Encoding` : Added to give the model information about the temporal position of audio features

2. **Encoder**
   
- `Multiple Encoder Blocks` : A stack of identical Transformer encoder blocks. These process the audio features and create rich contextual representations. Each block contains self-attention mechanisms and feed-forward networks

3. **Cross-attention**
   
The decoder attends to the encoder's output. This allows the decoder to focus on relevant parts of the audio while generating text

4. **Decoder**
   
Multiple Decoder Blocks: A stack of Transformer decoder blocks

5. **Learned Positional Encoding**

Different from the encoder, uses learned position embeddings

5. **Input tokens**

Special tokens in multitask training format including:
- `<|SOT|> (Start of Transcript)`
- `Language Identifier`
- `Task Identifier`
- `Timestamp`
- `Transcription tokens`
  
6. **Output**

- `Next-token prediction` : The model predicts tokens autoregressively

![asr-summary-of-model-architecture-desktop](https://github.com/user-attachments/assets/bfceab10-090f-4a34-a9bc-61a405e93bca)

**Key Design Features**

- `Multitask format` : The special tokens allow Whisper to handle multiple tasks (transcription, translation, language detection) with a single model
- `Autoregressive generation` : The decoder generates one token at a time, using previously generated tokens
- `Encoder-decoder architecture` : Separates audio understanding (encoder) from text generation (decoder)
- `Audio chunks` : The log-mel spectrogram represents fixed-length audio segments
  
This architecture enables Whisper to perform robust speech recognition across multiple languages and handle various tasks through its flexible token-based task specification system.

## Usage

### Quick Start

1. Upload `transcribe-maltese.ipynb` to Kaggle
2. Enable GPU acceleration (Settings → Accelerator → GPU T4 x2)
3. Enable internet access (Settings → Internet → On)
4. Update the `YOUTUBE_URL` variable with your video URL
5. Run all cells

### Configuration Options

```python
# Your YouTube video URL
YOUTUBE_URL = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# Model size (affects accuracy and speed)
MODEL_SIZE = "large-v3"  # Recommended for Maltese
# Other options: "small" (faster), "medium" (balanced)
```

### Model Sizes

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| `small` | Fast | Lower | Testing, quick drafts |
| `medium` | Moderate | Good | Balanced use cases |
| `large-v3` | Slower | Best | Final transcriptions (recommended) |

## Technical Details

### Anti-Repetition Settings

The notebook includes specialized settings to prevent text repetition, a common issue when transcribing low-resource languages:

- `condition_on_previous_text=False` - Prevents repetition loops
- `beam_size=5` - Better accuracy with multiple candidates
- `temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)` - Multiple temperatures to avoid getting stuck
- `compression_ratio_threshold=2.4` - Reduces repetitive output
- `logprob_threshold=-1.0` - Stricter quality filtering
- `word_timestamps=True` - Enables word-level timing

### Processing Time

Transcription time depends on video length and model size:
- ~15 minute video with `large-v3`: approximately 12-15 minutes
- Processing rate: ~130 frames/second on Tesla T4

## Output Format

### maltese_transcript.txt
```
VIDEO: [Video Title]
URL: [YouTube URL]
======================================================================

[Full transcription text]

======================================================================
TIMESTAMPED SEGMENTS:
======================================================================
[00:00 - 00:05] First segment text
[00:05 - 00:10] Second segment text
...
```

### maltese_subtitles.srt
```
1
00:00:00,000 --> 00:00:05,000
First segment text

2
00:00:05,000 --> 00:00:10,000
Second segment text
...
```

## Limitations

- Requires GPU for reasonable processing speed (CPU transcription is very slow)
- Large-v3 model requires ~3GB VRAM download
- Only works with publicly accessible YouTube videos
- Accuracy depends on audio quality and accent clarity

## Troubleshooting

**"No JavaScript runtime found" warning**
- This is normal and doesn't affect functionality. The notebook uses yt-dlp's default extraction methods.

**Out of memory errors**
- Try using a smaller model (`medium` or `small`)
- Ensure GPU is enabled in Kaggle settings

**Poor transcription quality**
- Check audio quality of the source video
- Try increasing model size to `large-v3`
- Ensure the video is primarily in Maltese (mixed-language videos may have issues)

**Video download fails**
- Verify the YouTube URL is correct and accessible
- Check that internet is enabled in Kaggle settings
- Some videos may be restricted or unavailable

## Example Use Case

This notebook was developed to transcribe content from Maltese cultural and historical videos, such as documentaries about the National Archives of Malta. It's particularly useful for:

- Creating accessible transcripts of Maltese media
- Generating subtitles for Maltese videos
- Preserving Maltese language content
- Research requiring Maltese text analysis

## Credits

- **Whisper**: OpenAI's automatic speech recognition system
- **yt-dlp**: Command-line YouTube downloader
- Optimized for Maltese language transcription based on community feedback


## Contributing

Suggestions for improvement are welcome, especially:
- Additional anti-repetition techniques
- Better Maltese-specific prompts
- Performance optimizations
- Support for other Maltese audio sources


