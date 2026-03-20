# YT Insight — AI Video Chat

A Streamlit application that enables intelligent conversations with YouTube video content using AI-powered transcription, summarization, and semantic search.

## Features

- 🎥 **Video Transcription**: Automatically fetch transcripts from YouTube videos or transcribe audio using OpenAI's Whisper
- 🤖 **AI-Powered Chat**: Ask questions about the video content and get intelligent responses powered by Google's Generative AI
- 🔍 **Semantic Search**: Find relevant sections of the video using advanced vector search (FAISS)
- 🌍 **Multi-Language Support**: Automatically detects and handles multiple languages
- 💾 **Vector Database**: Efficiently stores and retrieves video embeddings for fast semantic matching

## Tech Stack

- **Frontend**: Streamlit
- **AI Models**: 
  - Google Generative AI (Gemini)
  - OpenAI Whisper
  - Sentence Transformers
- **Vector Search**: FAISS
- **Video Processing**: yt-dlp
- **Transcription**: YouTube Transcript API
- **Language Detection**: langdetect

## Requirements

- Python 3.8+
- FFmpeg (required for Whisper audio processing)
- API Keys:
  - Google Generative AI (Gemini)

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Youtube_Transcriber
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install System Dependencies
- **FFmpeg**: Required for audio processing
  - Windows: Download from https://ffmpeg.org/download.html
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt-get install ffmpeg`

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_generative_ai_api_key_here
```

### Getting API Keys

**Google Generative AI:**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file

## Usage

### Run the Application
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### How to Use

1. **Input a YouTube URL**: Paste a YouTube video link in the sidebar
2. **Choose Transcript Source**:
   - Use official YouTube transcript (if available)
   - Or transcribe audio using Whisper
3. **Ask Questions**: Type questions about the video content in the chat
4. **Receive Answers**: Get AI-generated responses based on the video content

## How It Works

1. **Transcript Extraction**: Fetches or generates video transcript
2. **Text Chunking**: Splits transcript into manageable segments
3. **Embedding**: Converts text chunks into vector embeddings using Sentence Transformers
4. **Vector Storage**: Stores embeddings in a FAISS index for fast retrieval
5. **Semantic Search**: Finds relevant chunks based on your questions
6. **AI Response**: Generates contextual answers using Google Generative AI

## Dependencies

See `requirements.txt` for complete list:
- streamlit
- youtube-transcript-api
- google-generativeai
- openai-whisper
- yt-dlp
- sentence-transformers
- faiss-cpu
- numpy
- langdetect
- python-dotenv

## Project Structure

```
Youtube_Transcriber/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── packages.txt          # System dependencies (for deployment)
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Troubleshooting

### FFmpeg Not Found
- Ensure FFmpeg is installed and in your system PATH
- Restart your terminal after installing FFmpeg

### API Key Errors
- Verify your `.env` file is in the project root
- Check that `GOOGLE_API_KEY` is set correctly
- Ensure the API key has the necessary permissions

### Import Errors
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
- Verify you're using the correct Python virtual environment

## Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add secrets in Settings:
   - `GOOGLE_API_KEY`: Your Google API key
4. Deploy!

## Future Enhancements

- [ ] Support for multiple video URLs in one session
- [ ] Video playback with timestamp highlights
- [ ] Video summarization
- [ ] Export chat history
- [ ] Support for playlist processing
- [ ] Custom vector database selection
- [ ] Advanced filtering and search options

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on GitHub.

---

**Built with ❤️ using Streamlit**
