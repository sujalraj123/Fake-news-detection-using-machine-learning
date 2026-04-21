import os
import re
import json
import anthropic
import requests
from flask import Flask, request, jsonify, render_template

# Setup Flask
app = Flask(__name__)

# Use an environment variable for the API key to keep it secure
CLAUDE_KEY = os.environ.get("ANTHROPIC_API_KEY")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
ai_client = anthropic.Anthropic(api_key=CLAUDE_KEY)

# --- ANALYSIS PROMPTS ---
# These act as instructions for the AI to ensure we get a consistent JSON format back

TEXT_PROMPT = """You are a senior media analyst. Look at the news content provided and judge its credibility.
You must return a JSON object only.
Required fields: prediction (Real/Fake), confidence (0-100), reasoning, red_flags (list), credibility_signals (list), fake_probability, and real_probability.
Keep probabilities balanced (Real + Fake = 100%)."""

VIDEO_PROMPT = """You are a video forensics expert. Analyze the YouTube metadata and transcript provided.
Determine if the content is clickbait, misleading, or factual.
Return a JSON object with: prediction, confidence, reasoning, red_flags, credibility_signals, video_specific_flags, fake_probability, and real_probability."""

# --- HELPER FUNCTIONS ---

def get_id_from_url(url):
    """
    Tries to grab the 11-char YouTube ID using common URL patterns.
    """
    regex_patterns = [
        r'v=([0-9A-Za-z_-]{11})',
        r'youtu\.be/([0-9A-Za-z_-]{11})',
        r'shorts/([0-9A-Za-z_-]{11})',
        r'embed/([0-9A-Za-z_-]{11})'
    ]
    
    for p in regex_patterns:
        match = re.search(p, url)
        if match:
            return match.group(1)
    return None

def scrape_video_info(vid_id):
    """
    Uses yt-dlp to get the title, channel, and stats without an official API key.
    """
    import yt_dlp
    opts = {'quiet': True, 'skip_download': True}
    
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            raw_info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid_id}", download=False)
            return {
                "title": raw_info.get("title", "N/A"),
                "channel": raw_info.get("uploader", "Unknown"),
                "transcript_preview": raw_info.get("description", "")[:800], # Match frontend expected key
                "view_count": raw_info.get("view_count", 0),
                "upload_date": raw_info.get("upload_date", "Unknown")
            }
    except Exception:
        return None

def get_related_news(query):
    """
    Searches NewsAPI for articles related to the query string.
    """
    if not NEWS_API_KEY or not query:
        return []
    
    try:
        # Search for the query, limiting to 3 articles for brevity
        url = f"https://newsapi.org/v2/everything?q={query}&pageSize=3&language=en&apiKey={NEWS_API_KEY}"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        articles = data.get("articles", [])
        return [{
            "title": a.get("title"),
            "url": a.get("url"),
            "source": a.get("source", {}).get("name")
        } for a in articles]
    except Exception:
        return []

def get_video_text(vid_id):
    """
    Fetches the closed captions/transcript.
    """
    from youtube_transcript_api import YouTubeTranscriptApi
    try:
        raw_transcript = YouTubeTranscriptApi.get_transcript(vid_id)
        # Combine all lines into one block of text
        full_text = " ".join([line['text'] for line in raw_transcript])
        return full_text[:5000] # Cap it so we don't hit AI token limits
    except:
        return "No transcript found for this video."

def clean_ai_json(text):
    """
    Sometimes the AI adds extra text around the JSON block. This strips it out.
    """
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return json.loads(text.strip())

# --- ROUTES ---

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def analyze_text():
    user_input = request.get_json()
    content = user_input.get("text", "")

    if len(content) < 15:
        return jsonify({"error": "Input too short to analyze."}), 400

    try:
        # Call Anthropic API
        resp = ai_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=800,
            system=TEXT_PROMPT,
            messages=[{"role": "user", "content": content}]
        )
        
        final_data = clean_ai_json(resp.content[0].text)
        
        # Use the first 100 characters of input as a search query for related news
        final_data["related_news"] = get_related_news(content[:100])
        
        return jsonify(final_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analyze-video", methods=["POST"])
def analyze_youtube():
    user_input = request.get_json()
    link = user_input.get("url", "")
    
    video_id = get_id_from_url(link)
    if not video_id:
        return jsonify({"error": "That doesn't look like a valid YouTube link."}), 400

    # Gather all the data for the AI
    meta = scrape_video_info(video_id)
    script = get_video_text(video_id)

    if not meta:
        return jsonify({"error": "Could not fetch video info."}), 500

    ai_payload = f"Title: {meta['title']}\nChannel: {meta['channel']}\nTranscript: {script}"

    try:
        resp = ai_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            system=VIDEO_PROMPT,
            messages=[{"role": "user", "content": ai_payload}]
        )
        
        report = clean_ai_json(resp.content[0].text)
        report["video_metadata"] = meta # Match frontend expected key

        # Use the video title to find related news articles
        report["related_news"] = get_related_news(meta['title'])
        
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": "AI analysis failed."}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)