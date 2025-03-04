import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi

# Language mapping
LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
}

# Function to extract transcript details
def extract_transcript_details(youtube_video_url, language="en"):
    try:
        # Extract the video ID from the URL
        if "v=" in youtube_video_url:
            video_id = youtube_video_url.split("v=")[1].split("&")[0]
        else:
            return "Invalid YouTube URL. Please enter a URL in the format https://www.youtube.com/watch?v=VIDEO_ID."
        
        # Fetch the transcript for the video
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        
        # Format the transcript
        transcript = [
            {"start": i["start"], "text": i["text"]} for i in transcript_text
        ]
        return transcript
    except Exception as e:
        if "Subtitles are disabled for this video" in str(e):
            return "Subtitles are disabled for this video, and no transcript is available."
        else:
            return f"Error fetching transcript: {str(e)}"

# Streamlit interface
st.title("YouTube Video Transcript with Timestamps")

# User input for the YouTube video link
youtube_link = st.text_input("Enter YouTube Video Link:")
language_choice = st.selectbox(
    "Choose Transcript Language:",
    list(LANGUAGE_CODES.keys())
)

# Display video thumbnail if the link is provided
if youtube_link:
    if "v=" in youtube_link:
        video_id = youtube_link.split("v=")[1].split("&")[0]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)

# Button to fetch and display the transcript
if st.button("Show Transcript with Timestamps"):
    if not youtube_link:
        st.error("Please enter a valid YouTube URL.")
    else:
        with st.spinner("Fetching transcript..."):
            # Fetch the transcript
            language_code = LANGUAGE_CODES.get(language_choice, "en")
            transcript = extract_transcript_details(youtube_link, language_code)

            # Handle errors or display the transcript
            if isinstance(transcript, str):
                st.error(transcript)
                if "Subtitles are disabled" in transcript:
                    st.markdown(
                        """
                        ### What you can do:
                        - Check if the video has subtitles enabled on YouTube.
                        - Try with a different video URL that supports subtitles.
                        - If you believe this is an error, check the [YouTube Transcript API repository](https://github.com/jdepoix/youtube-transcript-api/issues) for solutions.
                        """
                    )
            else:
                st.markdown("### Transcript with Timestamps")
                for item in transcript:
                    start_time = item["start"]
                    text = item["text"]

                    # Convert start time to minutes:seconds
                    minutes = int(start_time // 60)
                    seconds = int(start_time % 60)
                    timestamp = f"{minutes:02}:{seconds:02}"

                    # Display the timestamp and text
                    st.write(f"**{timestamp}** - {text}")
