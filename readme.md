

## YouTube Transcript Application
This is a simple and interactive web application built with Streamlit that fetches and displays the transcript of a YouTube video along with timestamps. It uses the YouTube Transcript API to extract subtitles (if available) in the user’s preferred language.

## Features
Video Transcript Extraction: Fetch subtitles for a YouTube video in various languages.
Language Selection: Supports multiple languages such as English, Hindi, Spanish, French, German, and Japanese.
Timestamped Transcripts: Displays the transcript text with readable timestamps in MM:SS format.
Video Thumbnail Preview: Automatically displays the video thumbnail for visual confirmation of the video.
Error Handling: Provides actionable error messages if subtitles are unavailable or the video URL is invalid.
Installation and Setup
Prerequisites
Python 3.7 or higher
pip (Python package manager)

## Step-by-Step Instructions

## Clone the repository:
bash
Copy code
git clone https://github.com/kishan444444/YouTube-Transcript-Application.git
cd your-repo-name

## Install required packages:
bash
Copy code
pip install -r requirements.txt

## Ensure the following libraries are included in the requirements.txt file:
plaintext
Copy code
streamlit
youtube-transcript-api
Run the application:

## bash
Copy code
streamlit run app.py
Open the application:
Open your browser and navigate to http://localhost:8501.

## How to Use
Enter the YouTube video URL in the text input field.
Example: https://www.youtube.com/watch?v=VIDEO_ID.
Select your preferred language for the transcript from the dropdown menu.
Click the "Show Transcript with Timestamps" button.
View the transcript with timestamps in the MM:SS format.
If subtitles are unavailable or the video URL is invalid, appropriate error messages will guide you.

