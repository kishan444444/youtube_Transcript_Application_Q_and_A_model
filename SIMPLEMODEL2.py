import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import  SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# Language mapping
LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
}

# Load environment variables
import os
from dotenv import load_dotenv
load_dotenv()  #load all the environment variables
## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

groq_api_key=os.getenv("GROQ_API_KEY")


# Function to extract transcript details
def extract_transcript_details(youtube_video_url, language="en"):
    try:
        if "v=" in youtube_video_url:
            video_id = youtube_video_url.split("v=")[1].split("&")[0]
        else:
            return "Invalid YouTube URL. Please enter a valid format."

        # Fetch transcript in the desired language
        try:
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        except:
            # Fallback to available languages
            available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            available_langs = [t.language for t in available_transcripts]
            return f"Transcript not found in {language}. Available languages: {', '.join(available_langs)}"

        return transcript_data
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

def process_and_query_transcript(transcript, query):
    try:
        # Combine transcript text
        transcript_text = " ".join([item["text"] for item in transcript])

        # Split into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(transcript_text)

        # Generate embeddings using LangChain's SentenceTransformerEmbeddings
        embeddings =  SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma.from_texts(chunks, embeddings)

        # Retrieve results for the query
        retriever = db.as_retriever()

        model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

        # Updated prompt template
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert AI Engineer. Use the provided context to answer questions accurately."),
                ("user", "Context: {context}\n\nQuestion: {input}")
            ]
        )

        document_chain = create_stuff_documents_chain(model, prompt)

        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Invoke the retrieval chain
        results = retrieval_chain.invoke({"input": query})

        # Extract the answer from the response
        if isinstance(results, dict) and "answer" in results:
            return [results["answer"]]
        else:
            return ["Unexpected response format."]

    except Exception as e:
        return [f"Error processing and querying transcript: {str(e)}"]



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

# Query input
query = st.text_input("Enter your query:")

if st.button("Fetch and Query"):
    if not youtube_link:
        st.error("Please enter a valid YouTube URL.")
    else:
        language_code = LANGUAGE_CODES.get(language_choice, "en")
        transcript = extract_transcript_details(youtube_link, language_code)

        if isinstance(transcript, list):  # Ensure transcript is successfully fetched
            st.write("Transcript fetched successfully!")
            results = process_and_query_transcript(transcript, query)
            st.write("Query Results:")
            for idx, res in enumerate(results, start=1):
                st.write(f"{idx}. {res}")
        else:
            st.error(transcript)
