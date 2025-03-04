import streamlit as st
import logging
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()  # Load all the environment variables

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

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
        # Extract video ID from URL
        if "v=" in youtube_video_url:
            video_id = youtube_video_url.split("v=")[1].split("&")[0]
        else:
            raise ValueError("Invalid YouTube URL format")

        # Fetch transcript in the desired language
        try:
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        except TranscriptsDisabled:
            raise ValueError(f"Transcripts are disabled for this video.")
        except Exception as e:
            logging.error(f"Error fetching transcript for video {video_id}: {e}")
            available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            available_langs = [t.language for t in available_transcripts]
            raise ValueError(f"Transcript not found in {language}. Available languages: {', '.join(available_langs)}")

        return transcript_data
    except Exception as e:
        logging.error(f"Error extracting transcript: {str(e)}")
        raise

def process_and_query_transcript(transcript, query):
    try:
        # Combine transcript text into a single string
        transcript_text = " ".join([item["text"] for item in transcript])

        # Split transcript text into smaller chunks for efficient processing
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(transcript_text)

        # Generate embeddings for the transcript chunks
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_texts(chunks, embeddings)

        # Create retriever from FAISS database
        retriever = db.as_retriever()

        # Initialize Groq model for query processing
        model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

        # Define contextualization prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

        # Define system prompt for answering the query
        system_prompt = (
            "You are an AI assistant that helps summarize and answer questions from documents.\n\n"
            "Context:\n{context}\n\n"
            "Chat History:\n{chat_history}\n\n"
            "User Question:\n{input}"
        )

        qa_prompt = ChatPromptTemplate.from_template(system_prompt)

        # Create a chain to process the query
        question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Get the response from the chain
        chat_history = []
        response = rag_chain.invoke({"input": query, "chat_history": chat_history})

        return response['answer']
    except Exception as e:
        logging.error(f"Error processing transcript and querying: {str(e)}")
        raise

# Streamlit UI
st.title("YouTube Video Transcript & Query System")

# User input for YouTube link and language
youtube_link = st.text_input("Enter YouTube Video Link:")
language_choice = st.selectbox("Choose Transcript Language:", list(LANGUAGE_CODES.keys()))

# Show video thumbnail if the link is provided
if youtube_link:
    if "v=" in youtube_link:
        video_id = youtube_link.split("v=")[1].split("&")[0]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)

# Button to fetch and display transcript
if st.button("Show Transcript with Timestamps"):
    if not youtube_link:
        st.error("Please enter a valid YouTube URL.")
    else:
        try:
            language_code = LANGUAGE_CODES.get(language_choice, "en")
            transcript = extract_transcript_details(youtube_link, language_code)

            # Display transcript with timestamps
            st.markdown("### Transcript with Timestamps")
            for item in transcript:
                start_time = item["start"]
                text = item["text"]
                minutes = int(start_time // 60)
                seconds = int(start_time % 60)
                timestamp = f"{minutes:02}:{seconds:02}"
                st.write(f"**{timestamp}** - {text}")
        except Exception as e:
            st.error(f"Error fetching transcript: {str(e)}")

# Query input section
query = st.text_input("Enter your query:")

if st.button("Fetch and Query"):
    if not youtube_link:
        st.error("Please enter a valid YouTube URL.")
    else:
        try:
            language_code = LANGUAGE_CODES.get(language_choice, "en")
            transcript = extract_transcript_details(youtube_link, language_code)

            if isinstance(transcript, list):  # Check if transcript is successfully fetched
                st.write("Transcript fetched successfully!")
                results = process_and_query_transcript(transcript, query)
                st.write("### Query Results:")
                st.write(results)
            else:
                st.error(transcript)
        except Exception as e:
            st.error(f"Error processing your query: {str(e)}")


