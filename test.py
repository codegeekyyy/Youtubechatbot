import langchain
import transformers
import streamlit as st
import langchain_community
from youtube_transcript_api import YouTubeTranscriptApi


print(langchain.__version__)
print(transformers.__version__)
print(st.__version__)
print(langchain_community.__version__)
print(hasattr(YouTubeTranscriptApi, 'list_transcripts'))



