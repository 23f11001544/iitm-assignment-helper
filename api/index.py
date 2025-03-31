import os
import json
import tempfile
import zipfile
import csv
import pandas as pd
import requests
from bs4 import BeautifulSoup
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
import re
from pytube import YouTube
import openai
import io

app = Flask(__name__)

# Set your OpenAI API key
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    print("Warning: OPENAI_API_KEY environment variable not set")

@app.route('/', defaults={'path': ''}, methods=['GET'])
def home(path):
    return jsonify({"status": "API is running", "usage": "Send POST requests to / with 'question' and optional 'file'"})

@app.route('/', methods=['POST'])
def process_question():
    try:
        # Extract question from form data
        question = request.form.get('question', '')
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        # Check if there's a file attached
        attached_file = None
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                # Save file to temp location
                with tempfile.NamedTemporaryFile(delete=False) as temp:
                    file.save(temp.name)
                    attached_file = temp.name
        
        # Process the question and get the answer
        answer = process_assignment_question(question, attached_file)
        
        # Clean up temp file if it exists
        if attached_file and os.path.exists(attached_file):
            os.remove(attached_file)
        
        return jsonify({"answer": answer})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def process_assignment_question(question, file_path=None):
    """
    Process an assignment question and return the answer.
    
    Args:
        question (str): The assignment question
        file_path (str, optional): Path to an attached file
        
    Returns:
        str: The answer to the question
    """
    # Check if question involves CSV processing
    if file_path and (re.search(r'\.zip\b', question, re.IGNORECASE) or re.search(r'\.csv\b', question, re.IGNORECASE)):
        return process_csv_question(question, file_path)
    
    # Check if question involves PDF processing
    elif file_path and re.search(r'\.pdf\b', question, re.IGNORECASE):
        return process_pdf_question(question, file_path)
    
    # Check if question involves YouTube video
    elif re.search(r'youtube\.com|youtu\.be', question, re.IGNORECASE):
        return process_youtube_question(question)
    
    # Check if question involves web page content
    elif re.search(r'https?://\S+', question, re.IGNORECASE):
        return process_webpage_question(question)
    
    # Default to LLM-based approach for other questions
    else:
        return process_general_question(question)

def process_csv_question(question, file_path):
    """Process questions requiring CSV file analysis"""
    # If it's a ZIP file containing a CSV
    if file_path.endswith('.zip') or zipfile.is_zipfile(file_path):
        # Extract all files from ZIP to a temporary directory
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find CSV files in the extracted content
        csv_files = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            return "No CSV files found in the ZIP archive."
        
        # For simplicity, use the first CSV file found
        csv_file_path = csv_files[0]
        
        # Read the CSV with pandas
        df = pd.read_csv(csv_file_path)
        
        # Look for specific patterns in the question to determine what to do
        if "answer column" in question.lower():
            if "answer" in df.columns:
                # Return the first value in the "answer" column if it exists
                return str(df["answer"].iloc[0])
            else:
                return "No 'answer' column found in the CSV."
        
        # More CSV analysis logic can be added here based on common question patterns
        
        # If not sure what to do, use LLM to analyze the CSV
        csv_content = df.head(10).to_string()  # Convert first 10 rows to string
        return process_general_question(f"{question}\n\nCSV content (first 10 rows):\n{csv_content}")
    
    # If it's directly a CSV file
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        
        # Similar logic as above
        if "answer column" in question.lower():
            if "answer" in df.columns:
                return str(df["answer"].iloc[0])
            else:
                return "No 'answer' column found in the CSV."
        
        csv_content = df.head(10).to_string()
        return process_general_question(f"{question}\n\nCSV content (first 10 rows):\n{csv_content}")
    
    return "Unsupported file format."

def process_pdf_question(question, file_path):
    """Process questions requiring PDF analysis"""
    # For PDF files, we would normally use a library like PyPDF2 or pdfplumber
    # Since we have limited resources, we'll use LLM to handle this by explaining
    return "PDF processing requires additional libraries. Please extract the relevant information from the PDF and include it in your question."

def process_youtube_question(question):
    """Process questions related to YouTube videos"""
    # Extract YouTube URL from the question
    youtube_urls = re.findall(r'(https?://(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]+)', question)
    
    if not youtube_urls:
        return "No YouTube URL found in the question."
    
    youtube_url = youtube_urls[0][0]  # Get the first matched URL
    
    try:
        # Get video details using pytube
        yt = YouTube(youtube_url)
        video_title = yt.title
        video_description = yt.description
        
        # We can't download videos due to resource constraints, but we can analyze metadata
        video_info = f"Title: {video_title}\nDescription: {video_description}"
        
        return process_general_question(f"{question}\n\nVideo information:\n{video_info}")
    except Exception as e:
        return f"Error processing YouTube video: {str(e)}"

def process_webpage_question(question):
    """Process questions requiring web page content analysis"""
    # Extract URL from the question
    urls = re.findall(r'(https?://\S+)', question)
    
    if not urls:
        return "No URL found in the question."
    
    url = urls[0]  # Get the first matched URL
    
    try:
        # Fetch webpage content
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text content (limited to save resources)
        page_text = soup.get_text()[:5000]  # Limit to first 5000 characters
        
        return process_general_question(f"{question}\n\nWebpage content (excerpt):\n{page_text}")
    except Exception as e:
        return f"Error processing webpage: {str(e)}"

def process_general_question(question):
    """Process general questions using OpenAI API"""
    try:
        if not openai_api_key:
            return "API key not configured. Please set the OPENAI_API_KEY environment variable."
        
        # Configure OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)
        
        # Send question to OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for IIT Madras' Online Degree in Data Science. Your job is to provide direct, concise answers to assignment questions. Only provide the final answer without explanation."},
                {"role": "user", "content": question}
            ],
            max_tokens=150
        )
        
        # Extract and return the answer
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error using OpenAI API: {str(e)}"

# For local testing
if __name__ == '__main__':
    app.run(debug=True)