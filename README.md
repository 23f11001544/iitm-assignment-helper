# IIT Madras Assignment Helper API

An API service that automatically answers questions from IIT Madras' Online Degree in Data Science graded assignments.

## Overview

This application provides an API endpoint that accepts questions from the Tools in Data Science course graded assignments and returns the corresponding answers. It can process various types of questions, including those that require:

- CSV data analysis
- ZIP file extraction
- PDF content analysis
- YouTube video information
- Web page content analysis
- General knowledge questions

## API Usage

### Endpoint

```
POST https://your-app.vercel.app/api/
```

### Request Format

Send a `multipart/form-data` request with:

- `question`: The assignment question text (required)
- `file`: An optional file attachment (e.g., CSV, ZIP, PDF)

### Example Request

```bash
curl -X POST "https://your-app.vercel.app/api/" \
  -H "Content-Type: multipart/form-data" \
  -F "question=Download and unzip file abcd.zip which has a single extract.csv file inside. What is the value in the 'answer' column of the CSV file?" \
  -F "file=@abcd.zip"
```

### Response Format

```json
{
  "answer": "1234567890"
}
```

## Features

- **File Processing**: Handles ZIP archives, CSV files, and other data formats
- **Web Content Analysis**: Extracts information from web pages and YouTube videos
- **AI-Powered Answers**: Uses OpenAI's GPT models to generate accurate responses
- **Error Handling**: Provides informative error messages

## Local Development

### Prerequisites

- Python 3.8 or higher
- An OpenAI API key

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/your-username/iit-madras-assignment-helper.git
   cd iit-madras-assignment-helper
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. Run the application locally:
   ```
   python api/index.py
   ```

## Deployment

This application is configured for deployment on Vercel:

1. Install the Vercel CLI:
   ```
   npm install -g vercel
   ```

2. Deploy the application:
   ```
   vercel
   ```

3. Set environment variables on Vercel:
   ```
   vercel env add OPENAI_API_KEY
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
