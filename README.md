
# PDF Q&A with RAG (Retrieval-Augmented Generation)

A Streamlit-based application that enables intelligent question-answering over PDF documents using Google's Gemini AI and persistent vector storage with Chroma DB.

## Features

- **PDF Upload & Processing**: Upload PDF documents and automatically extract text content
- **Intelligent Text Chunking**: Uses RecursiveCharacterTextSplitter for optimal document segmentation
- **Persistent Vector Storage**: Chroma DB with file-based persistence to avoid reprocessing
- **Advanced Embeddings**: Google's text-embedding-004 model for high-quality vector representations
- **Smart Retrieval**: Similarity-based document retrieval with configurable parameters
- **Gemini 2.0 Flash Integration**: Powered by Google's latest language model for accurate responses
- **Optimized Performance**: Hash-based caching system for vectorstore reuse

## Project Structure

```
├── rag.py          # Main Streamlit application
├── .env            # Environment variables (API keys)
├── temp_pdf/       # Temporary PDF storage (auto-created)
└── db_persist/     # Persistent vector databases (auto-created)
```

## Prerequisites

- Python 3.8+
- Google AI API Key (from Google AI Studio)

## Installation

1. **Install required packages**
   ```bash
   pip install streamlit python-dotenv
   pip install langchain==0.3.25
   pip install langchain-chroma==0.2.4
   pip install langchain-community==0.3.24
   pip install langchain-core==0.3.60
   pip install langchain-experimental==0.3.4
   pip install langchain-google-genai==2.0.10
   pip install langchain-text-splitters==0.3.8
   ```

2. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_google_ai_api_key_here
   ```
   
   To get your Google AI API key:
   - Visit [Google AI Studio](https://aistudio.google.com/)
   - Create an account or sign in
   - Generate an API key
   - Copy the key to your `.env` file

## Usage

1. **Run the application**
   ```bash
   python -m streamlit run rag.py
   ```

2. **Using the application**
   - Open your browser and navigate to the displayed local URL (typically `http://localhost:8501`)
   - Upload a PDF document using the file uploader
   - Wait for the document to be processed and vectorized
   - Ask questions about the content in the text input field
   - Get intelligent, context-aware answers powered by Gemini AI

## How It Works

1. **Document Processing**: PDF files are loaded and split into manageable chunks using RecursiveCharacterTextSplitter
2. **Vectorization**: Text chunks are converted to embeddings using Google's text-embedding-004 model
3. **Storage**: Vectors are stored in Chroma DB with persistence based on file hash for efficient reuse
4. **Retrieval**: When a question is asked, the most relevant document chunks are retrieved using similarity search
5. **Generation**: The retrieved context and question are sent to Gemini 2.0 Flash for answer generation

## Configuration Options

You can modify these parameters in `rag.py`:

- `chunk_size=1000`: Size of text chunks for processing
- `chunk_overlap=100`: Overlap between consecutive chunks
- `search_kwargs={"k": 8}`: Number of similar documents to retrieve
- `temperature=0.3`: Controls randomness in AI responses (0.0 = deterministic, 1.0 = creative)

## Dependencies

- **streamlit**: Web application framework
- **langchain**: LLM application framework
- **langchain-google-genai**: Google AI integration
- **langchain-chroma**: Vector database integration
- **langchain-community**: Community tools and loaders
- **python-dotenv**: Environment variable management

## Performance Features

- **Hash-based Caching**: Each PDF gets a unique vectorstore based on its MD5 hash
- **Persistent Storage**: Vectorstores are saved to disk and reused for identical files
- **Efficient Retrieval**: Only the most relevant chunks are used for answer generation

## Troubleshooting

**Common Issues:**

1. **API Key Error**: Ensure your `GOOGLE_API_KEY` is correctly set in the `.env` file
2. **Module Import Error**: Make sure all dependencies are installed with the exact versions specified
3. **File Upload Issues**: Check that you're uploading valid PDF files only
4. **Slow Processing**: Large PDFs may take time to process initially; subsequent queries will be faster due to caching
