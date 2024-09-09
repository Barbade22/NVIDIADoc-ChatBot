# Project: NVIDIA CUDA Documentation Question Answering System

## Overview
This project implements a question answering system for the NVIDIA CUDA documentation using web crawling, data chunking, vector database storage, and a language model for answering queries. Due to hardware limitations, ChromaDB is used instead of Milvus for storing and retrieving data.

## Files and Functionalities

### 1. WebCrawler.py
**Overview:**
- **Functionality:** Crawls the NVIDIA CUDA documentation website (https://docs.nvidia.com/cuda/) up to a specified depth to extract plain text content from web pages and their sub-links.
- **Output:** Saves the crawled data with URLs into a JSON file (cuda_docs_crawled_data.json).

**Usage:**
- Ensure Python dependencies (requests, beautifulsoup4) are installed.
- Adjust parameters (base_url, max_depth, max_pages) for crawling depth and limit.

### 2. ChromaDB.py
**Overview:**
- **Functionality:** Initializes ChromaDB to create a collection and insert text chunks with associated metadata (URLs).
- **Output:** Stores segmented text data into ChromaDB for efficient retrieval.

**Usage:**
- Requires chromadb Python library.
- Adjust settings for collection name and interaction with ChromaDB based on specific needs.

### 3. Robeata.py
**Overview:**
- **Functionality:** Implements a Streamlit-based application for question answering using data stored in ChromaDB and a pre-trained RoBERTa model.
- **Output:** Allows users to input questions and retrieves answers based on the stored NVIDIA CUDA documentation.

**Usage:**
- Requires streamlit, transformers, chromadb, torch.
- Provides a user-friendly interface to interactively query and receive answers.

### 4. Interactive.ipynb
**Overview:**
- **Functionality:** Jupyter Notebook script for interactive data chunking, database interaction with ChromaDB, and question answering using a pre-trained model.
- **Output:** Segments input data into chunks, stores them in ChromaDB, and answers queries using the RoBERTa model.

**Usage:**
- Requires Jupyter Notebook and relevant Python libraries (transformers, torch, chromadb, beautifulsoup4, requests).
- Executes sequentially to perform data chunking, database insertion, and question answering tasks.

### 5. README.md
**Overview:**
- **Functionality:** This file provides an overview of the project, descriptions of each file, usage instructions, and notes about using ChromaDB instead of Milvus due to hardware limitations.
- **Output:** Guides users on setting up dependencies, running each script, and adapting functionalities as needed.

## Setup and Dependencies

### Python: 
Ensure Python 3.x is installed.

### Dependencies: 
Install required Python libraries using pip:
```bash
pip install transformers torch chromadb streamlit beautifulsoup4 requests



Usage Instructions
Web Crawling:

Run WebCrawler.py to extract data from the NVIDIA CUDA documentation website.
Adjust crawling parameters (base_url, max_depth, max_pages) as required.
ChromaDB Initialization and Data Insertion:

Execute ChromaDB.py to initialize ChromaDB, create a collection, and insert segmented data with metadata.
Modify collection settings and data handling based on project needs.
Question Answering Interface:

Launch Robeata.py to start the Streamlit application.
Input questions about the NVIDIA CUDA documentation to retrieve answers using the RoBERTa model.
Interactive Data Processing:

Open Interactive.ipynb in Jupyter Notebook and execute cells sequentially.
Perform interactive data chunking, ChromaDB interaction, and question answering using the integrated RoBERTa model.
Notes
Adjust configurations, parameters, and input data formats according to specific project requirements.
ChromaDB is chosen over Milvus for database operations due to hardware constraints.
Ensure sufficient hardware resources for running intensive NLP tasks and database operations effectively.
Author

[Govind Barbade]

