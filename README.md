# 🤖 Building a Legal Assistant AI Chatbot with LangChain and Python

## 📋 System Requirements

- Python 3.8 or higher, recommended version 3.8.18 (Download: [python.org](https://www.python.org/downloads/))
- Docker Desktop (Download: [docker.com](https://www.docker.com/products/docker-desktop/))
- OpenAI API key (Register: [platform.openai.com](https://platform.openai.com/api-keys))
- Approximately 4GB of free RAM

## 🚀 Installation and Setup Steps

### Step 1: Set Up Environment

- Recommended to use Python version 3.8.18.
- Use conda to set up the environment with the command: `conda create -n myenv python=3.8.18`
- Activate the environment with the command: `conda activate myenv`
- Open Terminal/Command Prompt and run the following command:
  - `pip install -r requirements.txt`

### Step 2: Download Ollama

- Visit: [ollama.com/download](https://ollama.com/download)
- Choose the appropriate version for your operating system
- Follow the installation instructions
- Run the command: `ollama run llama2`

### Step 3: Install and Run Milvus Database

1. Start Docker Desktop
2. Open Terminal/Command Prompt and run the command:
   ```sh
   docker compose up --build
   ```

Option: Install attu to view data seeded into Milvus:
  1. Run the command
   ```sh
   docker run -p 8000:3000 -e MILVUS_URL={milvus server IP}:19530 zilliz/attu:v2.4
   ```
   2. Replace "milvus server IP" with your local internet IP. To get the local IP, run:
   ```sh
   ipconfig
   ```
   or similar commands for other operating systems.

### Step 4: Configure OpenAI API
1. Create a `.env` file
2. Visit OpenAI to get your `OPENAI_API_KEY:` [platform.openai.com](platform.openai.com)
3. Add the API key to the `.env` file
```sh
OPENAI_API_KEY=sk-your-api-key-here
```
Options: Configure Langsmith:
1. Visit Langsmith to get your `LANGCHAIN_API_KEY`: [smith.langchain.com](smith.langchain.com)
2. Add the following lines to the `.env` file:
```sh
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your-langchain-api-key-here"
LANGCHAIN_PROJECT="project-name"
```

### Step 5: Run the Application
1. Crawl data to local: Open Terminal/Command Prompt, navigate to the src directory with `cd src` and run:
```sh
python crawl.py
```
2. Seed data into Milvus
Check if the data is in Milvus by visiting: http://localhost:8000/#/databases/default/collections Ensure Milvus is running with:
```sh
docker run -p 8000:3000 -e MILVUS_URL={milvus server IP}:19530 zilliz/attu:v2.4
```
3. Run the application
```sh 
streamlit run main.py
```
## 💻 Usage
### 1. Start the Application
1. Ensure Docker Desktop is running
2. Ensure Ollama is running with the llama2 model
3. Open Terminal/Command Prompt, navigate to the src directory
4. Run the command: streamlit run main.py
### 2. Load and Process Data
#### Option 1: From Local JSON File

1. Select the "File Local" tab on the sidebar
2. Enter the directory path containing the JSON file (default: data)
3. Enter the JSON file name (default: stack.json)
4. Click "Load data from file"
5. Wait for the system to process and notify success
#### Option 2: From URL

1. Select the "Direct URL" tab on the sidebar
2. Enter the URL to crawl data
3. Click "Crawl data"
4. Wait for the system to crawl and process the data
#### 3. Interact with the Chatbot
1. Enter your question in the chat box at the bottom of the screen
2. Press Enter or click the send button to submit your question
3. The chatbot will:
- Search for relevant information in the database
- Combine results from multiple sources
- Generate a context-based response
- Chat history will be displayed in the main screen area
#### 4. View System Information
1. Monitor Milvus connection status on the sidebar
2. Check the number of documents loaded
3. View information about the model being used
## 📚 References
- LangChain: https://python.langchain.com/docs/introduction/
  - Agents: https://python.langchain.com/docs/tutorials/qa_chat_history/#tying-it-together-1
  - BM25: https://python.langchain.com/docs/integrations/retrievers/bm25/#create-a-new-retriever-with-documents
  - How to combine results from multiple retrievers: https://python.langchain.com/docs/how_to/ensemble_retriever/
  - Langchain Milvus: https://python.langchain.com/docs/integrations/vectorstores/milvus/#initialization
  - Recursive URL: https://python.langchain.com/docs/integrations/document_loaders/recursive_url/#overview
  - Langchain Streamlit: https://python.langchain.com/docs/integrations/callbacks/streamlit/#installation-and-setup
  - Langchain Streamlit: https://python.langchain.com/docs/integrations/providers/streamlit/#memory
- Milvus Standalone: https://milvus.io/docs/v2.0.x/install_standalone-docker.md
  - Attu: https://github.com/zilliztech/attu
- Streamlit Documentation: https://docs.streamlit.io/
- OpenAI API: https://platform.openai.com/docs
