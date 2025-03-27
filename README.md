# Activate venv
source ai-venv/bin/activate

# FastAPI Backend
uvicorn server.main:app --reload

# Streamlit UI
streamlit run ui/app.py

# AI Model
(Small)  ollama run tinyllama
(Medium) ollama run llama3
(Large)  ollama run mistral



# Issues:


# Solved:
-Uvicorn server ModuleNotFoundError
Typo, Fixed by changing 'request,py' to 'request.py'

-Index error
Missing logic, Fixed by adding logic to enable a response if no context is provided.

-Chunk-text & Embed Attribute Errors
Missing methods, Fixed by adding chunk-text & embed methods to vectorstore.py

-ValueError: not enough values to unpack (expected 2, got 1)
