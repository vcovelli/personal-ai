import requests

# def query_ollama(prompt: str, model: str = "mistral"):
   # response = requests.post(
        #"http://localhost:11434/api/generate",
        #json={"model": model, "prompt": prompt, "stream": False},
    #)
    #return response.json()["response"]

def query_ollama(prompt: str, model: str) -> str:
    return f"Simulated response to: '{prompt}' using model '{model}'"
