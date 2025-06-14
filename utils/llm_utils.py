from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()



# Use environment variables or hardcode for now (NOT recommended for prod)
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN") or "your-aipipe-token-here"
AIPIPE_BASE_URL = "https://aipipe.org/openrouter/v1"  # or /openai/v1

# Configure the client
client = OpenAI(
    base_url=AIPIPE_BASE_URL,
    api_key=AIPIPE_TOKEN
)

def get_answer(question: str):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4.1-nano",  # example model via AIPipe
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ]
        )
        answer = response.choices[0].message.content.strip()
        return answer, []
    except Exception as e:
        print("Error in get_answer:", e)
        return "Sorry, something went wrong.", []
