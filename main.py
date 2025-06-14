import os
import json
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Optional
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

# Load scraped content
with open("data/scraped_content.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [entry["text"] for entry in data]
urls = [entry["url"] for entry in data]

# Prepare TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(texts)

# FastAPI app
app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None

@app.post("/api/")
async def answer_question(req: QuestionRequest):
    question = req.question
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, tfidf_matrix).flatten()

    # Get top 3 matches
    top_indices = similarities.argsort()[-3:][::-1]
    top_matches = [data[i] for i in top_indices]

    context = "\n\n---\n\n".join([match["text"] for match in top_matches])

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=[
                {"role": "system", "content": "You're a helpful teaching assistant for a course."},
                {"role": "user", "content": f"Answer the question:\n\n{question}\n\nOnly using the context below:\n{context}"}
            ]
        )

        answer = response.choices[0].message.content.strip()

        return {
            "answer": answer,
            "links": [
                {"url": match["url"], "text": match["text"][:100].strip() + "..."}
                for match in top_matches
            ]
        }

    except Exception as e:
        return {"error": str(e)}
