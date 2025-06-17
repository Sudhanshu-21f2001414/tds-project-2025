# import os
# import json
# import logging
# from fastapi import FastAPI
# from pydantic import BaseModel
# from dotenv import load_dotenv
# from typing import Optional
# from openai import OpenAI
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Setup
# load_dotenv()
# logging.basicConfig(level=logging.INFO)
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

# # Load content
# try:
#     with open("data/scraped_content.json", "r", encoding="utf-8") as f:
#         data = json.load(f)
# except FileNotFoundError:
#     logging.error("❌ scraped_content.json not found. Make sure it exists.")
#     data = []

# texts = [entry["text"] for entry in data]
# urls = [entry["url"] for entry in data]
# vectorizer = TfidfVectorizer(stop_words="english")
# tfidf_matrix = vectorizer.fit_transform(texts) if texts else None

# # App
# app = FastAPI()

# class QuestionRequest(BaseModel):
#     question: str
#     image: Optional[str] = None

# @app.get("/")
# async def root():
#     return {"status": "FastAPI is running"}

# @app.post("/api/")
# async def answer_question(req: QuestionRequest):
#     logging.info(f"Incoming question: {req.question}")
#     if not tfidf_matrix:
#         return {"error": "Context data not loaded"}

#     question_vec = vectorizer.transform([req.question])
#     similarities = cosine_similarity(question_vec, tfidf_matrix).flatten()
#     top_indices = similarities.argsort()[-3:][::-1]
#     top_matches = [data[i] for i in top_indices]
#     context = "\n\n---\n\n".join([match["text"] for match in top_matches])

#     try:
#         response = client.chat.completions.create(
#             model=os.getenv("OPENAI_MODEL"),
#             messages=[
#                 {"role": "system", "content": "You're a helpful teaching assistant for a course."},
#                 {"role": "user", "content": f"Answer the question:\n\n{req.question}\n\nOnly using the context below:\n{context}"}
#             ]
#         )

#         answer = response.choices[0].message.content.strip()
#         return {
#             "answer": answer,
#             "links": [{"url": match["url"], "text": match["text"][:100].strip() + "..."} for match in top_matches]
#         }

#     except Exception as e:
#         logging.error(f"OpenAI call failed: {str(e)}")
#         return {"error": str(e)}




import os
import json
import logging
import traceback
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# Load content
try:
    with open("data/scraped_content.json", "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    logging.error("❌ scraped_content.json not found. Make sure it exists.")
    data = []

texts = [entry["text"] for entry in data]
urls = [entry["url"] for entry in data]
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(texts) if texts else None

# App
app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None

@app.get("/")
async def root():
    return {"status": "FastAPI is running"}

@app.post("/api/")
async def answer_question(req: QuestionRequest):
    logging.info(f"📥 Incoming question: {req.question}")

    if tfidf_matrix is None or tfidf_matrix.shape[0] == 0:
        return {"error": "Context data not loaded"}


    try:
        question_vec = vectorizer.transform([req.question])
        similarities = cosine_similarity(question_vec, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-3:][::-1]
        top_matches = [data[i] for i in top_indices]
        context = "\n\n---\n\n".join([match["text"] for match in top_matches])

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=[
                {"role": "system", "content": "You're a helpful teaching assistant for a course."},
                {"role": "user", "content": f"Answer the question:\n\n{req.question}\n\nOnly using the context below:\n{context}"}
            ]
        )

        answer = response.choices[0].message.content.strip()
        return {
            "answer": answer,
            "links": [{"url": match["url"], "text": match["text"][:100].strip() + "..."} for match in top_matches]
        }

    except Exception as e:
        logging.error("⚠️ OpenAI call failed:")
        traceback.print_exc()
        return {"error": str(e)}
