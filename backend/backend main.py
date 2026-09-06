import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="AI Research Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    topic: str
    depth: str = "standard"


@app.get("/")
def home():
    return {"message": "AI Research Agent API is running"}


@app.post("/research")
def research(request: ResearchRequest):

    prompt = f"""
You are an AI Research Agent.

Research topic:
{request.topic}

Research depth:
{request.depth}

Your task is to create a structured research report.

Follow this structure:

1. Executive Summary
2. Introduction
3. Key Research Questions
4. Major Findings
5. Benefits
6. Risks and Limitations
7. Different Perspectives
8. Important Statistics or Evidence
9. Future Trends
10. Conclusion
11. References

Important rules:
- Clearly separate facts from opinions.
- Do not invent sources, statistics, authors, papers, or URLs.
- If you are uncertain about a fact, say that it needs verification.
- Make the report useful for a college student.
- Use clear headings and bullet points.
"""

    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )

    return {
        "topic": request.topic,
        "depth": request.depth,
        "report": response.output_text
    }
