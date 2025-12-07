import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env (if present)
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def analyze_exercise_metrics(metrics_file="exercise_metrics.txt"):
    try:
        with open(metrics_file, "r") as f:
            exercise_data = f.read()
    except FileNotFoundError:
        exercise_data = "No exercise metrics found. Please complete a workout first."

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    messages = [
        (
            "system",
            "You are a helpful fitness assistant that analyzes exercise metrics and suggests improvements and tips. At last, suggest which sport the user should try based on their performance data. "
            "Provide specific, actionable feedback based on the scores and data provided.",
        ),
        (
            "human",
            f"Here are my exercise metrics:\n\n{exercise_data}\n\nPlease analyze my performance and provide specific suggestions for improvement.",
        ),
    ]

    ai_msg = llm.invoke(messages)
    return ai_msg.content


if __name__ == "__main__":
    result = analyze_exercise_metrics()
    print(result)
