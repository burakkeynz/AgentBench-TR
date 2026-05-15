import time
from difflib import SequenceMatcher
from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from graph.pipeline import pipeline
from graph.state import AgentState

_embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")


def _string_similarity(a: str, b: str) -> float:
    """Calculating string similarity using SequenceMatcher."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _semantic_similarity(a: str, b: str) -> float:
    """Calculating semantic similarity using sentence embeddings."""
    embs = _embedder.encode([a, b])
    return float(cosine_similarity([embs[0]], [embs[1]])[0][0])


def run_consistency_check(question: str, n: int = 10) -> dict:
    """Running consistency check by invoking pipeline n times and comparing answers."""
    print(f"Running consistency check — '{question[:50]}' x{n}...")
    answers = []

    for i in range(n):
        state = AgentState(query=question)
        result = pipeline.invoke(state)
        answers.append(result["final_answer"])
        print(f"  Collecting answer {i+1}/{n} — done.")

    # Calculating pairwise similarities
    string_scores = []
    semantic_scores = []

    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            string_scores.append(_string_similarity(answers[i], answers[j]))
            semantic_scores.append(_semantic_similarity(answers[i], answers[j]))

    avg_string   = round(sum(string_scores)   / len(string_scores),   4) if string_scores   else 0.0
    avg_semantic = round(sum(semantic_scores) / len(semantic_scores), 4) if semantic_scores else 0.0
    avg_combined = round((avg_string + avg_semantic) / 2, 4)

    print(f"Running consistency check — string={avg_string}, semantic={avg_semantic}, combined={avg_combined}.")

    return {
        "question":       question,
        "n":              n,
        "answers":        answers,
        "string_sim":     avg_string,
        "semantic_sim":   avg_semantic,
        "combined_score": avg_combined,
    }