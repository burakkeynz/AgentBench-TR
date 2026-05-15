from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.query import router as query_router
from api.routes.metrics import router as metrics_router

app = FastAPI(
    title="AgentBench-TR API",
    description="Multi-agent QA system with eval framework.",
    version="0.1.0",
)

# Adding CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registering routers
app.include_router(query_router)
app.include_router(metrics_router)


@app.get("/health")
def health_check():
    """Checking API health status."""
    return {"status": "ok", "service": "AgentBench-TR"}