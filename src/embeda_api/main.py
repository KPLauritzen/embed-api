from fastapi import FastAPI

app = FastAPI(title="embeda-api")


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}
