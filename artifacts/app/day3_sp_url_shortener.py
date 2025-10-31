import random
import string
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, HttpUrl

app = FastAPI(
    title="Simple URL Shortener",
    description="A basic URL shortener service using FastAPI and in-memory storage.",
    version="1.0.0",
)

# In-memory storage for URL mappings
url_db = {}

class URLItem(BaseModel):
    url: HttpUrl

def generate_short_code(length: int = 6) -> str:
    """Generate a random alphanumeric short code of a given length."""
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))

@app.post("/shorten", summary="Create a short URL")
async def create_short_url(url_item: URLItem):
    """
    Takes a long URL and returns a unique short code.
    
    - **url**: The original URL to be shortened.
    """
    long_url = str(url_item.url)
    
    # Ensure the generated code is unique
    while True:
        short_code = generate_short_code()
        if short_code not in url_db:
            break
            
    url_db[short_code] = long_url
    return {"short_code": short_code}

@app.get("/{short_code}", summary="Redirect to the original URL")
async def redirect_to_url(short_code: str):
    """
    Redirects a short code to its original long URL.
    
    - **short_code**: The 6-character unique code.
    """
    if short_code in url_db:
        long_url = url_db[short_code]
        return RedirectResponse(url=long_url, status_code=301)
    else:
        raise HTTPException(status_code=404, detail="Short URL not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)