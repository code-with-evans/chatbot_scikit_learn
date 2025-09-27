import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot_inference import Chatbot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="ML Chatbot API", description="A machine learning chatbot using CLINC150 and frankdarkluo/DailyDialog datasets", version="1.0.0")

try:
    chatbot = Chatbot()
    logging.info("Chatbot initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize chatbot: {e}")
    raise

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    intent: str | None
    intent_confidence: float
    response_confidence: float

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        user_message = request.message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        result = chatbot.generate_response(user_message)

        logging.info(f"Processed message: {user_message} | Response: {result['response']}")
        return ChatResponse(**result)
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
