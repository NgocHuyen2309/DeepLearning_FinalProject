import uvicorn

if __name__ == "__main__":
    print("Khởi động Deepfake Detection API (Microservice)...")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8001, reload=True)
