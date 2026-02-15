"""
Production Inference Server
===========================
FastAPI-based REST API for real-time ML inference using ONNX Runtime.
Supports single image and batch prediction with automatic preprocessing.

Author: SQ1111
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import time
from typing import Optional
from pydantic import BaseModel


# ============================================================
# App Configuration
# ============================================================
app = FastAPI(
    title="MLforge Inference API",
    description="Real-time ML inference with ONNX Runtime",
    version="1.0.0",
    docs_url="/docs",
)

MODEL_PATH = "outputs/model_int8.onnx"
IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Global session (loaded once at startup)
session: Optional[ort.InferenceSession] = None


# ============================================================
# Models
# ============================================================
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_5: list
    latency_ms: float
    model: str = "EfficientNet-B0 INT8"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    runtime: str


class BenchmarkResponse(BaseModel):
    iterations: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    throughput_fps: float


# ============================================================
# Preprocessing
# ============================================================
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess uploaded image for inference."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    
    # Convert to numpy and normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = (img_array - MEAN) / STD
    
    # HWC → CHW → NCHW
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


# ============================================================
# Startup / Shutdown
# ============================================================
@app.on_event("startup")
async def load_model():
    """Load ONNX model at server startup."""
    global session
    try:
        session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )
        print(f"✓ Model loaded: {MODEL_PATH}")
        print(f"  Providers: {session.get_providers()}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("  Server will start but /predict will return errors")


# ============================================================
# Endpoints
# ============================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server and model health."""
    return HealthResponse(
        status="healthy" if session else "model_not_loaded",
        model_loaded=session is not None,
        model_path=MODEL_PATH,
        runtime=f"ONNX Runtime {ort.__version__}",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Run inference on uploaded image."""
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read and preprocess
    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)
    
    # Inference with timing
    start = time.perf_counter()
    outputs = session.run(None, {"input": input_tensor})
    latency = (time.perf_counter() - start) * 1000
    
    # Post-process
    logits = outputs[0][0]
    probabilities = softmax(logits)
    
    # Top-5 predictions
    top_indices = np.argsort(probabilities)[::-1][:5]
    top_5 = [
        {"class": CLASS_NAMES[i], "confidence": round(float(probabilities[i]), 4)}
        for i in top_indices
    ]
    
    return PredictionResponse(
        predicted_class=CLASS_NAMES[top_indices[0]],
        confidence=round(float(probabilities[top_indices[0]]), 4),
        top_5=top_5,
        latency_ms=round(latency, 2),
    )


@app.post("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(iterations: int = 100):
    """Benchmark inference latency."""
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    dummy = np.random.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        session.run(None, {"input": dummy})
    
    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, {"input": dummy})
        latencies.append((time.perf_counter() - start) * 1000)
    
    latencies.sort()
    mean = sum(latencies) / len(latencies)
    
    return BenchmarkResponse(
        iterations=iterations,
        p50_ms=round(latencies[len(latencies)//2], 2),
        p95_ms=round(latencies[int(len(latencies)*0.95)], 2),
        p99_ms=round(latencies[int(len(latencies)*0.99)], 2),
        mean_ms=round(mean, 2),
        throughput_fps=round(1000 / mean, 1),
    )


# ============================================================
# Run Server
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "inference_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )
