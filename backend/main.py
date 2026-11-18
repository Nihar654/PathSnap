from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import csv, io, os

app = FastAPI(title="PathSnap API", version="1.0.0",
 description="Evaluator API. Connect your team's real classifier and HMM/Viterbi before enabling predictions.")
origins=os.getenv("FRONTEND_ORIGINS","http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in origins],
 allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class PredictionRequest(BaseModel):
    trip_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    use_hmm: bool = True

@app.get("/")
def root():
    return {"service":"PathSnap API","docs":"/docs"}

@app.get("/api/health")
def health():
    return {"status":"ok","prediction_ready":False,
            "detail":"API is running; connect the team's trained model to enable predictions."}

@app.get("/api/models")
def models():
    return {"models":["Logistic Regression","KNN","Random Forest","LightGBM","XGBoost"]}

@app.post("/api/preview-track")
async def preview_track(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400,"Upload a CSV with latitude and longitude columns.")
    try:
        reader=csv.DictReader(io.StringIO((await file.read()).decode("utf-8-sig")))
        if not reader.fieldnames: raise ValueError("CSV has no header row")
        keys={x.strip().lower():x for x in reader.fieldnames}
        lat=next((keys[k] for k in ("latitude","lat") if k in keys),None)
        lon=next((keys[k] for k in ("longitude","lon","lng") if k in keys),None)
        if not lat or not lon: raise ValueError("Required columns: latitude, longitude (or lat, lon)")
        pts=[]
        for row in reader:
            a,b=float(row[lat]),float(row[lon])
            if -90<=a<=90 and -180<=b<=180: pts.append({"lat":a,"lon":b})
        if not pts: raise ValueError("No valid GPS points found")
        return {"filename":file.filename,"point_count":len(pts),"points":pts,
                "preview_only":True,"message":"GPS preview only; no model prediction was performed."}
    except (UnicodeDecodeError,ValueError,TypeError) as e:
        raise HTTPException(400,str(e))

@app.post("/api/predict")
def predict(req: PredictionRequest):
    # Fail closed: never fabricate route predictions.
    raise HTTPException(501,detail={
      "message":"Prediction pipeline is not connected yet.",
      "required":["Load the team's trained classifier","Call the team's real feature engineering",
                  "Generate candidate road segments","Optionally run the team's HMM/Viterbi decoder"]})
