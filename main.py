from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from bson import ObjectId
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import os
import base64
from typing import List
from contextlib import asynccontextmanager
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
from llama_cpp import Llama
from enum import Enum


# MongoDB connection
uri = "mongodb+srv://suhuyiniyahaya09:zqm2IS0lUD54IUJq@cluster1.tv4stmr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1"
client = MongoClient(uri, server_api=ServerApi('1'))

# Database and collections
db = client.auth_db
users_collection = db.users
patients_collection = db.patients
medical_records_collection = db.medical_records

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = "your-secret-key-here"  # Change this to a secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Security
security = HTTPBearer()

# TensorFlow Model Settings
MODEL_PATH = "converted_model"  # Path to your trained model
CLASS_NAMES = ["uninfected", "pneumonia", "pneumothorax", "mass", "effusion", "copd"]
model = None  # Will be loaded in lifespan function
llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model
    try:
        client.admin.command('ping')
        print("✅ Successfully connected to MongoDB!")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
    
    # Load TensorFlow model
    try:
        # Loading tensorflow model
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Load the model
        llm = Llama(model_path="./llama-2-7b-chat.Q4_K_M.gguf", n_ctx=2048)
        print("✅ Successfully loaded TensorFlow model!")
        
    except Exception as e:
        print(f"❌ Failed to load TensorFlow model: {e}")
        print("Note: Make sure the 'converted_model' directory exists in your project root")
    
    yield
    
    # Shutdown (if needed)
    # Add any cleanup code here if necessary
    pass

app = FastAPI(title="Authentication API", version="1.0.0", lifespan=lifespan)

# Pydantic models
class UserSignup(BaseModel):
    name: str
    username: str
    password: str
    license_number: str

class UserSignin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class UserResponse(BaseModel):
    username: str
    license_number: str

class PatientInfo(BaseModel):
    name: str
    age: int
    blood_group: str
    contact: str
    emergency_contact_name: str
    emergency_contact: str
    sex: str
    email: str
    address: str

class PatientResponse(BaseModel):
    patient_id: str
    name: str
    age: int
    blood_group: str
    contact: str
    emergency_contact: str
    doctor_username: str

class MedicalRecordResponse(BaseModel):
    record_id: str
    patient_id: str
    symptoms: str
    allergies: str
    special_cases: str
    has_image: bool
    # image: Optional[dict] = None  # Will contain filename, content_type, and base64 data
    prediction: Optional[dict] = None  # Will contain AI prediction results
    created_at: str

class PredictionResponse(BaseModel):
    top_class: str
    top_confidence: float
    all_predictions: dict

# Utility functions
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user(username: str):
    user = users_collection.find_one({"username": username})
    return user

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user["password"]):
        return False
    return user

def get_patient_by_id(patient_id: str):
    try:
        patient = patients_collection.find_one({"_id": ObjectId(patient_id)})
        return patient
    except:
        return None

def preprocess_image_from_bytes(image_bytes):
    """Preprocess image from bytes for TensorFlow prediction"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((224, 224))
        
        # Convert to array and normalize
        img_array = np.array(image, dtype=np.float32)  # Explicitly use float32
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # This will maintain float32
        
        return img_array
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing image: {str(e)}"
        )

def predict_disease_from_bytes(image_bytes):
    """Predict disease from image bytes"""
    global model
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI model is not available"
        )
    
    try:
        img_array = preprocess_image_from_bytes(image_bytes)
        
        # Make prediction
        try:
            # Normal Keras model
            preds = model.predict(img_array)[0]
        except AttributeError:
            # Handle SavedModel (_UserObject)
            infer = model.signatures["serving_default"]
            outputs = infer(tf.constant(img_array))
            # Get the first tensor in the output dict
            preds = list(outputs.values())[0].numpy()[0]
        
        # Convert predictions to percentages
        predictions = {CLASS_NAMES[i]: float(preds[i] * 100) for i in range(len(CLASS_NAMES))}
        sorted_predictions = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))
        
        top_class = max(predictions, key=predictions.get)
        top_confidence = predictions[top_class]
        
        return {
            "top_class": top_class,
            "top_confidence": top_confidence,
            "all_predictions": sorted_predictions
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}"
        )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user(username=username)
    if user is None:
        raise credentials_exception
    return user

# API Routes
@app.get("/")
async def root():
    return {"message": "Authentication API is running"}

@app.post("/signup", response_model=dict)
async def signup(user: UserSignup):
    # Check if user already exists
    if get_user(user.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Hash the password
    hashed_password = hash_password(user.password)
    
    # Create user document
    user_doc = {
        "name": user.name,
        "username": user.username,
        "password": hashed_password,
        "license_number": user.license_number,
        "created_at": datetime.utcnow()
    }
    
    # Insert user into database
    try:
        result = users_collection.insert_one(user_doc)
        if result.inserted_id:
            return {
                "message": "User created successfully",
                "username": user.username,
                "license_number": user.license_number
            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}"
        )

@app.post("/signin", response_model=Token)
async def signin(user: UserSignin):
    # Authenticate user
    authenticated_user = authenticate_user(user.username, user.password)
    if not authenticated_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": authenticated_user["username"]}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/profile", response_model=UserResponse)
async def get_profile(current_user: dict = Depends(get_current_user)):
    return {
        "username": current_user["username"],
        "license_number": current_user["license_number"]
    }

@app.post("/addpatientinfo", response_model=PatientResponse)
async def add_patient_info(patient: PatientInfo, current_user: dict = Depends(get_current_user)):
    # Create patient document
    patient_doc = {
        "name": patient.name,
        "age": patient.age,
        "blood_group": patient.blood_group,
        "contact": patient.contact,
        "email":patient.email,
        "emergency_contact_name":patient.emergency_contact_name,
        "sex":patient.sex,
        "address":patient.address,
        "emergency_contact": patient.emergency_contact,
        "doctor_username": current_user["username"],
        "doctor_id": str(current_user["_id"]),
        "created_at": datetime.utcnow()
    }
    
    try:
        result = patients_collection.insert_one(patient_doc)
        if result.inserted_id:
            return {
                "patient_id": str(result.inserted_id),
                "name": patient.name,
                "age": patient.age,
                "blood_group": patient.blood_group,
                "contact": patient.contact,
                "emergency_contact": patient.emergency_contact,
                "doctor_username": current_user["username"],
                "email":patient.email,
                "emergency_contact_name":patient.emergency_contact_name,
                "sex":patient.sex,
                "address":patient.address,

            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating patient record: {str(e)}"
        )

@app.post("/addmedicalrecord", response_model=MedicalRecordResponse)
async def add_medical_record(
    patient_id: str = Form(...),
    symptoms: str = Form(...),
    allergies: str = Form(...),
    special_cases: str = Form(...),
    image: UploadFile = File(None),
    current_user: dict = Depends(get_current_user)
):
    # Verify patient exists and belongs to the current doctor
    patient = get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    if patient["doctor_username"] != current_user["username"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only add medical records for your own patients"
        )
    
    # Handle image upload and AI prediction
    image_data = None
    prediction_result = None
    
    if image:
        # Validate image file type
        allowed_types = ["image/jpeg", "image/png", "image/jpg"]
        if image.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only JPEG and PNG images are allowed"
            )
        
        # Read image content
        image_content = await image.read()
        
        # Make AI prediction
        try:
            prediction_result = predict_disease_from_bytes(image_content)
        except Exception as e:
            # Log the error but don't fail the entire request
            print(f"AI prediction failed: {str(e)}")
            prediction_result = None
        
        # Store image data
        image_data = {
            "filename": image.filename,
            "content_type": image.content_type,
            "data": base64.b64encode(image_content).decode('utf-8')
        }
    
    # Create medical record document
    record_doc = {
        "patient_id": patient_id,
        "symptoms": symptoms,
        "allergies": allergies,
        "special_cases": special_cases,
        "image": image_data,
        "prediction": prediction_result,
        "doctor_username": current_user["username"],
        "doctor_id": str(current_user["_id"]),
        "created_at": datetime.utcnow()
    }
    
    try:
        result = medical_records_collection.insert_one(record_doc)
        if result.inserted_id:
            return {
                "record_id": str(result.inserted_id),
                "patient_id": patient_id,
                "symptoms": symptoms,
                "allergies": allergies,
                "special_cases": special_cases,
                "has_image": image is not None,
                "prediction": prediction_result['top_class'],
                "created_at": record_doc["created_at"].isoformat()
            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating medical record: {str(e)}"
        )

@app.get("/patients", response_model=List[PatientResponse])
async def get_my_patients(current_user: dict = Depends(get_current_user)):
    """Get all patients for the current doctor"""
    try:
        patients = list(patients_collection.find({"doctor_username": current_user["username"]}))
        return [
            {
                "patient_id": str(patient["_id"]),
                "name": patient["name"],
                "age": patient["age"],
                "blood_group": patient["blood_group"],
                "contact": patient["contact"],
                "emergency_contact": patient["emergency_contact"],
                "doctor_username": patient["doctor_username"]
            }
            for patient in patients
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving patients: {str(e)}"
        )

@app.get("/patient/{patient_id}/records", response_model=List[MedicalRecordResponse])
async def get_patient_records(patient_id: str, current_user: dict = Depends(get_current_user)):
    """Get all medical records for a specific patient"""
    # Verify patient belongs to current doctor
    patient = get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    if patient["doctor_username"] != current_user["username"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view records for your own patients"
        )
    
    try:
        records = list(medical_records_collection.find({"patient_id": patient_id}))
        return [
            {
                "record_id": str(record["_id"]),
                "patient_id": record["patient_id"],
                "symptoms": record["symptoms"],
                "allergies": record["allergies"],
                "special_cases": record["special_cases"],
                "has_image": record["image"] is not None,
                "image": record["image"],
                "prediction": record.get("prediction"),
                "created_at": record["created_at"].isoformat()
            }
            for record in records
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving medical records: {str(e)}"
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict_disease_endpoint(
    image: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Standalone endpoint for disease prediction from medical images"""
    # Validate image file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPEG and PNG images are allowed"
        )
    
    # Read image content
    image_content = await image.read()
    
    # Make prediction
    prediction_result = predict_disease_from_bytes(image_content)
    
    return prediction_result

@app.post("/testrespons")
async def model_test(
    image: UploadFile = File(None)
):
    global model
    
    if not image:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image provided"
        )
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI model is not available"
        )
    
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPEG and PNG images are allowed"
        )
    
    # Read image content
    image_content = await image.read()
    
    try:
        preprocessed_image = preprocess_image_from_bytes(image_content)
        
        # Handle both Keras models and SavedModels
        try:
            # Try normal Keras model first
            preds = model.predict(preprocessed_image)[0]
        except AttributeError:
            # Handle SavedModel (_UserObject)
            infer = model.signatures["serving_default"]
            # Ensure input is float32
            input_tensor = tf.constant(preprocessed_image, dtype=tf.float32)
            outputs = infer(input_tensor)
            # Get the first tensor in the output dict
            preds = list(outputs.values())[0].numpy()[0]
        
        # Convert to regular Python list for JSON serialization
        preds_list = preds.tolist() if hasattr(preds, 'tolist') else list(preds)
        
        # Create predictions dictionary
        predictions = {CLASS_NAMES[i]: float(preds[i] * 100) for i in range(len(CLASS_NAMES))}
        
        return {
            "raw_predictions": preds_list,
            "class_predictions": predictions,
            "model_type": "SavedModel" if hasattr(model, 'signatures') else "Keras"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image or making prediction: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)