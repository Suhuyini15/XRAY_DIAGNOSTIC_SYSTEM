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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        client.admin.command('ping')
        print("✅ Successfully connected to MongoDB!")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
    
    yield
    
    # Shutdown (if needed)
    # Add any cleanup code here if necessary
    pass

app = FastAPI(title="Authentication API", version="1.0.0", lifespan=lifespan)

# Pydantic models
class UserSignup(BaseModel):
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
    emergency_contact: str

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
    image: Optional[dict] = None  # Will contain filename, content_type, and base64 data
    created_at: str

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
                "doctor_username": current_user["username"]
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
    
    # Handle image upload
    image_data = None
    if image:
        # Validate image file type
        allowed_types = ["image/jpeg", "image/png", "image/jpg"]
        if image.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only JPEG and PNG images are allowed"
            )
        
        # Read and encode image
        image_content = await image.read()
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
                "image": image_data,
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
                "created_at": record["created_at"].isoformat()
            }
            for record in records
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving medical records: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)