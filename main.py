from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    status,
    File,
    UploadFile,
    Form,
    Body,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
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
from typing import Dict, Any

# MongoDB connection
uri = "mongodb+srv://suhuyiniyahaya09:zqm2IS0lUD54IUJq@cluster1.tv4stmr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1"
client = MongoClient(uri, server_api=ServerApi("1"))

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
    global model, llm
    try:
        client.admin.command("ping")
        print("✅ Successfully connected to MongoDB!")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")

    # Load TensorFlow model
    try:
        # Loading tensorflow model
        # model = tf.keras.models.load_model(MODEL_PATH)
        model = tf.saved_model.load(MODEL_PATH)
        print("✅ Successfully loaded tensorflow model!")

        # Load the model
        llm = Llama(model_path="./llama-2-7b-chat.Q4_K_M.gguf", n_ctx=2048)
        print("✅ Successfully loaded LLM model!")

    except Exception as e:
        print(f"❌ Failed to load TensorFlow model: {e}")
        print(
            "Note: Make sure the 'converted_model' directory exists in your project root"
        )

    yield

    # Shutdown (if needed)
    # Add any cleanup code here if necessary
    pass


app = FastAPI(title="Authentication API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Add these enum and constants after your existing models
class RespiratoryCondition(Enum):
    ASTHMA = "asthma"
    PNEUMONIA = "pneumonia"
    LUNG_CANCER = "lung cancer"
    TUBERCULOSIS = "tuberculosis"
    PNEUMOTHORAX = "pneumothorax"
    COPD = "chronic obstructive pulmonary disease"


MODEL_TO_ENUM = {
    "pneumonia": RespiratoryCondition.PNEUMONIA,
    "pneumothorax": RespiratoryCondition.PNEUMOTHORAX,
    "copd": RespiratoryCondition.COPD,
    "mass": RespiratoryCondition.LUNG_CANCER,
    "effusion": RespiratoryCondition.PNEUMONIA,
    "uninfected": None,  # Handle normal cases
}

DISEASE_KNOWLEDGE = {
    RespiratoryCondition.ASTHMA: {
        "treatments": [
            "Inhaled corticosteroids",
            "Beta-agonists",
            "Leukotriene modifiers",
        ],
        "contraindications": {
            "Aspirin allergy": ["Avoid NSAIDs", "Use alternative anti-inflammatory"],
            "Beta-blocker sensitivity": ["Avoid non-selective beta-blockers"],
        },
        "common_symptoms": [
            "wheezing",
            "shortness of breath",
            "chest tightness",
            "coughing",
        ],
        "severity_indicators": [
            "difficulty speaking",
            "blue lips",
            "rapid breathing",
            "chest retractions",
        ],
        "home_advice": [
            "Identify and avoid asthma triggers such as dust, smoke, and allergens.",
            "Practice controlled breathing exercises to reduce symptoms.",
            "Take prescribed inhalers regularly, even when symptoms improve.",
            "Keep a rescue inhaler (short-acting bronchodilator) available at all times.",
            "Stay physically active with low-impact exercises.",
            "Maintain a healthy weight to reduce pressure on lungs.",
        ],
    },
    RespiratoryCondition.PNEUMONIA: {
        "treatments": [
            "Antibiotics",
            "Supportive care",
            "Oxygen therapy",
            "Rest and fluids",
        ],
        "contraindications": {
            "Antibiotic allergy": ["Alternative antibiotic selection required"],
            "Kidney disease": ["Adjust antibiotic dosing"],
        },
        "common_symptoms": [
            "fever",
            "cough with phlegm",
            "chest pain",
            "fatigue",
            "chills",
        ],
        "severity_indicators": [
            "confusion",
            "low blood pressure",
            "high heart rate",
            "low oxygen",
        ],
        "home_advice": [
            "Take the full course of antibiotics as prescribed, even if symptoms improve.",
            "Stay hydrated by drinking plenty of fluids to thin mucus.",
            "Rest adequately and avoid strenuous physical activity during recovery.",
            "Use a humidifier or breathe warm steam to ease breathing discomfort.",
        ],
    },
    RespiratoryCondition.LUNG_CANCER: {
        "treatments": [
            "Chemotherapy",
            "Radiation therapy",
            "Surgical resection",
            "Targeted therapy",
            "Immunotherapy",
        ],
        "contraindications": {
            "Liver impairment": ["Adjust or avoid certain chemotherapeutic agents"],
            "Low white blood cell count": [
                "Delay chemotherapy or use growth factor support"
            ],
            "Autoimmune disease": [
                "Use caution with immunotherapy, may trigger flare-ups"
            ],
        },
        "common_symptoms": [
            "persistent cough",
            "blood in sputum",
            "chest pain",
            "weight loss",
            "hoarseness",
        ],
        "severity_indicators": [
            "difficulty breathing",
            "bone pain",
            "headaches",
            "swelling",
        ],
        "home_advice": [
            "Get adequate rest and manage fatigue with planned rest periods.",
            "Practice stress-reducing activities like meditation or breathing exercises.",
            "Keep all scheduled chemotherapy, radiation, or surgical appointments.",
            "Monitor and report new symptoms such as coughing up blood, worsening pain, or breathing difficulty.",
            "Avoid exposure to lung irritants like dust, fumes, or strong chemicals.",
        ],
    },
    RespiratoryCondition.TUBERCULOSIS: {
        "treatments": [
            "Isoniazid",
            "Rifampin",
            "Ethambutol",
            "Pyrazinamide",
            "Supportive care",
        ],
        "contraindications": {
            "Liver disease": [
                "Avoid or adjust Isoniazid and Rifampin due to hepatotoxicity"
            ],
            "Optic neuritis": ["Avoid Ethambutol due to risk of vision damage"],
            "Pregnancy": [
                "Use drugs cautiously, avoid Pyrazinamide unless benefits outweigh risks"
            ],
        },
        "common_symptoms": [
            "persistent cough",
            "night sweats",
            "fever",
            "weight loss",
            "fatigue",
        ],
        "severity_indicators": ["coughing blood", "chest pain", "difficulty breathing"],
        "home_advice": [
            "Complete the entire treatment course, even if symptoms improve.",
            "Cover your mouth when coughing or sneezing to prevent spreading TB to others.",
            "Wear a mask in public and ensure proper ventilation at home.",
            "Eat a healthy diet rich in protein and vitamins to boost immunity.",
        ],
    },
    RespiratoryCondition.PNEUMOTHORAX: {
        "treatments": [
            "Oxygen therapy",
            "Needle decompression",
            "Chest tube insertion",
            "Surgical repair if recurrent",
        ],
        "contraindications": {
            "Severe COPD": [
                "Caution with positive-pressure ventilation due to risk of worsening air leak"
            ],
            "Bleeding disorders": [
                "Use caution with invasive procedures like chest tube placement"
            ],
        },
        "common_symptoms": [
            "sudden chest pain",
            "shortness of breath",
            "rapid heart rate",
            "dry cough",
        ],
        "severity_indicators": [
            "severe breathing difficulty",
            "low blood pressure",
            "cyanosis",
            "shock",
        ],
        "home_advice": [
            "Avoid heavy lifting, intense exercise, or sudden chest strain during recovery.",
            "Do not travel by airplane until your doctor confirms it is safe.",
            "Avoid scuba diving and high-altitude activities permanently unless cleared by a specialist.",
            "Report any sudden chest pain, shortness of breath, or worsening symptoms immediately.",
            "Practice gentle breathing exercises to improve lung capacity once cleared.",
        ],
    },
    RespiratoryCondition.COPD: {
        "treatments": [
            "Bronchodilators",
            "Combination inhalers",
            "Oxygen therapy",
            "Pulmonary rehabilitation",
            "Surgical options (Lung volume reduction surgery, Lung transplant)",
        ],
        "contraindications": {
            "Beta-blocker use": [
                "Can worsen bronchospasm, use cardioselective beta-blockers cautiously"
            ],
            "Uncontrolled glaucoma": [
                "Avoid anticholinergic bronchodilators like Ipratropium"
            ],
            "Severe liver disease": ["Caution with Roflumilast"],
            "Active respiratory infection": [
                "Delay elective surgeries and some inhaled steroids until controlled"
            ],
        },
        "common_symptoms": [
            "Chronic cough",
            "Excess sputum production",
            "Shortness of breath (especially during physical activity)",
            "Wheezing",
            "Chest tightness",
            "Fatigue",
            "Frequent respiratory infections",
        ],
        "severity_indicators": [
            "Severe shortness of breath at rest",
            "Cyanosis (bluish lips or fingertips)",
            "Confusion or mental status changes",
            "Swelling in ankles, feet, or legs (sign of right heart failure)",
            "Rapid worsening of symptoms (COPD exacerbation)",
            "Very low oxygen levels (hypoxemia)",
        ],
        "home_advice": [
            "Quit smoking completely and avoid secondhand smoke.",
            "Practice pursed-lip breathing to control shortness of breath.",
            "Stay hydrated to keep mucus thin and easier to clear.",
            "Eat small, frequent, high-calorie meals to maintain energy.",
            "Exercise gently (e.g., walking or light stretching) as tolerated.",
        ],
    },
}


# Add response models
class AIAgentResponse(BaseModel):
    patient_id: str
    condition: str
    severity_assessment: str
    treatment_recommendations: List[str]
    contraindication_alerts: List[str]
    monitoring_steps: List[str]
    home_advice: List[str]
    ai_analysis: str
    quick_assessment: str


class DetailedAIResponse(BaseModel):
    analysis: str
    structured_recommendations: Dict[str, Any]


# Helper functions for AI agent
def create_disease_context(condition):
    """Create context for AI analysis"""
    disease_info = DISEASE_KNOWLEDGE[condition]
    context = f"""You are a clinical decision support AI assisting doctors in managing {condition.value}.
    
Condition: {condition.value}

Treatment options:
{', '.join(disease_info['treatments'])}

Common symptoms: {', '.join(disease_info['common_symptoms'])}
Severe indicators: {', '.join(disease_info['severity_indicators'])}

Contraindications:
"""
    for condition_type, warnings in disease_info["contraindications"].items():
        context += f"\n- {condition_type}: {'; '.join(warnings)}"
    return context


def get_symptom_based_recommendations(condition, symptoms):
    """Get quick recommendations based on symptoms"""
    disease_info = DISEASE_KNOWLEDGE[condition]
    symptoms_lower = (
        [s.lower() for s in symptoms]
        if isinstance(symptoms, list)
        else [symptoms.lower()]
    )

    severe_symptoms = [
        s
        for s in disease_info["severity_indicators"]
        if any(s.lower() in symptom_text for symptom_text in symptoms_lower)
    ]

    if severe_symptoms:
        return "URGENT: Severe symptoms detected. Immediate medical attention required."
    return (
        f"Suggested first-line treatments: {', '.join(disease_info['treatments'][:3])}"
    )


# Pydantic models
class UserSignup(BaseModel):
    name: str
    username: str
    email: str
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


class PatientResponse(BaseModel):
    patient_id: str
    name: str
    age: int
    blood_group: str
    contact: str
    emergency_contact: str
    doctor_username: str


class AiAgentInput(BaseModel):
    patient_id: str
    symptoms: str
    allergies: str
    special_cases: str
    prediction: str


class MedicalRecordResponse(BaseModel):
    record_id: str
    patient_id: str
    symptoms: str
    allergies: str
    special_cases: str
    prediction_results: str
    created_at: str


class PredictionResponse(BaseModel):
    top_class: str
    top_confidence: float
    all_predictions: dict


def run_ai_agent(
    predicted_condition: str, patient_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Main AI agent function integrated with your existing system"""
    global llm

    if llm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI model (LLaMA) is not available",
        )

    # Map model prediction to enum
    condition_enum = MODEL_TO_ENUM.get(predicted_condition.lower())
    if not condition_enum:
        return {
            "error": f"Condition '{predicted_condition}' not recognized by AI agent.",
            "analysis": f"The prediction '{predicted_condition}' is not currently supported by the clinical decision support system.",
        }

    if condition_enum is None:  # uninfected case
        return {
            "analysis": "No significant respiratory pathology detected. Continue routine monitoring and maintain healthy respiratory habits.",
            "structured_recommendations": {
                "severity": "Normal",
                "treatments": ["Routine follow-up", "Preventive care"],
                "contraindications": [],
                "monitoring": ["Annual check-ups"],
                "home_advice": [
                    "Maintain healthy lifestyle",
                    "Avoid smoking",
                    "Regular exercise",
                ],
            },
        }

    # Quick assessment
    symptoms = (
        patient_info.get("symptoms", "").split(",")
        if patient_info.get("symptoms")
        else []
    )
    quick_check = get_symptom_based_recommendations(condition_enum, symptoms)

    # Build AI context
    system_context = create_disease_context(condition_enum)

    # Add treatment & home advice
    disease_info = DISEASE_KNOWLEDGE[condition_enum]
    treatments_text = "\n".join([f"- {t}" for t in disease_info["treatments"]])
    home_advice_text = "\n".join([f"- {ad}" for ad in disease_info["home_advice"]])

    # Parse allergies and special cases
    allergies = (
        patient_info.get("allergies", "").split(",")
        if patient_info.get("allergies")
        else []
    )
    special_cases = patient_info.get("special_cases", "")

    prompt = f"""{system_context}

Patient Info:
- Symptoms: {patient_info.get('symptoms', 'Not specified')}
- Allergies: {', '.join(allergies) if allergies else 'None reported'}
- Special Cases/Medical History: {special_cases if special_cases else 'None reported'}

[Quick Assessment]: {quick_check}

Generate a structured clinical recommendation including:
1. Likely severity assessment (mild/moderate/severe)
2. Evidence-based treatment plan prioritized by effectiveness
3. Specific contraindication alerts based on patient allergies and conditions
4. Monitoring parameters and escalation criteria
5. Patient education and counseling points
6. Detailed home management advice

Available Treatments:
{treatments_text}

Evidence-Based Home Advice:
{home_advice_text}

Provide a comprehensive but concise clinical analysis focused on actionable recommendations.
"""

    try:
        # Call LLaMA model
        response = llm(
            prompt,
            max_tokens=500,
            temperature=0.3,
            stop=["Patient Info:", "Available Treatments:"],
        )
        ai_analysis = response["choices"][0]["text"].strip()

        return {
            "analysis": ai_analysis,
            "structured_recommendations": {
                "condition": condition_enum.value,
                "severity": "To be assessed by physician",
                "treatments": disease_info["treatments"],
                "contraindications": list(disease_info["contraindications"].keys()),
                "monitoring": [
                    "Regular follow-up",
                    "Symptom monitoring",
                    "Treatment response assessment",
                ],
                "home_advice": disease_info["home_advice"],
                "quick_assessment": quick_check,
            },
        }

    except Exception as e:
        return {
            "error": f"AI analysis failed: {str(e)}",
            "analysis": f"Unable to generate AI analysis. Manual clinical assessment required for {condition_enum.value}.",
            "structured_recommendations": {
                "condition": condition_enum.value,
                "treatments": disease_info["treatments"],
                "home_advice": disease_info["home_advice"],
            },
        }


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
        if image.mode != "RGB":
            image = image.convert("RGB")

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
            detail=f"Error processing image: {str(e)}",
        )


def predict_disease_from_bytes(image_bytes):
    """Predict disease from image bytes"""
    global model

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI model is not available",
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
        predictions = {
            CLASS_NAMES[i]: float(preds[i] * 100) for i in range(len(CLASS_NAMES))
        }
        sorted_predictions = dict(
            sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        )

        top_class = max(predictions, key=predictions.get)
        top_confidence = predictions[top_class]

        return {
            "top_class": top_class,
            "top_confidence": top_confidence,
            "all_predictions": sorted_predictions,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}",
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
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
            detail="Username already registered",
        )

    # Hash the password
    hashed_password = hash_password(user.password)

    # Create user document
    user_doc = {
        "name": user.name,
        "username": user.username,
        "email": user.email,
        "password": hashed_password,
        "license_number": user.license_number,
        "created_at": datetime.utcnow(),
    }

    # Insert user into database
    try:
        result = users_collection.insert_one(user_doc)
        if result.inserted_id:
            return {
                "message": "User created successfully",
                "username": user.username,
                "license_number": user.license_number,
            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}",
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
        "license_number": current_user["license_number"],
    }


@app.post("/addpatientinfo", response_model=PatientResponse)
async def add_patient_info(
    patient: PatientInfo, current_user: dict = Depends(get_current_user)
):
    # Create patient document
    patient_doc = {
        "name": patient.name,
        "age": patient.age,
        "blood_group": patient.blood_group,
        "contact": patient.contact,
        "email": patient.email,
        "emergency_contact_name": patient.emergency_contact_name,
        "sex": patient.sex,
        "emergency_contact": patient.emergency_contact,
        "doctor_username": current_user["username"],
        "doctor_id": str(current_user["_id"]),
        "created_at": datetime.utcnow(),
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
                "email": patient.email,
                "emergency_contact_name": patient.emergency_contact_name,
                "sex": patient.sex,
            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating patient record: {str(e)}",
        )


@app.post("/addmedicalrecord", response_model=MedicalRecordResponse)
async def add_medical_record(
    patient_id: str = Form(...),
    symptoms: str = Form(...),
    allergies: str = Form(...),
    special_cases: str = Form(...),
    prediction_results: str = Form(...),
    current_user: dict = Depends(get_current_user),
):
    # Verify patient exists and belongs to the current doctor
    patient = get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found"
        )

    if patient["doctor_username"] != current_user["username"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only add medical records for your own patients",
        )

    # Create medical record document
    record_doc = {
        "patient_id": patient_id,
        "symptoms": symptoms,
        "allergies": allergies,
        "special_cases": special_cases,
        "doctor_username": current_user["username"],
        "doctor_id": str(current_user["_id"]),
        "prediction": prediction_results,
        "created_at": datetime.utcnow(),
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
                "prediction": prediction_results,
                "created_at": record_doc["created_at"].isoformat(),
            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating medical record: {str(e)}",
        )


@app.get("/patients", response_model=List[PatientResponse])
async def get_my_patients(current_user: dict = Depends(get_current_user)):
    """Get all patients for the current doctor"""
    try:
        patients = list(
            patients_collection.find({"doctor_username": current_user["username"]})
        )
        return [
            {
                "patient_id": str(patient["_id"]),
                "name": patient["name"],
                "age": patient["age"],
                "blood_group": patient["blood_group"],
                "contact": patient["contact"],
                "emergency_contact": patient["emergency_contact"],
                "doctor_username": patient["doctor_username"],
            }
            for patient in patients
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving patients: {str(e)}",
        )


#  response_model=List[MedicalRecordResponse]
@app.get(
    "/patient/{patient_id}/records",
)
async def get_patient_records(
    patient_id: str, current_user: dict = Depends(get_current_user)
):
    """Get all medical records for a specific patient"""
    # Verify patient belongs to current doctor
    patient = get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found"
        )

    if patient["doctor_username"] != current_user["username"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view records for your own patients",
        )

    try:
        # Since patient is a single record, return it as a list with one item
        print(patient)

        # Safely extract the first image if available
        images = patient.get("image") or []
        image = images[0] if len(images) > 0 else None

        return [
            {
                "patient_id": str(patient.get("_id")),
                "name": patient.get("name"),
                "age": patient.get("age"),
                "blood_group": patient.get("blood_group"),
                "contact": patient.get("contact"),
                "email": patient.get("email"),
                "sex": patient.get("sex"),
                "emergency_contact_name": patient.get("emergency_contact_name"),
                "emergency_contact": patient.get("emergency_contact"),
                "doctor_username": patient.get("doctor_username"),
                "created_at": patient.get("created_at"),
                "symptoms": patient.get("symptoms"),
                "allergies": patient.get("allergies"),
                "special_cases": patient.get("special_cases"),
                "image_filename": image.get("filename") if image else None,
                "image_content_type": image.get("content_type") if image else None,
                "image_data": image.get("image_data") if image else None,
                "prediction": patient.get("prediction"),
                "analysis": patient.get("analysis", ""),
                "severity": patient.get("severity", ""),
                "treatments": patient.get("treatments", []),
                "monitoring": patient.get("monitoring", []),
                "home_advice": patient.get("home_advice", []),
                "quick_assessment": patient.get("quick_assessment", ""),
            }
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving medical records: {str(e)}",
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict_disease_endpoint(
    image: UploadFile = File(...), current_user: dict = Depends(get_current_user)
):
    """Standalone endpoint for disease prediction from medical images"""
    # Validate image file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPEG and PNG images are allowed",
        )

    # Read image content
    image_content = await image.read()

    # Make prediction
    prediction_result = predict_disease_from_bytes(image_content)

    return prediction_result


@app.post("/patient/{patient_id}/records/images")
async def save_image(
    patient_id: str,
    image: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Standalone endpoint for saving patient image + disease prediction"""

    # Validate image type
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPEG and PNG images are allowed",
        )

    # Read image content
    image_content = await image.read()

    # Make disease prediction
    prediction_result = predict_disease_from_bytes(image_content)

    # Convert image to base64 for storage
    image_base64 = base64.b64encode(image_content).decode("utf-8")

    # Prepare record
    record = {
        "filename": image.filename,
        "content_type": image.content_type,
        "image_data": image_base64,  # or use GridFS for large images
        "prediction": prediction_result,
        "uploaded_by": current_user["_id"],
    }

    # Save to patient record
    update_result = patients_collection.update_one(
        {"_id": ObjectId(patient_id)}, {"$push": {"image": record}}
    )

    if update_result.modified_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found"
        )

    return {"message": "Image saved successfully", "prediction": prediction_result}


@app.post("/patient/{patient_id}/records/notes")
async def save_patient_notes(
    patient_id: str,
    notes: dict = Body(
        ...,
        example={
            "symptoms": "Headache, fever",
            "allergies": "Penicillin",
            "special_cases": "Diabetes",
        },
    ),
    current_user: dict = Depends(get_current_user),
):
    """Save or update patient notes (symptoms, allergies, special cases)."""

    # Verify patient belongs to current doctor
    patient = get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found"
        )

    if patient["doctor_username"] != current_user["username"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update notes for your own patients",
        )

    # Update notes
    update_data = {
        "symptoms": notes.get("symptoms"),
        "allergies": notes.get("allergies"),
        "special_cases": notes.get("special_cases"),
    }

    result = patients_collection.update_one(
        {"_id": patient["_id"]}, {"$set": update_data}
    )

    if result.modified_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Notes were not updated"
        )

    return {"message": "Patient notes updated successfully", "notes": update_data}


@app.post("/patient/{patient_id}/records/prediction")
async def save_patient_prediction(
    patient_id: str,
    prediction: dict = Body(..., example={"disease": "Pneumonia"}),
    current_user: dict = Depends(get_current_user),
):
    """Save or update disease prediction for a patient"""

    # Verify patient exists and belongs to doctor
    patient = get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found"
        )

    if patient["doctor_username"] != current_user["username"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update predictions for your own patients",
        )

    # Extract disease
    disease_prediction = prediction.get("disease")
    if not disease_prediction:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'disease' is required",
        )

    # Save prediction
    result = patients_collection.update_one(
        {"_id": patient["_id"]}, {"$set": {"prediction": disease_prediction}}
    )

    if result.modified_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Prediction was not updated"
        )

    return {
        "message": "Prediction saved successfully",
        "prediction": disease_prediction,
    }


# Add this endpoint to your existing FastAPI routes
@app.post("/ai-agent", response_model=DetailedAIResponse)
async def ai_agent_analysis(
    agent_input: AiAgentInput, current_user: dict = Depends(get_current_user)
):
    """
    AI-powered clinical decision support for respiratory conditions
    """

    patient = get_patient_by_id(agent_input.patient_id)

    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found"
        )

    if patient["doctor_username"] != current_user["username"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update notes for your own patients",
        )

    try:
        # Prepare patient info for AI agent
        patient_info = {
            "symptoms": agent_input.symptoms,
            "allergies": agent_input.allergies,
            "special_cases": agent_input.special_cases,
        }

        result = patients_collection.update_one(
            {"_id": patient["_id"]}, {"$set": patient_info}
        )

        # Run AI agent analysis
        result = run_ai_agent(agent_input.prediction, patient_info)

        recommendations = result.get("structured_recommendations", {})

        # Prepare detailed AI response
        detailed_ai_results = {
            "analysis": result.get("analysis"),
            "severity": recommendations.get("severity"),
            "prediction": recommendations.get("condition"),
            "treatments": recommendations.get("treatments", []),
            "monitoring": recommendations.get("monitoring", []),
            "home_advice": recommendations.get("home_advice", []),
            "quick_assessment": recommendations.get("quick_assessment"),
        }

        print(detailed_ai_results)

        saved_results = patients_collection.update_one(
            {"_id": patient["_id"]}, {"$set": detailed_ai_results}, upsert=True
        )

        return DetailedAIResponse(
            analysis=result.get("analysis", "Analysis not available"),
            structured_recommendations=result.get("structured_recommendations", {}),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in AI agent analysis: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
