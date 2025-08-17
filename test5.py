from llama_cpp import Llama
from enum import Enum
import tensorflow as tf
import numpy as np

# CNN MODEL
model = tf.keras.models.load_model("xrAy_model.h5")

def preprocess_xray(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_disease(image_path):
    img_array = preprocess_xray(image_path)
    preds = model.predict(img_array)
    predicted_index = np.argmax(preds)
    classes = ["uninfected", "pneumonia", "pneumothorax", "mass", "effusion", "copd"]
    return classes[predicted_index]

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
    "mass": RespiratoryCondition.LUNG_CANCER,     # mapped, but will add disclaimer
    "effusion": RespiratoryCondition.PNEUMONIA,   # mapped, but will add disclaimer
}

llm = Llama(model_path="./llama-2-7b-chat.Q4_K_M.gguf", n_ctx=2048)

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
        ]
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
            "Beta-blocker use": ["Can worsen bronchospasm, use cardioselective beta-blockers cautiously"],
            "Uncontrolled glaucoma": ["Avoid anticholinergic bronchodilators like Ipratropium"],
            "Severe liver disease": ["Caution with Roflumilast"],
            "Active respiratory infection": ["Delay elective surgeries and some inhaled steroids until controlled"],
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
        "home_advice":[
            "Quit smoking completely and avoid secondhand smoke.",
            "Practice pursed-lip breathing to control shortness of breath.",
            "Stay hydrated to keep mucus thin and easier to clear.",
            "Eat small, frequent, high-calorie meals to maintain energy.",
            "Exercise gently (e.g., walking or light stretching) as tolerated.",
        ],
    },
}

# Conversation config
MAX_HISTORY_TOKENS = 1500
conversation_history = []

# === New Function: Patient Intake Form ===
def get_patient_details():
    print("\n--- Patient Intake Form ---")
    patient_id = input("Enter Patient ID: ").strip()
    name = input("Enter Patient Name: ").strip()
    age = input("Enter Age: ").strip()

    image_path = input("Enter path to patient's X-ray image: ").strip()
    predicted_condition_str = predict_disease(image_path)
    print(f"Predicted condition from X-ray: {predicted_condition_str}")

    if predicted_condition_str == "uninfected":
        print("No significant abnormality detected. AI agent will not proceed with recommendations.")
        exit()

    condition_input = MODEL_TO_ENUM.get(predicted_condition_str)
    if condition_input is None:
        raise ValueError(f"Predicted condition '{predicted_condition_str}' not recognized in system.")

    # Symptoms and history
    symptoms = input("Enter main symptoms (comma-separated): ").split(",")
    history = input("Enter relevant medical history or medications: ").strip()
    allergies = input("Enter patient allergies or special complications if any: ").strip()

    patient_record = {
        "id": patient_id,
        "name": name,
        "age": age,
        "condition": condition_input,
        "symptoms": [s.strip() for s in symptoms],
        "history": history,
        "allergies": [d.strip() for d in allergies]
    }

    # Disclaimers for indirect mappings
    if predicted_condition_str in ["mass", "effusion"]:
        print(f"\n⚠ Disclaimer: The detected '{predicted_condition_str}' is mapped to {condition_input.value} "
              f"for recommendation purposes. Further clinical evaluation is required to confirm diagnosis.\n")

    print("\nPatient details captured successfully.\n")
    return patient_record

def create_disease_context(condition):
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
    disease_info = DISEASE_KNOWLEDGE[condition]
    severe_symptoms = [
        s for s in disease_info["severity_indicators"]
        if any(s.lower() in symptom.lower() for symptom in symptoms)
    ]
    if severe_symptoms:
        return "URGENT: Severe symptoms detected. Immediate medical attention required."
    return f"Suggested first-line treatments: {', '.join(disease_info['treatments'][:3])}"


# === Main Workflow ===
print("\n=== Respiratory Clinical Support System ===\n")
patient = get_patient_details()

quick_check = get_symptom_based_recommendations(patient["condition"], patient["symptoms"])
print(f"\n[Quick Assessment]: {quick_check}\n")

# Build AI recommendation
system_context = create_disease_context(patient["condition"])
prompt = f"""{system_context}

Patient Info:
- Name: {patient['name']} (ID: {patient['id']}, Age: {patient['age']})
- Symptoms: {', '.join(patient['symptoms'])}
- Medical History: {patient['history']}

Generate a structured recommendation including:
1. Likely severity (mild/moderate/severe)
2. Treatment plan from evidence-based list
3. Contraindication alerts
4. Monitoring/escalation steps
5. Patient counseling & home advice
"""

response = llm(prompt, max_tokens=400, temperature=0.6)
print("\n--- Clinical Support AI Recommendation ---\n")
print(response["choices"][0]["text"].strip())
print("\n-------------------------------------------")