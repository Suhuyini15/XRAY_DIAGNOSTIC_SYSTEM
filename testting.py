# Add this to your existing FastAPI application

# Add these imports to your existing imports section
from typing import Dict, Any


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


# Add this endpoint to your existing FastAPI routes
@app.post("/ai-agent", response_model=DetailedAIResponse)
async def ai_agent_analysis(
    agent_input: AiAgentInput, current_user: dict = Depends(get_current_user)
):
    """
    AI-powered clinical decision support for respiratory conditions
    """
    try:
        # Prepare patient info for AI agent
        patient_info = {
            "symptoms": agent_input.symptoms,
            "allergies": agent_input.allergies,
            "special_cases": agent_input.special_cases,
        }

        # Run AI agent analysis
        result = run_ai_agent(agent_input.prediction, patient_info)

        return DetailedAIResponse(
            analysis=result.get("analysis", "Analysis not available"),
            structured_recommendations=result.get("structured_recommendations", {}),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in AI agent analysis: {str(e)}",
        )


@app.post("/patient/{patient_id}/ai-analysis", response_model=AIAgentResponse)
async def patient_ai_analysis(
    patient_id: str,
    agent_input: AiAgentInput,
    current_user: dict = Depends(get_current_user),
):
    """
    Generate AI analysis for a specific patient with full context
    """
    # Verify patient exists and belongs to the current doctor
    patient = get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found"
        )

    if patient["doctor_username"] != current_user["username"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only analyze your own patients",
        )

    try:
        # Prepare comprehensive patient info
        patient_info = {
            "name": patient.get("name"),
            "age": patient.get("age"),
            "gender": patient.get("sex"),
            "symptoms": agent_input.symptoms,
            "allergies": agent_input.allergies,
            "special_cases": agent_input.special_cases,
        }

        # Run AI agent analysis
        result = run_ai_agent(agent_input.prediction, patient_info)

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"]
            )

        # Extract structured information
        recommendations = result.get("structured_recommendations", {})
        condition = recommendations.get("condition", agent_input.prediction)

        # Get condition-specific data
        condition_enum = MODEL_TO_ENUM.get(agent_input.prediction.lower())
        if condition_enum and condition_enum in DISEASE_KNOWLEDGE:
            disease_info = DISEASE_KNOWLEDGE[condition_enum]
            treatments = disease_info["treatments"]
            home_advice = disease_info["home_advice"]
            contraindications = list(disease_info["contraindications"].keys())
        else:
            treatments = recommendations.get("treatments", [])
            home_advice = recommendations.get("home_advice", [])
            contraindications = recommendations.get("contraindications", [])

        return AIAgentResponse(
            patient_id=patient_id,
            condition=condition,
            severity_assessment=recommendations.get(
                "quick_assessment", "Assessment required"
            ),
            treatment_recommendations=treatments,
            contraindication_alerts=contraindications,
            monitoring_steps=recommendations.get(
                "monitoring", ["Regular follow-up required"]
            ),
            home_advice=home_advice,
            ai_analysis=result.get("analysis", "Analysis not available"),
            quick_assessment=recommendations.get(
                "quick_assessment", "Assessment required"
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in patient AI analysis: {str(e)}",
        )
