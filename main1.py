from runner import predict_disease
from ai_agent import MODEL_TO_ENUM, run_ai_agent

# Collect patient info (based on fields in test4.py)
patient_info = {
    "name": input("Enter patient name: "),
    "age": int(input("Enter patient age: ")),
    "gender": input("Enter patient gender (Male/Female): "),
    "weight": float(input("Enter patient weight (kg): ")),
    "height": float(input("Enter patient height (cm): ")),
    "symptoms": input("List symptoms (comma-separated): "),
    "medical_history": input("Enter relevant medical history: "),
    "allergies": input("Enter known allergies (if none, type 'None'): "),
    "current_medications": input("Enter current medications: ")
}

# Ask for X-ray image path
image_path = input("Enter path to X-ray image: ")

# Step 1: Model prediction
predicted_condition, confidence, all_predictions = predict_disease(image_path)

print("\n=== Model Predictions ===")
for condition, conf in all_predictions.items():
    print(f"{condition}: {conf:.2f}%")

print(f"\nTop Prediction: {predicted_condition} ({confidence:.2f}% confidence)")

# Step 2: Pass to AI agent if abnormal
if predicted_condition.lower() == "uninfected":
    print("\nNo abnormality detected. AI agent will not run.")
else:
    condition_enum = MODEL_TO_ENUM.get(predicted_condition.lower())
    if condition_enum:
        recommendation = run_ai_agent(predicted_condition, patient_info)
        print("\n=== AI Agent Recommendation ===")
        print(recommendation)
    else:
        print(f"Condition '{predicted_condition}' not recognized by AI agent.")
