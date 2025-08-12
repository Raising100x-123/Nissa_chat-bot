# sample_doctors.py

from datetime import datetime, timedelta
from doctor_manager import DoctorManager

# MongoDB connection details
MONGODB_URI = "mongodb+srv://raising100x:vNb3t4WLQZKMN2OZ@cluster0.v5haryq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "Gemini"

# Initialize doctor manager
manager = DoctorManager(MONGODB_URI, DB_NAME)

def generate_slots():
    """Generate daily appointment slots from 10:00 AM to 6:00 PM, skipping 1 PM."""
    slots = []
    start = datetime.strptime("10:00", "%H:%M")
    end = datetime.strptime("18:00", "%H:%M")

    while start < end:
        if start.hour == 13:  # Skip 1 PM for lunch
            start += timedelta(minutes=60)
            continue
        slots.append({
            "time": start.strftime("%I:%M %p"),
            "status": "available"
        })
        start += timedelta(minutes=30)
    return slots

# Sample doctors data
sample_doctors = [
    {"name": "Dr. Aarav Mehta", "specialization": "Cardiologist", "experience": "15 years", "qualification": "MD Cardiology", "availability": "Mon-Fri", "fees": "₹1500", "daily_slots": generate_slots()},
    {"name": "Dr. Nisha Reddy", "specialization": "Dermatologist", "experience": "12 years", "qualification": "MD Dermatology", "availability": "Mon-Sat", "fees": "₹1200", "daily_slots": generate_slots()},
    {"name": "Dr. Kabir Singh", "specialization": "Orthopedic Surgeon", "experience": "18 years", "qualification": "MS Orthopedics", "availability": "Mon-Fri", "fees": "₹2000", "daily_slots": generate_slots()},
    {"name": "Dr. Rani Sharma", "specialization": "Neurologist", "experience": "20 years", "qualification": "DM Neurology", "availability": "Tue-Sat", "fees": "₹2500", "daily_slots": generate_slots()},
    {"name": "Dr. Aditya Verma", "specialization": "ENT Specialist", "experience": "10 years", "qualification": "MS ENT", "availability": "Mon-Fri", "fees": "₹1000", "daily_slots": generate_slots()},
    {"name": "Dr. Meera Das", "specialization": "Gynecologist", "experience": "14 years", "qualification": "MD Gynecology", "availability": "Mon-Sat", "fees": "₹1300", "daily_slots": generate_slots()},
    {"name": "Dr. Sameer Ali", "specialization": "Pediatrician", "experience": "16 years", "qualification": "MD Pediatrics", "availability": "Mon-Sun", "fees": "₹1100", "daily_slots": generate_slots()}
]

def setup_doctors():
    """Clear old doctors and insert sample doctors with slots."""
    # Clear existing doctors
    manager.doctor_collection.delete_many({})
    print("Old doctors data cleared.")

    # Insert sample doctors
    for doctor in sample_doctors:
        try:
            result = manager.add_doctor(doctor)
            print(f"Added doctor: {doctor['name']} - ID: {result.inserted_id}")
        except Exception as e:
            print(f"Error adding {doctor['name']}: {str(e)}")

    total = len(manager.get_all_doctors())
    print(f"\nTotal doctors in database: {total}")

if __name__ == "__main__":
    setup_doctors()
