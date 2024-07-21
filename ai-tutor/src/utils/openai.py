import openai

openai.api_key = 'API_KEY'

def generate_feedback(student_response, expected_answer):
    prompt = f"Student's response: {student_response}\nExpected answer: {expected_answer}\nProvide feedback:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

def get_student_history(student_id):
    # Fetch student history from database or records
    history = {
        "weak_areas": ["algebra", "geometry"],
        "performance_trend": "improving",
        "previous_feedback": ["Work on your algebra skills.", "Great improvement in geometry."]
    }
    return history

def generate_personalized_feedback(student_response, expected_answer, student_history):
    prompt = f"Student's response: {student_response}\nExpected answer: {expected_answer}\nStudent history: {student_history}\nProvide personalized feedback:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()
