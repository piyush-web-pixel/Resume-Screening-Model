import streamlit as st
import joblib
import re
from io import BytesIO
from PyPDF2 import PdfReader
import docx

# -----------------------------
# 1️⃣ Load the saved model files
# -----------------------------
model = joblib.load("nb_model.pkl")
columns = joblib.load("nb_columns.pkl")
vectorizer = joblib.load("nb_vectorization.pkl")

# -----------------------------
# 2️⃣ Define your label mapping
# -----------------------------
label_map = {
    0: "Data Science",
    1: "HR",
    2: "Advocate",
    3: "Arts",
    4: "Web Designing",
    5: "Mechanical Engineer",
    6: "Sales",
    7: "Health and Fitness",
    8: "Civil Engineer",
    9: "Java Developer",
    10: "Business Analyst",
    11: "SAP Developer",
    12: "Automation Testing",
    13: "Electrical Engineering",
    14: "Operations Manager",
    15: "Python Developer",
    16: "DevOps Engineer",
    17: "Network Security Engineer",
    18: "PMO",
    19: "Database Developer"
}

# -----------------------------
# 3️⃣ Helper function to clean text
# -----------------------------
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)  # remove links
    text = re.sub(r'[^a-zA-Z\s]', '', text)     # remove numbers & symbols
    text = text.lower()                         # lowercase text
    text = re.sub(r'\s+', ' ', text).strip()    # remove extra spaces
    return text

# -----------------------------
# 4️⃣ Functions to extract text from uploaded files
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return " ".join([para.text for para in doc.paragraphs])

# -----------------------------
# 5️⃣ Streamlit UI
# -----------------------------
st.set_page_config(page_title="Resume Screening App", page_icon="📄", layout="centered")

st.title("📄 Resume Screening ML App")
st.write("Upload your resume or paste text below to predict the job category using our trained ML model.")

# Upload section
uploaded_file = st.file_uploader("📤 Upload Resume File", type=["pdf", "docx", "txt"])

resume_text = ""

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    elif file_type == "docx":
        resume_text = extract_text_from_docx(uploaded_file)
    elif file_type == "txt":
        resume_text = uploaded_file.read().decode("utf-8")
    else:
        st.error("❌ Unsupported file format. Please upload a PDF, DOCX, or TXT file.")

# Manual text area (optional)
manual_text = st.text_area("📝 Or paste resume text manually", height=200, placeholder="Paste candidate resume text here...")
if manual_text.strip():
    resume_text = manual_text

# Predict button
if st.button("Predict"):
    if not resume_text.strip():
        st.warning("⚠️ Please upload a resume or paste some text.")
    else:
        # Clean and vectorize text
        clean_resume = clean_text(resume_text)
        vectorized_text = vectorizer.transform([clean_resume])

        # Predict
        prediction = model.predict(vectorized_text)[0]
        category = label_map.get(prediction, "Unknown Category")

        # Display result
        st.success(f"✅ Predicted Job Category: **{category}**")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & Naive Bayes")
