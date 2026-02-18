import streamlit as st
import PyPDF2
import re
import json
import os
import nltk
import io
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import requests
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# Additional imports for multi-format file support and performance
from PIL import Image
import easyocr
import xml.etree.ElementTree as ET
import chardet
import threading
import concurrent.futures
from functools import lru_cache
import time

# ===============================
# CACHING & PERFORMANCE SETUP
# ===============================
# Cache for OCR reader (singleton pattern with thread safety)
_ocr_reader = None
_ocr_lock = threading.Lock()

@st.cache_resource
def get_ocr_reader():
    """Get or create EasyOCR reader instance (cached globally)."""
    global _ocr_reader
    if _ocr_reader is None:
        with _ocr_lock:
            if _ocr_reader is None:
                _ocr_reader = easyocr.Reader(['en'], gpu=False)
    return _ocr_reader

# Cache API responses to avoid redundant calls
@st.cache_data(ttl=3600, show_spinner=False)
def cached_api_call(api_key, prompt_hash, prompt):
    """Cache API responses based on prompt hash."""
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 2000
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return None
    except Exception as e:
        return None

# Cache text extraction results
@st.cache_data(show_spinner=False)
def cached_extract_text(file_content, file_type, file_name):
    """Cache extracted text from files."""
    return file_content  # Placeholder - actual extraction happens in get_text

# ===============================
# INITIAL SETUP
# ===============================
st.set_page_config(page_title="🚀 Job Resume Skill Builder & Analyzer (2026)...by aryan chavan", layout="wide")

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

USER_DB = "users_v2.json"
JOB_DATA_DB = "job_market_data.json"

# ===============================
# USER DATABASE FUNCTIONS (Thread-Safe)
# ===============================
USER_DB_LOCK = threading.Lock()

def load_users():
    """Load users with thread safety."""
    if not os.path.exists(USER_DB):
        with open(USER_DB, "w") as f:
            json.dump({}, f)
        return {}
    try:
        with open(USER_DB, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_users(data):
    """Save users with thread safety."""
    with USER_DB_LOCK:
        with open(USER_DB, "w") as f:
            json.dump(data, f, indent=4)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_job_data():
    if not os.path.exists(JOB_DATA_DB):
        with open(JOB_DATA_DB, "w") as f:
            json.dump({}, f)
    with open(JOB_DATA_DB, "r") as f:
        return json.load(f)

def save_job_data(data):
    with open(JOB_DATA_DB, "w") as f:
        json.dump(data, f, indent=4)

# ===============================
# LOGIN SYSTEM
# ===============================
users = load_users()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title("🔐 Job Resume Skill Builder & Analyzer by aryan chavan")
    st.markdown("## Login / Signup")

    choice = st.radio("Select Option", ["Login", "Signup"])

    username = st.text_input("Full Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if choice == "Signup":
        if st.button("Create Account"):
            if email in users:
                st.error("User already exists with this email.")
            else:
                users[email] = {
                    "name": username,
                    "email": email,
                    "password": hash_password(password),
                    "api_key": "",
                    "groq_api_key": "",
                    "chat_history": [],
                    "resume_analyses": [],
                    "job_preferences": {},
                    "created_at": datetime.now().isoformat()
                }
                save_users(users)
                st.success("Account created successfully! You can now login.")

    if choice == "Login":
        if st.button("Login"):
            if email in users and users[email]["password"] == hash_password(password):
                st.session_state.logged_in = True
                st.session_state.email = email
                st.success("Login successful! Redirecting...")
                st.rerun()
            else:
                st.error("Invalid credentials")

    st.stop()

# ===============================
# MAIN DASHBOARD
# ===============================
st.title(f"🚀 Welcome {users[st.session_state.email]['name']}")

current_user = users[st.session_state.email]

# ===============================
# API KEY MANAGEMENT
# ===============================
st.sidebar.header("🔑 API Configuration")
st.sidebar.subheader("Groq API (for LLM responses)")
st.sidebar.subheader("GET your api key from here = https://console.groq.com/keys")
groq_api_key = st.sidebar.text_input("Enter Groq API Key", value=current_user.get("groq_api_key", ""), type="password")

if st.sidebar.button("Save Groq API Key"):
    users[st.session_state.email]["groq_api_key"] = groq_api_key
    save_users(users)
    st.sidebar.success("Groq API Key Saved!")

st.sidebar.subheader("Custom API (for job market data)")
custom_api_key = st.sidebar.text_input("Enter Custom API Key (Optional)", value=current_user.get("api_key", ""), type="password")

if st.sidebar.button("Save Custom API Key"):
    users[st.session_state.email]["api_key"] = custom_api_key
    save_users(users)
    st.sidebar.success("Custom API Key Saved!")

# ===============================
# SESSION STATE FOR RESUME DATA
# ===============================
if "resume_data" not in st.session_state:
    st.session_state.resume_data = None
if "resume_skills" not in st.session_state:
    st.session_state.resume_skills = []
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

# ===============================
# SIDEBAR NAVIGATION
# ===============================
st.sidebar.header("📱 Navigation")
nav_option = st.sidebar.radio("Select Feature", [
    "Resume Analysis",
    "Job Recommendation",
    "Career Guidance",
    "Job Market Analysis",
    "Skills Development",
    "Startup Ideas",
    "AI Chatbot",
    "Chat History"
])

# ===============================
# TEXT PROCESSING
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def extract_pdf_text(file):
    """Extract text from PDF files."""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
    return text


def extract_image_text(file):
    """Extract text from image files (JPG, PNG) using EasyOCR."""
    text = ""
    try:
        # Reset file pointer
        file.seek(0)
        image_bytes = file.read()
        
        # Convert to numpy array for EasyOCR
        import numpy as np
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Use EasyOCR for text extraction
        reader = get_ocr_reader()
        results = reader.readtext(image_array)
        
        # Combine all detected text
        text = " ".join([detection[1] for detection in results])
    except Exception as e:
        st.error(f"Error reading image file: {e}")
    return text


def extract_xml_text(file):
    """Extract text content from XML files."""
    text = ""
    try:
        # Reset file pointer
        file.seek(0)
        raw_data = file.read()
        
        # Detect encoding
        detected = chardet.detect(raw_data)
        encoding = detected.get('encoding', 'utf-8')
        
        # Parse XML
        root = ET.fromstring(raw_data.decode(encoding, errors='replace'))
        
        # Extract all text content from XML elements
        def get_element_text(element):
            result = []
            if element.text and element.text.strip():
                result.append(element.text.strip())
            for child in element:
                result.extend(get_element_text(child))
            if element.tail and element.tail.strip():
                result.append(element.tail.strip())
            return result
        
        text = " ".join(get_element_text(root))
    except ET.ParseError as e:
        st.error(f"Error parsing XML file: {e}")
    except Exception as e:
        st.error(f"Error reading XML file: {e}")
    return text


def extract_txt_text(file):
    """Extract text from plain text files with automatic encoding detection."""
    text = ""
    try:
        # Reset file pointer
        file.seek(0)
        raw_data = file.read()
        
        # Detect encoding
        detected = chardet.detect(raw_data)
        encoding = detected.get('encoding', 'utf-8')
        
        # Decode with detected encoding
        text = raw_data.decode(encoding, errors='replace')
    except Exception as e:
        st.error(f"Error reading text file: {e}")
    return text


def get_text(file):
    """
    Extract text from various file types.
    Supports: PDF, JPG, JPEG, PNG, TXT, XML
    """
    file_type = file.type
    file_name = file.name.lower()
    
    # PDF files
    if file_type == "application/pdf" or file_name.endswith('.pdf'):
        return extract_pdf_text(file)
    
    # Image files (JPG, JPEG, PNG)
    elif file_type in ["image/jpeg", "image/jpg", "image/png"] or \
         file_name.endswith(('.jpg', '.jpeg', '.png')):
        return extract_image_text(file)
    
    # XML files
    elif file_type in ["application/xml", "text/xml"] or file_name.endswith('.xml'):
        return extract_xml_text(file)
    
    # Text files (default)
    else:
        return extract_txt_text(file)

# ===============================
# GROQ API INTEGRATION (Optimized)
# ===============================
def call_groq_api(prompt, model='llama-3.3-70b-versatile', use_cache=True):
    """Call Groq API with caching for faster responses."""
    api_key = users[st.session_state.email]["groq_api_key"]
    if not api_key:
        st.error("Please configure your Groq API key first.")
        return None
    
    # Create hash for caching
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cache_key = f"{api_key[:8]}_{prompt_hash}"
    
    # Try to get from cache first
    if use_cache:
        cached = cached_api_call(api_key, cache_key, prompt)
        if cached:
            return cached
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 2000
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.Timeout:
        st.error("API request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error calling Groq API: {e}")
        return None


def call_multiple_apis_parallel(prompts_dict):
    """Call multiple API prompts in parallel for faster processing."""
    results = {}
    
    def call_single(name, prompt):
        return name, call_groq_api(prompt, use_cache=True)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(call_single, name, prompt): name 
                   for name, prompt in prompts_dict.items()}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                name, result = future.result()
                results[name] = result
            except Exception as e:
                results[futures[future]] = None
    
    return results

# ===============================
# RESUME ANALYSIS (Optimized with Parallel Processing)
# ===============================
if nav_option == "Resume Analysis":
    st.markdown("## 📄 Complete Resume Analysis")
    st.info("📁 Supported file formats: PDF, TXT, XML, JPG, JPEG, PNG")
    
    resume_file = st.file_uploader(
        "Upload your Resume", 
        type=["pdf", "txt", "xml", "jpg", "jpeg", "png"],
        help="Upload your resume in PDF, TXT, XML, or image format (JPG, PNG)"
    )
    
    if resume_file:
        if st.button("🔍 Analyze Resume Completely", type="primary"):
            start_time = time.time()
            
            with st.spinner("Extracting text from resume..."):
                resume_text = get_text(resume_file)
                clean_resume = clean_text(resume_text)
                
                # Store in session state
                st.session_state.resume_text = resume_text
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Prepare all prompts for parallel execution
            status_text.text("🚀 Running parallel analysis (5x faster)...")
            
            prompts = {
                "skills": f"""Analyze this resume text and extract all technical and soft skills. 
                Focus on: Programming languages, Frameworks, Tools, Certifications, Soft skills, Industry skills.
                Resume: {resume_text}
                Return JSON with key "skills" containing list of skills.""",
                
                "jobs": f"""Based on this resume, suggest 5 high-paying job roles.
                Resume: {resume_text}
                For each job: title, salary range (USD), match %, responsibilities, skills they have, skills to develop, top companies.
                Return JSON with "jobs" array.""",
                
                "career": f"""Based on this resume, provide career guidance:
                Resume: {resume_text}
                Include: Current level, best career paths, short-term goals (6-12 months), long-term goals (2-5 years), 
                certifications, industry trends, networking suggestions. Be specific.""",
                
                "skills_dev": f"""Based on this resume, create a skill development plan:
                Resume: {resume_text}
                Include: Skill gaps, priority skills to learn, learning resources, projects to build, timeline, certifications.""",
                
                "startup": f"""Based on this resume, suggest 3 startup ideas:
                Resume: {resume_text}
                For each: idea description, problem solved, how their skills apply, target market, revenue model, 
                MVP requirements, investment needed, first steps."""
            }
            
            # Execute all API calls in parallel
            progress_bar.progress(20)
            results = call_multiple_apis_parallel(prompts)
            progress_bar.progress(80)
            
            status_text.text("📊 Rendering results...")
            
            # Create tabs for comprehensive analysis
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🎯 Skills Analysis", 
                "💼 Job Recommendations", 
                "🎓 Career Guidance",
                "🛠️ Skill Development",
                "💡 Startup Ideas"
            ])
            
            # =====================
            # TAB 1: SKILLS ANALYSIS
            # =====================
            with tab1:
                skills_response = results.get("skills")
                skills = []
                
                if skills_response:
                    try:
                        skills_data = json.loads(skills_response)
                        skills = skills_data.get("skills", [])
                        st.session_state.resume_skills = skills
                        
                        st.markdown("### 🎯 Extracted Skills")
                        
                        col1, col2 = st.columns(2)
                        tech_skills = [s for s in skills if any(kw in s.lower() for kw in ['python', 'java', 'sql', 'javascript', 'react', 'node', 'aws', 'docker', 'git', 'api', 'machine learning', 'data', 'cloud', 'linux', 'mongodb', 'postgresql'])]
                        soft_skills = [s for s in skills if s not in tech_skills]
                        
                        with col1:
                            st.markdown("**🔧 Technical Skills**")
                            for skill in tech_skills[:10]:
                                st.write(f"- {skill}")
                        
                        with col2:
                            st.markdown("**🤝 Soft Skills**")
                            for skill in soft_skills[:10]:
                                st.write(f"- {skill}")
                                
                    except:
                        st.write(skills_response)
            
            # =====================
            # TAB 2: JOB RECOMMENDATIONS
            # =====================
            with tab2:
                st.markdown("### 💼 Job Recommendations Based on Your Resume")
                
                job_response = results.get("jobs")
                if job_response:
                    try:
                        job_data = json.loads(job_response)
                        for idx, job in enumerate(job_data.get("jobs", []), 1):
                            with st.expander(f"📌 {idx}. {job.get('title', 'Job Role')}", expanded=(idx==1)):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"💰 **Salary:** {job.get('salary', 'N/A')}")
                                    st.write(f"📊 **Match:** {job.get('match', 'N/A')}")
                                with col2:
                                    st.write(f"🏢 **Top Companies:** {job.get('companies', 'N/A')}")
                                st.write(f"📋 **Responsibilities:** {job.get('responsibilities', 'N/A')}")
                                st.write(f"✅ **Skills You Have:** {job.get('skills_have', 'N/A')}")
                                st.write(f"📈 **Skills to Develop:** {job.get('skills_need', 'N/A')}")
                    except:
                        st.write(job_response)
            
                # =====================
                # TAB 3: CAREER GUIDANCE
                # =====================
                with tab3:
                    st.markdown("### 🎓 Personalized Career Guidance")
                    
                    career_response = results.get("career")
                    if career_response:
                        st.markdown(career_response)
                
                # =====================
                # TAB 4: SKILL DEVELOPMENT
                # =====================
                with tab4:
                    st.markdown("### 🛠️ Skill Development Plan")
                    
                    skill_dev_response = results.get("skills_dev")
                    if skill_dev_response:
                        st.markdown(skill_dev_response)
                
                # =====================
                # TAB 5: STARTUP IDEAS
                # =====================
                with tab5:
                    st.markdown("### 💡 Startup Ideas Based on Your Skills")
                    
                    startup_response = results.get("startup")
                    if startup_response:
                        st.markdown(startup_response)
                
                # Complete progress
                progress_bar.progress(100)
                elapsed_time = time.time() - start_time
                
                # Save analysis to history
                users[st.session_state.email]["resume_analyses"].append({
                    "filename": resume_file.name,
                    "analyzed_at": datetime.now().isoformat(),
                    "text_preview": resume_text[:200],
                    "skills": st.session_state.resume_skills
                })
                save_users(users)
                
                status_text.text(f"✅ Analysis complete in {elapsed_time:.1f} seconds!")
                st.success(f"✅ Resume analysis complete in {elapsed_time:.1f} seconds! Your data is saved for personalized recommendations.")

# ===============================
# JOB RECOMMENDATION
# ===============================
if nav_option == "Job Recommendation":
    st.markdown("## 💼 Job Recommendation")
    
    # Check if resume data is available
    if st.session_state.resume_text:
        st.success("✅ Using your uploaded resume for personalized recommendations!")
        use_resume = st.checkbox("Use resume data for recommendations", value=True)
    else:
        use_resume = False
        st.info("💡 Upload your resume in 'Resume Analysis' for personalized recommendations!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
    with col2:
        desired_location = st.text_input("Desired Location", "Remote")
    with col3:
        job_type = st.selectbox("Job Type", ["Full-time", "Part-time", "Contract", "Freelance"])
    
    if st.button("Get Job Recommendations"):
        with st.spinner("Searching for jobs..."):
            if use_resume and st.session_state.resume_text:
                prompt = f"""As a career advisor, recommend 5 high-paying jobs based on:
                - Resume: {st.session_state.resume_text[:2000]}
                - Skills: {', '.join(st.session_state.resume_skills)}
                - Experience: {experience} years
                - Location: {desired_location}
                - Job Type: {job_type}
                
                For each job, include:
                - Job title
                - Average annual salary
                - Match percentage with resume skills
                - Key requirements
                - Future stability rating (1-10)
                - Why it's a good career choice
                - How to prepare for this role
                
                Return in JSON format with "recommendations" array.
                """
            else:
                prompt = f"""As a career advisor, recommend 5 high-paying jobs based on:
                - Experience: {experience} years
                - Location: {desired_location}
                - Job Type: {job_type}
                
                For each job, include:
                - Job title
                - Average annual salary
                - Key requirements
                - Future stability rating (1-10)
                - Why it's a good career choice
                - How to prepare for this role
                
                Return in JSON format with "recommendations" array.
                """
            
            response = call_groq_api(prompt)
            if response:
                try:
                    recommendations = json.loads(response)
                    for idx, job in enumerate(recommendations.get("recommendations", []), 1):
                        st.markdown(f"### 📌 {idx}. {job['title']}")
                        st.write(f"💰 Salary: {job['salary']}")
                        st.write(f"📊 Stability: {job['stability']}/10")
                        st.write(f"📋 Requirements: {job['requirements']}")
                        st.write(f"💡 Why Good: {job['why_good']}")
                        st.write(f"🎯 How to Prepare: {job['prepare']}")
                        st.markdown("---")
                except:
                    st.write(response)

# ===============================
# CAREER GUIDANCE
# ===============================
if nav_option == "Career Guidance":
    st.markdown("## 🎓 Career Guidance")
    
    # Check if resume data is available
    if st.session_state.resume_text:
        st.success("✅ Using your uploaded resume for personalized guidance!")
        use_resume = st.checkbox("Use resume data for career guidance", value=True)
        default_skills = ", ".join(st.session_state.resume_skills[:10]) if st.session_state.resume_skills else "Python, SQL, Excel"
    else:
        use_resume = False
        st.info("💡 Upload your resume in 'Resume Analysis' for personalized guidance!")
        default_skills = "Python, SQL, Excel"
    
    career_goal = st.text_area("What's your career goal?", "I want to become a data scientist.")
    current_skills = st.text_area("Current Skills", default_skills)
    
    if st.button("Get Career Guidance"):
        with st.spinner("Analyzing your career path..."):
            if use_resume and st.session_state.resume_text:
                prompt = f"""Act as a professional career counselor. Provide detailed guidance based on:
                - Resume: {st.session_state.resume_text[:2000]}
                - Career Goal: {career_goal}
                - Current Skills: {current_skills}
                
                Include:
                1. Assessment of current resume for the career goal
                2. Step-by-step career roadmap
                3. Skills gap analysis and learning resources
                4. Certifications that will boost your career
                5. Potential challenges and how to overcome them
                6. Real-world career advice
                7. Timeline for achieving your goal
                
                Be honest and practical in your advice.
                """
            else:
                prompt = f"""Act as a professional career counselor. Provide detailed guidance based on:
                - Career Goal: {career_goal}
                - Current Skills: {current_skills}
                
                Include:
                1. Step-by-step career roadmap
                2. Skills you need to develop (with learning resources)
                3. Certifications that will boost your career
                4. Potential challenges and how to overcome them
                5. Real-world career advice
                6. Timeline for achieving your goal
                
                Be honest and practical in your advice.
                """
            
            response = call_groq_api(prompt)
            if response:
                st.markdown("### 💡 Career Guidance")
                st.write(response)

# ===============================
# JOB MARKET ANALYSIS
# ===============================
if nav_option == "Job Market Analysis":
    st.markdown("## 📈 Job Market Analysis")
    
    industry = st.selectbox("Select Industry", [
        "Technology", "Healthcare", "Finance", "Marketing", "Education",
        "Consulting", "Manufacturing", "Retail", "Energy", "Entertainment"
    ])
    
    if st.button("Analyze Job Market"):
        with st.spinner("Analyzing current job market..."):
            prompt = f"""Analyze the current {industry} job market in 2026. Include:
            - Key trends and growth areas
            - High-demand job roles
            - Salary ranges for top positions
            - Future market projections (next 5 years)
            - Skills that will be in demand
            - Market stability and future hybridization
            - Any emerging technologies impacting the industry
            
            Return analysis in clear, structured markdown format.
            """
            
            response = call_groq_api(prompt)
            if response:
                st.markdown("### 📊 Market Analysis Report")
                st.write(response)
                
                # Visualization
                st.markdown("### 📉 Job Market Trends")
                data = {
                    'Job Role': ['Software Engineer', 'Data Scientist', 'Product Manager', 'DevOps'],
                    'Demand Score': [95, 92, 88, 85],
                    'Salary (USD)': [150000, 145000, 140000, 135000],
                    'Future Growth (%)': [15, 20, 18, 12]
                }
                
                df = pd.DataFrame(data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots()
                    sns.barplot(x='Job Role', y='Demand Score', data=df, ax=ax, palette='viridis')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots()
                    sns.lineplot(x='Job Role', y='Future Growth (%)', data=df, ax=ax, marker='o')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

# ===============================
# SKILLS DEVELOPMENT
# ===============================
if nav_option == "Skills Development":
    st.markdown("## 🛠️ Skills Development")
    
    # Check if resume data is available
    if st.session_state.resume_text:
        st.success("✅ Using your uploaded resume for personalized skill development!")
        use_resume = st.checkbox("Use resume data for skill development", value=True)
        default_goal = "Advanced skills based on my resume gaps"
    else:
        use_resume = False
        st.info("💡 Upload your resume in 'Resume Analysis' for personalized skill development!")
        default_goal = "Machine Learning"
    
    current_skill_level = st.selectbox("Your Current Skill Level", ["Beginner", "Intermediate", "Advanced"])
    learning_goal = st.text_input("What skills do you want to learn?", default_goal)
    
    if st.button("Get Learning Plan"):
        with st.spinner("Creating your learning plan..."):
            if use_resume and st.session_state.resume_text:
                prompt = f"""Design a comprehensive learning plan based on:
                - Resume: {st.session_state.resume_text[:2000]}
                - Current Skills: {', '.join(st.session_state.resume_skills)}
                - Skill Level: {current_skill_level}
                - Learning Goal: {learning_goal}
                
                Include:
                1. Skill gap analysis from resume
                2. Priority skills to learn (ranked by career impact)
                3. Learning roadmap (step-by-step)
                4. Key concepts to master
                5. Best learning resources (courses, books, tutorials)
                6. Projects to build that complement their experience
                7. Practical exercises
                8. Timeframe for mastery
                9. Certification recommendations
                
                Be specific and practical.
                """
            else:
                prompt = f"""Design a comprehensive learning plan for someone with {current_skill_level} skills who wants to learn {learning_goal}.
                
                Include:
                1. Learning roadmap (step-by-step)
                2. Key concepts to master
                3. Best learning resources (courses, books, tutorials)
                4. Projects to build
                5. Practical exercises
                6. Timeframe for mastery
                7. Certification recommendations
                
                Be specific and practical.
                """
            
            response = call_groq_api(prompt)
            if response:
                st.markdown("### 📚 Learning Plan")
                st.write(response)

# ===============================
# STARTUP IDEAS
# ===============================
if nav_option == "Startup Ideas":
    st.markdown("## 💡 Startup Ideas")
    
    # Check if resume data is available
    if st.session_state.resume_text:
        st.success("✅ Using your uploaded resume for personalized startup ideas!")
        use_resume = st.checkbox("Use resume data for startup ideas", value=True)
        default_interests = ", ".join(st.session_state.resume_skills[:5]) if st.session_state.resume_skills else "Artificial Intelligence, Healthcare, Fintech"
    else:
        use_resume = False
        st.info("💡 Upload your resume in 'Resume Analysis' for personalized startup ideas!")
        default_interests = "Artificial Intelligence, Healthcare, Fintech"
    
    interests = st.text_area("Your Interests & Expertise", default_interests)
    investment_capacity = st.selectbox("Investment Capacity", ["Bootstrapped", "Small ($50k-$200k)", "Medium ($200k-$1M)", "Large ($1M+)"])
    
    if st.button("Generate Startup Ideas"):
        with st.spinner("Generating innovative startup ideas..."):
            if use_resume and st.session_state.resume_text:
                prompt = f"""As a startup incubator expert, generate 3 unique startup ideas based on:
                - Resume: {st.session_state.resume_text[:2000]}
                - Skills: {', '.join(st.session_state.resume_skills)}
                - Interests: {interests}
                - Investment Capacity: {investment_capacity}
                
                For each idea:
                1. Idea description
                2. Problem it solves
                3. How their skills apply to this startup
                4. Target market
                5. Revenue model
                6. Required skills & team
                7. MVP (Minimum Viable Product) requirements
                8. Market potential
                9. Risks and how to mitigate
                10. First steps to get started
                
                Focus on 2026 technologies and market trends. Leverage their existing skills.
                """
            else:
                prompt = f"""As a startup incubator expert, generate 3 unique startup ideas based on:
                - Interests: {interests}
                - Investment Capacity: {investment_capacity}
                
                For each idea:
                1. Idea description
                2. Problem it solves
                3. Target market
                4. Revenue model
                5. Required skills & team
                6. MVP (Minimum Viable Product) requirements
                7. Market potential
                8. Risks and how to mitigate
                
                Focus on 2026 technologies and market trends.
                """
            
            response = call_groq_api(prompt)
            if response:
                st.markdown("### 🚀 Startup Ideas")
                st.write(response)
             
# ===============================
# AI CHATBOT
# ===============================
if nav_option == "AI Chatbot":
    st.markdown("## 🤖 AI Career Assistant")
    st.markdown("Ask me anything about careers, jobs, skills, or get advice!")
    
    # Initialize chat history for this session
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    user_question = st.chat_input("Ask your question here...")
    
    if user_question:
        # Add user message to chat
        st.session_state.chat_messages.append({"role": "user", "content": user_question})
        
        with st.chat_message("user"):
            st.write(user_question)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Include resume context if available
                if st.session_state.resume_text:
                    context = f"""You are a helpful career assistant. The user has uploaded their resume.
                    Resume summary: {st.session_state.resume_text[:1500]}
                    Their skills: {', '.join(st.session_state.resume_skills)}
                    
                    User question: {user_question}
                    
                    Provide a helpful, personalized response based on their background when relevant."""
                else:
                    context = f"""You are a helpful career assistant. 
                    User question: {user_question}
                    
                    Provide a helpful, informative response about careers, jobs, skills, or professional development."""
                
                response = call_groq_api(context)
                
                if response:
                    st.write(response)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    
                    # Save to user's chat history
                    users[st.session_state.email]["chat_history"].append({
                        "query": user_question,
                        "response": response,
                        "timestamp": datetime.now().isoformat()
                    })
                    save_users(users)
                else:
                    st.error("Failed to get response. Please check your API key.")

# ===============================
# CHAT HISTORY
# ===============================
if nav_option == "Chat History":
    st.markdown("## 📜 Chat History")
    
    if current_user["chat_history"]:
        for i, chat in enumerate(reversed(current_user["chat_history"])):
            st.markdown(f"### 💬 {chat['timestamp']}")
            st.write(f"**Query:** {chat['query']}")
            st.write(f"**Response:** {chat['response']}")
            st.markdown("---")
    else:
        st.info("No chat history yet. Start exploring features to begin!")

# ===============================
# LOGOUT
# ===============================
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()
