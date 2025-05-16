import os
import uuid
import json
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
from jose import JWTError, jwt
import openai
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, String, Float, Text, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import fitz  # PyMuPDF
# import app
from .embeddings import (
    get_embedding, add_embedding, faiss_index, get_resume_embedding, cosine_similarity
)
from fastapi.middleware.cors import CORSMiddleware

# --- Security Setup ---
SECRET_KEY = "supersecretkey"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- DB Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'app.db')}"
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

app = FastAPI()

origins = [
    "http://localhost:3000",  
    "http://localhost:8080", 
    "http://localhost",       # Sometimes needed
]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3030"],  # Frontend origin
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Models ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="user")  # "user" or "admin"

class ResumeRecord(Base):
    __tablename__ = "resumes"
    id = Column(String, primary_key=True)
    user_id = Column(Integer)
    pdf_path = Column(String)
    name = Column(String)
    contact = Column(String)
    email = Column(String)
    skills = Column(Text)
    experience = Column(String)
    job_title = Column(String)
    openai_match_score = Column(Float)
    vector_match_score = Column(Float)
    recruiter_feedback = Column(Text)
    user_feedback = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    embedding_id = Column(Integer, unique=True, nullable=False)

Base.metadata.create_all(bind=engine)

# --- Auth Utilities ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_user(db, username: str):
    return db.query(User).filter(User.username == username).first()

def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    db = SessionLocal()
    user = get_user(db, username)
    db.close()
    if user is None:
        raise credentials_exception
    return user

def admin_required(user=Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

def user_required(user=Depends(get_current_user)):
    if user.role not in ["user", "admin"]:
        raise HTTPException(status_code=403, detail="User access required")
    return user

# --- Auth Endpoints ---
@app.post("/register/")
def register(username: str = Form(...), password: str = Form(...), role: str = Form("user")):
    db = SessionLocal()
    if get_user(db, username):
        db.close()
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(password)
    user = User(username=username, hashed_password=hashed_password, role=role)
    db.add(user)
    db.commit()
    db.close()
    return {"msg": "User registered"}

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    db = SessionLocal()
    user = authenticate_user(db, form_data.username, form_data.password)
    db.close()
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# --- 1. POST /resume_score (USER) ---
@app.post("/resume_score")
async def resume_score(
    resume: UploadFile = File(...),
    user: User = Depends(user_required)
):
    resume_bytes = await resume.read()
    resume_id = str(uuid.uuid4())
    pdf_path = os.path.join(UPLOAD_DIR, f"{resume_id}.pdf")
    with open(pdf_path, "wb") as f:
        f.write(resume_bytes)
    resume_text = extract_text_from_pdf(pdf_path)
    openai_score, feedback = get_openai_resume_score(resume_text)
    embedding = get_embedding(resume_text)
    embedding_id = int(uuid.uuid4().int % 1e9)
    add_embedding(faiss_index, embedding, embedding_id)
    fields = extract_fields(resume_text)
    db = SessionLocal()
    record = ResumeRecord(
        id=resume_id,
        user_id=user.id,
        pdf_path=pdf_path,
        openai_match_score=openai_score,
        vector_match_score=None,
        recruiter_feedback=None,
        user_feedback=None,
        created_at=datetime.utcnow(),
        embedding_id=embedding_id,
        **fields
    )
    db.add(record)
    db.commit()
    db.close()
    return {
        "resume_id": resume_id,
        "openai_score": openai_score,
        "feedback": feedback
    }
# --- 1.5 POST /resume_summary - to get resume_id ---
@app.get("/resumes_summary")
def resumes_summary(user: User = Depends(admin_required)):
    """
    List all resumes with: resume_id, name, openai_match_score, vector_match_score, job_title.
    Admin only.
    """
    db = SessionLocal()
    records = db.query(ResumeRecord).all()
    db.close()
    return [
        {
            "resume_id": r.id,
            "name": r.name,
            "openai_match_score": r.openai_match_score,
            "vector_match_score": r.vector_match_score,
            "job_title": r.job_title,
            "user_feedback": r.user_feedback,             
            "hr_feedback": getattr(r, "recruiter_feedback", None), 
        }
        for r in records
    ]
# --- 2. POST /resume_vs_jd_score (ADMIN) ---
@app.post("/resume_vs_jd_score")
async def resume_vs_jd_score(
    resume_id: str = Form(...),  
    resume: UploadFile = File(...),
    jd: UploadFile = File(...),
    user: User = Depends(admin_required)
):
    # Save files and extract text
    resume_path = save_temp_file(resume)
    jd_path = save_temp_file(jd)
    resume_text = extract_text_from_pdf(resume_path)
    jd_text = extract_text_from_pdf(jd_path)

    openai_match_score = get_openai_jd_match(resume_text, jd_text)
    resume_emb = get_embedding(resume_text)
    jd_emb = get_embedding(jd_text)
    vector_match_score = float(cosine_similarity(resume_emb.flatten(), jd_emb.flatten()) * 100)

    db = SessionLocal()
    record = db.query(ResumeRecord).filter_by(id=resume_id).first()
    if not record:
        db.close()
        raise HTTPException(status_code=404, detail="Resume not found")
    
    record.openai_match_score = openai_match_score
    record.vector_match_score = vector_match_score
    db.commit()
    db.close()

    return {
        "resume_id": resume_id,
        "openai_match_score": openai_match_score,
        "vector_match_score": vector_match_score
    }


# --- 3. GET /total_resume (ADMIN) ---
# @app.get("/total_resume")
# def total_resume(user: User = Depends(admin_required)):
#     db = SessionLocal()
#     count = db.query(ResumeRecord).count()
#     db.close()
#     return {"total_resumes": count}

# --- 3. GET /total_resume (ALL USER) ---
@app.get("/total_resume")
def total_resume(user: User = Depends(get_current_user)):
    db = SessionLocal()
    if user.role == "admin":
        count = db.query(ResumeRecord).count()
    else:
        count = db.query(ResumeRecord).filter(ResumeRecord.user_id == user.id).count()
    db.close()
    return {"total": count}


# --- 4. GET /resume_by_id (USER/ADMIN) ---
@app.get("/resume_by_id")
def resume_by_id(resume_id: str, user: User = Depends(get_current_user)):
    db = SessionLocal()
    record = db.query(ResumeRecord).filter_by(id=resume_id).first()
    db.close()
    if not record:
        raise HTTPException(status_code=404, detail="Resume not found")
    if user.role != "admin" and record.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    return {
        "resume_id": record.id,
        "pdf_path": record.pdf_path,
        "name": record.name,
        "contact": record.contact,
        "email": record.email,
        "skills": record.skills,
        "experience": record.experience,
        "job_title": record.job_title,
        "openai_match_score": record.openai_match_score,
        "vector_match_score": record.vector_match_score,
        "recruiter_feedback": record.recruiter_feedback,
        "user_feedback": record.user_feedback,
        "created_at": record.created_at,
    }

# --- 5. GET /resume_by_id_download (USER/ADMIN) ---
@app.get("/resume_by_id_download")
def resume_by_id_download(resume_id: str, user: User = Depends(get_current_user)):
    db = SessionLocal()
    record = db.query(ResumeRecord).filter_by(id=resume_id).first()
    db.close()
    if not record:
        raise HTTPException(status_code=404, detail="Resume not found")
    if user.role != "admin" and record.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    return FileResponse(record.pdf_path, media_type='application/pdf', filename=f"{resume_id}.pdf")

# --- 6. POST /resumes_for_jd_openai (ADMIN) ---
from typing import List

@app.post("/resumes_for_jd_openai")
async def resumes_for_jd_openai(
    resumes: List[UploadFile] = File(...),   # Accept multiple resumes
    jd: UploadFile = File(...),
    top_k: int = Form(...),
    user: User = Depends(admin_required)
):
    # Save and extract JD text
    jd_path = save_temp_file(jd)
    jd_text = extract_text_from_pdf(jd_path)
    db = SessionLocal()
    scored = []

    for resume_file in resumes:
        # Save resume
        resume_id = str(uuid.uuid4())
        resume_path = os.path.join(UPLOAD_DIR, f"{resume_id}.pdf")
        with open(resume_path, "wb") as f:
            f.write(await resume_file.read())
        resume_text = extract_text_from_pdf(resume_path)

        # Extract fields with OpenAI
        fields = extract_fields(resume_text)

        # Calculate scores
        openai_score = get_openai_jd_match(resume_text, jd_text)
        resume_emb = get_embedding(resume_text)
        jd_emb = get_embedding(jd_text)
        vector_score = float(cosine_similarity(resume_emb.flatten(), jd_emb.flatten()) * 100)

        # Store in FAISS and DB
        embedding_id = int(uuid.uuid4().int % 1e9)
        add_embedding(faiss_index, resume_emb, embedding_id)

        record = ResumeRecord(
            id=resume_id,
            user_id=user.id,
            pdf_path=resume_path,
            name=fields.get("name"),
            contact=fields.get("contact"),
            email=fields.get("email"),
            skills=fields.get("skills"),
            experience=fields.get("experience"),
            job_title=fields.get("job_title"),
            openai_match_score=openai_score,
            vector_match_score=vector_score,
            recruiter_feedback=None,
            user_feedback=None,
            created_at=datetime.utcnow(),
            embedding_id=embedding_id
        )
        db.add(record)
        db.commit()

        scored.append({
            "resume_id": resume_id,
            "pdf_path": resume_path,
            "name": fields.get("name"),
            "contact": fields.get("contact"),
            "email": fields.get("email"),
            "skills": fields.get("skills"),
            "experience": fields.get("experience"),
            "openai_match_score": openai_score,
            "vector_match_score": vector_score,
            "recruiter_feedback": None,
            "user_feedback": None,
            "created_at": record.created_at,
            "job_title": fields.get("job_title"),
        })

    db.close()
    scored = sorted(scored, key=lambda x: x["openai_match_score"], reverse=True)[:top_k]
    return scored

@app.post("/resumes_for_jd_vector")
async def resumes_for_jd_vector(
    jd: UploadFile = File(...),
    top_k: int = Form(...),
    user: User = Depends(admin_required)
):
    # Save and extract JD text
    jd_path = save_temp_file(jd)
    jd_text = extract_text_from_pdf(jd_path)
    jd_emb = get_embedding(jd_text)
    db = SessionLocal()
    records = db.query(ResumeRecord).all()
    scored = []

    for r in records:
        resume_emb = get_resume_embedding(r.embedding_id)
        if resume_emb is None:
            continue  # Skip if embedding is missing
        vector_score = float(cosine_similarity(resume_emb.flatten(), jd_emb.flatten()) * 100)
        scored.append({
            "resume_id": r.id,
            "pdf_path": r.pdf_path,
            "name": r.name,
            "contact": r.contact,
            "email": r.email,
            "skills": r.skills,
            "experience": r.experience,
            "openai_match_score": r.openai_match_score,
            "vector_match_score": vector_score,
            "recruiter_feedback": r.recruiter_feedback,
            "user_feedback": r.user_feedback,
            "created_at": r.created_at,
            "job_title": r.job_title,
        })

    db.close()
    # Sort by vector_match_score (descending) and return top_k
    scored = sorted(scored, key=lambda x: x["vector_match_score"], reverse=True)[:top_k]
    return scored


# --- 7. Feedback endpoints ---
@app.post("/feedback/user")
def user_feedback(resume_id: str = Form(...), feedback: str = Form(...), user: User = Depends(user_required)):
    db = SessionLocal()
    record = db.query(ResumeRecord).filter_by(id=resume_id, user_id=user.id).first()
    if not record:
        db.close()
        raise HTTPException(status_code=404, detail="Resume not found")
    record.user_feedback = feedback
    db.commit()
    db.close()
    return {"msg": "User feedback saved"}

@app.post("/feedback/hr")
def hr_feedback(resume_id: str = Form(...), feedback: str = Form(...), user: User = Depends(admin_required)):
    db = SessionLocal()
    record = db.query(ResumeRecord).filter_by(id=resume_id).first()
    if not record:
        db.close()
        raise HTTPException(status_code=404, detail="Resume not found")
    record.recruiter_feedback = feedback
    db.commit()
    db.close()
    return {"msg": "HR feedback saved"}

# --- Utility Functions ---
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def save_temp_file(upload_file: UploadFile):
    temp_path = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4()}.pdf")
    with open(temp_path, "wb") as f:
        f.write(upload_file.file.read())
    return temp_path

def extract_fields(resume_text):
    """
    Use OpenAI to extract name, contact, email, skills, experience, job_title from resume text.
    Returns a dict with those fields, or None if not found.
    """
    prompt = (
        "Extract the following fields from this resume. "
        "Return as a JSON object with these keys: name, contact, email, skills (as a list), "
        "experience (number of years), job_title (most recent):\n\n"
        f"{resume_text}\n"
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        print("OpenAI returned:", content)  # <--- DEBUG LINE
        # Try direct JSON
        try:
            data = json.loads(content)
        except Exception:
            # Try to extract JSON from within text
            try:
                start = content.index('{')
                end = content.rindex('}') + 1
                data = json.loads(content[start:end])
            except Exception as e:
                print("JSON extraction error:", e)
                data = {}
        return {
            "name": data.get("name"),
            "contact": data.get("contact"),
            "email": data.get("email"),
            "skills": json.dumps(data.get("skills", [])) if data.get("skills") else None,
            "experience": data.get("experience"),
            "job_title": data.get("job_title"),
        }
    except Exception as e:
        print(f"OpenAI extraction error: {e}")
        return {
            "name": None, "contact": None, "email": None,
            "skills": None, "experience": None, "job_title": None
        }
    
def get_openai_resume_score(resume_text):
    """
    Use OpenAI to give a resume a general match score and feedback.
    Returns (score, feedback).
    """
    prompt = (
        "You are an ATS. Give a match score (0-100) for the following resume based on general job fit, "
        "and a short feedback. Return as JSON: {\"score\": 80, \"feedback\": \"...\"}\n\nResume:\n"
        f"{resume_text}\n"
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        return data.get("score", 0), data.get("feedback", "")
    except Exception as e:
        print(f"OpenAI resume score error: {e}")
        return 80, "Sample feedback"

def get_openai_jd_match(resume_text, jd_text):
    """
    Use OpenAI to compare a resume to a job description and return a match score (0-100).
    """
    prompt = (
        "Compare this resume to the job description. Return a match score (0-100) as JSON: {\"score\": 85}\n\n"
        f"Resume:\n{resume_text}\n\nJob Description:\n{jd_text}\n"
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        return data.get("score", 0)
    except Exception as e:
        print(f"OpenAI JD match error: {e}")
        return 85

@app.get("/")
def health():
    return {"status": "ok"}
