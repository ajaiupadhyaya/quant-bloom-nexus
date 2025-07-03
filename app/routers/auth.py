from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr
from app.core.config import settings
import bcrypt
import jwt
from datetime import datetime, timedelta
from typing import Optional

router = APIRouter(prefix="/auth", tags=["Auth"])

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserOut(BaseModel):
    id: int
    username: str
    email: EmailStr
    is_active: bool
    created_at: datetime

# In-memory user storage for demo (replace with database in production)
demo_users = {
    "admin": {
        "id": 1,
        "username": "admin",
        "email": "admin@quantbloom.com",
        "hashed_password": bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode(),
        "is_active": True,
        "created_at": datetime.now()
    }
}

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

def create_access_token(data: dict, expires_delta: timedelta = timedelta(hours=12)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm="HS256")

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        return None

@router.post("/register", response_model=UserOut)
def register(user: UserCreate):
    if user.username in demo_users:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed = hash_password(user.password)
    user_id = len(demo_users) + 1
    
    demo_users[user.username] = {
        "id": user_id,
        "username": user.username,
        "email": user.email,
        "hashed_password": hashed,
        "is_active": True,
        "created_at": datetime.now()
    }
    
    return demo_users[user.username]

@router.post("/login")
def login(user: UserLogin):
    if user.username not in demo_users:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    db_user = demo_users[user.username]
    if not verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token({"sub": db_user["username"], "user_id": db_user["id"]})
    return {"access_token": token, "token_type": "bearer"}

@router.get("/me", response_model=UserOut)
def me(token: Optional[str] = None):
    if not token:
        raise HTTPException(status_code=401, detail="Token required")
    
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    username = payload.get("sub")
    if username not in demo_users:
        raise HTTPException(status_code=404, detail="User not found")
    
    return demo_users[username]

@router.get("/demo-login")
def demo_login():
    """Demo login endpoint for testing"""
    token = create_access_token({"sub": "admin", "user_id": 1})
    return {
        "access_token": token, 
        "token_type": "bearer",
        "message": "Demo login successful. Use this token for testing."
    } 