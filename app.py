from __future__ import annotations

import os
import pickle
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

from db import get_db, init_db
from models import CustomObject, DetectionLog, Face, User
from schemas import (
    AuthLogin,
    AuthRegister,
    DetectionResponse,
    FaceItem,
    LogItem,
    ObjectItem,
    TokenResponse,
    UserOut,
)
from services.face_service import FaceService
from services.yolo_service import YoloService

APP_STORAGE = os.path.join(os.path.dirname(__file__), "storage")
FACE_STORAGE = os.path.join(APP_STORAGE, "faces")
OBJECT_STORAGE = os.path.join(APP_STORAGE, "objects")

os.makedirs(FACE_STORAGE, exist_ok=True)
os.makedirs(OBJECT_STORAGE, exist_ok=True)

SECRET_KEY = os.getenv("AUTH_SECRET", "dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_MINUTES = int(os.getenv("ACCESS_TOKEN_MINUTES", "1440"))

pwd_context = CryptContext(
    schemes=["pbkdf2_sha256", "bcrypt_sha256", "bcrypt"],
    deprecated="auto",
)
security = HTTPBearer()

app = FastAPI(title="Blind Assistance Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_service = FaceService()
yolo_service = YoloService()


def ensure_cv2():
    if cv2 is None:
        raise HTTPException(status_code=503, detail="opencv-python is not installed")


def read_upload(file: UploadFile) -> bytes:
    contents = file.file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload")
    return contents


def decode_image(contents: bytes) -> np.ndarray:
    ensure_cv2()
    data = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return image


def save_bytes(contents: bytes, folder: str, filename: Optional[str]) -> str:
    os.makedirs(folder, exist_ok=True)
    safe_name = filename or "upload.jpg"
    stamped = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{safe_name}"
    path = os.path.join(folder, stamped)
    with open(path, "wb") as f:
        f.write(contents)
    return path


def get_password_hash(password: str) -> str:
    # PBKDF2 avoids bcrypt's 72-byte limit for new passwords.
    return pwd_context.hash(password, scheme="pbkdf2_sha256")

def verify_password(password: str, password_hash: str) -> bool:
    try:
        return pwd_context.verify(password, password_hash)
    except ValueError as exc:
        # Older bcrypt hashes can still hit the 72-byte bcrypt limit.
        if password_hash.startswith("$2") and "72 bytes" in str(exc):
            truncated = password.encode("utf-8")[:72].decode("utf-8", errors="ignore")
            return pwd_context.verify(truncated, password_hash)
        raise


def create_access_token(subject: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_MINUTES)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(status_code=401, detail="Invalid authentication")

    try:
        token = credentials.credentials  # ✅ extract actual token string

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")

        if user_id is None:
            raise credentials_exception

    except JWTError as exc:
        raise credentials_exception from exc

    user = db.query(User).filter(User.id == int(user_id)).first()

    if user is None:
        raise credentials_exception

    return user


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/register", response_model=TokenResponse)
def register(payload: AuthRegister, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        name=payload.name.strip(),
        email=payload.email.lower(),
        password_hash=get_password_hash(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(str(user.id))
    user_out = UserOut(id=user.id, name=user.name, email=user.email, created_at=user.created_at)
    return TokenResponse(access_token=token, token_type="bearer", user=user_out)


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: AuthLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email.lower()).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(str(user.id))
    user_out = UserOut(id=user.id, name=user.name, email=user.email, created_at=user.created_at)
    return TokenResponse(access_token=token, token_type="bearer", user=user_out)


@app.get("/auth/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    return UserOut(
        id=current_user.id,
        name=current_user.name,
        email=current_user.email,
        created_at=current_user.created_at,
    )


@app.post("/detect/objects", response_model=DetectionResponse)
def detect_objects(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if image.content_type is None or "image" not in image.content_type:
        raise HTTPException(status_code=400, detail="Upload an image file")

    try:
        contents = read_upload(image)
        frame = decode_image(contents)
        boxes = yolo_service.detect(frame)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    for box in boxes:
        db.add(
            DetectionLog(
                user_id=current_user.id,
                kind="object",
                label=box["label"],
                confidence=box["confidence"],
            )
        )
    db.commit()
    return {"boxes": boxes}


@app.post("/detect/faces", response_model=DetectionResponse)
def detect_faces(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if image.content_type is None or "image" not in image.content_type:
        raise HTTPException(status_code=400, detail="Upload an image file")

    try:
        contents = read_upload(image)
        frame = decode_image(contents)
        boxes = face_service.detect_faces(frame)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    known_faces = []
    for face in db.query(Face).filter(Face.user_id == current_user.id).all():
        embedding = pickle.loads(face.embedding) if face.embedding else None
        known_faces.append((face.id, face.name, embedding))

    for box in boxes:
        embedding = face_service.extract_embedding(frame, box)
        match = face_service.match_face(embedding, known_faces)
        if match and match[2] >= 0.75:
            box["label"] = match[1]
        else:
            box["label"] = "Unknown"
        db.add(
            DetectionLog(
                user_id=current_user.id,
                kind="face",
                label=box["label"],
                confidence=box["confidence"],
            )
        )

    db.commit()
    return {"boxes": boxes}


@app.post("/faces/register", response_model=FaceItem)
def register_face(
    name: str = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if image.content_type is None or "image" not in image.content_type:
        raise HTTPException(status_code=400, detail="Upload an image file")

    contents = read_upload(image)
    frame = decode_image(contents)
    boxes = face_service.detect_faces(frame)
    if not boxes:
        raise HTTPException(status_code=400, detail="No face detected")

    embedding = face_service.extract_embedding(frame, boxes[0])
    stored_image_path = save_bytes(contents, FACE_STORAGE, image.filename)

    face = Face(
        user_id=current_user.id,
        name=name,
        embedding=pickle.dumps(embedding) if embedding is not None else None,
        image_path=stored_image_path,
    )
    db.add(face)
    db.commit()
    db.refresh(face)

    return FaceItem(id=face.id, name=face.name, image_path=face.image_path, created_at=face.created_at)


@app.get("/faces", response_model=list[FaceItem])
def list_faces(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    faces = (
        db.query(Face)
        .filter(Face.user_id == current_user.id)
        .order_by(Face.created_at.desc())
        .all()
    )
    return [FaceItem(id=f.id, name=f.name, image_path=f.image_path, created_at=f.created_at) for f in faces]


@app.delete("/faces/{face_id}")
def delete_face(
    face_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    face = db.query(Face).filter(Face.id == face_id, Face.user_id == current_user.id).first()
    if not face:
        raise HTTPException(status_code=404, detail="Face not found")

    if face.image_path and os.path.exists(face.image_path):
        try:
            os.remove(face.image_path)
        except OSError:
            pass

    db.delete(face)
    db.commit()
    return {"status": "deleted"}


@app.post("/objects", response_model=ObjectItem)
def create_object(
    name: str = Form(...),
    category: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    image_path = None
    if image is not None:
        if image.content_type is None or "image" not in image.content_type:
            raise HTTPException(status_code=400, detail="Upload an image file")
        contents = read_upload(image)
        image_path = save_bytes(contents, OBJECT_STORAGE, image.filename)

    obj = CustomObject(
        user_id=current_user.id,
        name=name,
        category=category,
        image_path=image_path,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)

    return ObjectItem(
        id=obj.id,
        name=obj.name,
        category=obj.category,
        image_path=obj.image_path,
        created_at=obj.created_at,
    )


@app.get("/objects", response_model=list[ObjectItem])
def list_objects(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    objects = (
        db.query(CustomObject)
        .filter(CustomObject.user_id == current_user.id)
        .order_by(CustomObject.created_at.desc())
        .all()
    )
    return [
        ObjectItem(
            id=o.id,
            name=o.name,
            category=o.category,
            image_path=o.image_path,
            created_at=o.created_at,
        )
        for o in objects
    ]


@app.delete("/objects/{object_id}")
def delete_object(
    object_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    obj = (
        db.query(CustomObject)
        .filter(CustomObject.id == object_id, CustomObject.user_id == current_user.id)
        .first()
    )
    if not obj:
        raise HTTPException(status_code=404, detail="Object not found")

    if obj.image_path and os.path.exists(obj.image_path):
        try:
            os.remove(obj.image_path)
        except OSError:
            pass

    db.delete(obj)
    db.commit()
    return {"status": "deleted"}


@app.get("/logs", response_model=list[LogItem])
def list_logs(
    kind: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(DetectionLog).filter(DetectionLog.user_id == current_user.id)
    if kind:
        query = query.filter(DetectionLog.kind == kind)
    logs = query.order_by(DetectionLog.created_at.desc()).limit(200).all()

    return [
        LogItem(
            id=l.id,
            kind=l.kind,
            label=l.label,
            confidence=l.confidence,
            created_at=l.created_at,
        )
        for l in logs
    ]
