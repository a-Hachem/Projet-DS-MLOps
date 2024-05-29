from fastapi import FastAPI,APIRouter, Depends, HTTPException, status, Request, UploadFile, Form, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext
import secrets
from typing import Optional, List
import pandas as pd
from pydantic import BaseModel, Field
import os
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import uuid  
from pathlib import Path

import logging
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
import time
import sqlite3
from sqlite3 import Error

# Je teste l'api sur la machine Airflow
# =============================================================================
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',
    filemode='a'
)
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        start_time = time.time()
        try:
            response = await call_next(request)
        except Exception as e:
            # Log the exception and re-raise it
            logging.error(f"Error during request {request.url}: {str(e)}")
            raise e from None
        process_time = time.time() - start_time
        
        # Log request and response details
        logging.info(f"Request {request.method} {request.url} completed with status {response.status_code} in {process_time:.2f}s")
        log_end_signature = "="*50
        logging.info(f"End of a request: {log_end_signature}")
        
        return response

# =============================================================================
# ===> Ensure Appending to Log File if it already exists

def setup_logging():
    logger = logging.getLogger('uvicorn')
    handler = logging.FileHandler('app.log', mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Call this function at the very start of your main script
setup_logging()

# =============================================================================
# Initialize FastAPI and add middleware
api = FastAPI(title="Projet E-commerce session de Décembre 2023")
api.add_middleware(LoggingMiddleware)
# =============================================================================

class User(BaseModel):
    username: str
    password: str

# =============================================================================
# =====> Feedbacks

# ===> Class for feedbacks
class Feedback(BaseModel):
    user_id: str = Form(..., example="12345")
    image_id: str = Form(..., example="98765")
    feedback: str = Form(..., example="Great image quality!")
    rating: int = Form(..., example=5, ge=1, le=5)
#  Function to save feed backs

def create_connection(db_file):
    """Create a database connection to a SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(f"Error connecting to database: {e}")
    return conn

def create_table(conn):
    """Create a feedback table if it doesn't already exist."""
    try:
        sql_create_feedback_table = """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            image_id TEXT NOT NULL,
            feedback TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5)
        );
        """
        cursor = conn.cursor()
        cursor.execute(sql_create_feedback_table)
    except Error as e:
        print(f"Error creating table: {e}")

def save_feedback_to_db(user_id, image_id, feedback, rating):

    """Save feedback to the database."""
    database = "feedback.db"
    conn = create_connection(database)
    if conn is not None:
        create_table(conn)  # Ensure the feedback table exists
        try:
            sql_insert_feedback = """
            INSERT INTO feedback (user_id, image_id, feedback, rating)
            VALUES (?, ?, ?, ?);
            """
            cursor = conn.cursor()
            cursor.execute(sql_insert_feedback, (user_id, image_id, feedback, rating))
            conn.commit()
        except Error as e:
            print(f"Error saving feedback: {e}")
        finally:
            conn.close()
    else:
        print("Error! Cannot create the database connection.")

# =============================================================================

# Initialize FastAPI and security
# api = FastAPI(title="Projet E-commerce session de Décembre 2023") # For the main API
router = APIRouter() # For the ADMINs
security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# =============================================================================
# ===> Define users and their passwords

users_db = {
    "rakutenuser": {
        "username": "rakutenuser",
        "hashed_password": pwd_context.hash('MlopRakutenUser2024'),
    },
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash('MlopRakutenAdmin2024'),
    }
}

# =============================================================================
# =============================================================================

# ===> Authentication for users

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    user = users_db.get(username)
    if not user or not pwd_context.verify(credentials.password, user['hashed_password']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return user['username']  # Ou retourner l'objet utilisateur complet si nécessaire

# ===> Authentication for admins

# Fonction pour obtenir l'utilisateur actuel
def get_current_admin(credentials: HTTPBasicCredentials = Depends(security)):
    if ((credentials.username == "admin") and (credentials.password == "MlopRakutenAdmin2024")):  
        return credentials.username
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nom d'utilisateur ou mot de passe ADMIN incorrect",
            headers={"WWW-Authenticate": "Basic"},
        )

# =============================================================================
# =============================================================================
# ===> Customized function to preprocess a given image

def preprocessImage(img):
    img_resized = img.resize((224, 224))
    img_array = np.asarray(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

# =============================================================================
# =============================================================================
# ===> Customized function to load ML model => update the file name !!!

def MLmodel_load(MODEL_PATH_nalme):
    
    try:
        # model_ml = load_model(dir_model, compile=False)
        model_ml = tf.keras.models.load_model(MODEL_PATH_nalme, compile=False)
        return model_ml
    except Exception as e:
        print(f"Erro when loading the model : {e}")
        raise HTTPException(status_code=500, detail="Le modèle n'a pas pu être chargé.")

# =============================================================================
# =============================================================================

# Function to save the updated dataframe to a CSV file
def save_updated_dataframe(df, file_path):
    try:
        df.to_csv(file_path, index=False, encoding='utf-8')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save the updated dataframe: {e}")

# =============================================================================
# =============================================================================

# ===> Read questions data (Data base)
dtype_dict = {
    'Image': 'str',
    'Label_name': 'str',
    'Label': 'str',
}

def readDataframe():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(BASE_DIR, 'data_photos.csv')
    # df_db = pd.read_csv(csv_file_path, encoding='utf-8', dtype=dtype_dict)
    df_db = pd.read_csv(csv_file_path, encoding='utf-8')

    return df_db

# =============================================================================
# =============================================================================
# ===> New data file if a new question added successfully

# current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.getcwd()

# Choose where to save new image

AddnewImageToOldDb = True

if AddnewImageToOldDb:
    
    newDataFile = os.path.join(current_dir, "data_photos.csv")
else:
    newDataFile = os.path.join(current_dir, "data_photos_update.csv")

# ===> Function to save requests

def save_request_results_to_csv(input_data, filename):
    # Define the path to the file
    file_path = os.path.join(current_dir, filename)
    
    # Convert the dictionary to a DataFrame
    new_data_df = pd.DataFrame([input_data])
    
    if os.path.exists(file_path):
        # File exists, load the existing data and append the new data
        df = pd.read_csv(file_path)
        updated_df = pd.concat([df, new_data_df], ignore_index=True)
    else:
        # File does not exist, create a new DataFrame
        updated_df = new_data_df
    
    # Save the updated DataFrame to CSV
    updated_df.to_csv(file_path, index=False, encoding='utf-8')

# =============================================================================
# =============================================================================

# ===> Status root

@api.get('/status')
async def get_status():
    """_summary_

    Returns:
        The API status
    """
    return {"status": "Application is running ..."}


# =============================================================================
# =============================================================================

@router.post("/prediction/", status_code=status.HTTP_201_CREATED)
async def prediction(file: UploadFile, 
                     username: str = Depends(get_current_admin)):
    
    if username != "admin":
        raise HTTPException(status_code=403, detail="Only admin is allowed to create a new question !!!")
    
    else:
    
        # Line dashe delitimiting the end of a request
        
        try:
            imag = await file.read()
            imag_open = Image.open(io.BytesIO(imag))
            imag_process = preprocessImage(imag_open)
            
            # ML model name
            BASE_DIR_model = current_dir+"/modelSaved"
            model_ML_fname = os.path.join(BASE_DIR_model, "model1_best_weights.h5")
            
            # Make prediction
            model_pretrained = MLmodel_load(model_ML_fname)
            predictions = model_pretrained.predict(imag_process)
            predicted_class = np.argmax(predictions, axis=1)[0]  # Simplified access as single prediction
            predicted_score = np.max(predictions, axis=1)[0]

            data_photos_df = readDataframe()
            new_image_name = f"{uuid.uuid4().hex}.jpg"
            
            Save_Newimage = True
            if Save_Newimage:
                image_path = os.path.join(current_dir, new_image_name)
                with open(image_path, "wb") as image_file:
                    image_file.write(imag)
                    
            # Prepare feedback data
            user_id = username  # Assuming username can be used as user_id
            image_id = new_image_name
            
            new_Label_name = data_photos_df["Label_name"][data_photos_df["Label"]== predicted_class].values[0]
            new_data = {"Image": new_image_name, "Label_name": new_Label_name, "Label": predicted_class}
            new_df = pd.DataFrame([new_data])
            data_photos_df = pd.concat([data_photos_df, new_df], ignore_index=True)
            save_updated_dataframe(data_photos_df, newDataFile)
            
            prediction_results = {
                "User"      : username,
                "Predicted class": str(predicted_class),
                "Prediction score": str(predicted_score),
                "Category of the image": new_Label_name,
                "New image saved": new_image_name
            }
            
            # Save the prediction results to CSV
            prediction_results_to_save = prediction_results
            save_request_results_to_csv(prediction_results_to_save, "request_results.csv")
            
            # -----------------------------
            # Evaluate the current global prediction score average
            all_request_df = pd.read_csv(current_dir+"/request_results.csv")
            score_average = all_request_df["Prediction score"].mean()
            # print("score_average = ", score_average)
            # score_average =  0.4
            
            # A warning message will be returned if the prediction score average is very
            # 
            ref_score = 0.5
            
            if (score_average < ref_score):
                prediction_results.setdefault("WARNING !!!" , "Global prediction score very low, new model training required !!!")
                        
            # Log the detailed prediction results
            logging.info(f"Prediction results: {prediction_results}")
            logging.info(f"Current global prediction score average: {score_average}")
            # logging.info(f"Current global prediction score average: {score_average}")
            
            return JSONResponse(content=prediction_results)
        except Exception as err:
            logging.error(f"Prediction or database update error: {err}")
            raise HTTPException(status_code=500, detail=f"Prediction or database update error: {err}")
        finally:
            await file.close()
        
# =============================================================================
# =============================================================================
# ===> Feedback endpoint

@router.post("/feedback/", status_code=status.HTTP_201_CREATED)
async def receive_feedback( user_id: str = Query(None,  description="Nom d'utilisateur:", enum=["admin", "non admin"]),                            
                            feedback: str = Query(None,  description="A quelle catégorie l'image appartient le mieux ?",\
                            enum=["Watches", "Home Furnishing", "Baby Care", "Home Decor & Festive Needs", "Kitchen & Dining", \
                                  "Beauty and Personal Care", "Computers", "Other" ]),
                            rating: int = Query(None,  description="Quelle note donneriez-vous ?", enum=[1, 2, 3, 4, 5]),
                            image_id: str = Form(str, example= "a0571f9fc03f9bf78c19b461be33da37", description="Copier et coller l'identifiant de l'image via la requête:")):

    
    if ((rating < 1) or (rating > 5)):
        raise HTTPException(status_code=404, detail="The rating must be > 0 and <= 5")
    else:
        try:
            # Save feedbacks
            save_feedback_to_db(user_id, image_id,  feedback, rating)
            # Acknowledge the receipt of feedback
            return {"message": "Feedback received successfully, thank you for the feedback !"}
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to process feedback")

# ===> Include the router into the main API

api.include_router(router)

