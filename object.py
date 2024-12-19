import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from deepface import DeepFace
from supabase import create_client, Client
import datetime
from typing import Dict, List
from pydantic import BaseModel
import base64
from geopy.geocoders import Nominatim
import tensorflow as tf
import tempfile
import logging
import shutil
import traceback

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocationData(BaseModel):
    latitude: float
    longitude: float
    image: str
    is_checkout: bool = False

class Response(BaseModel):
    success: bool
    message: str = ""
    data: Dict = {}

class FacultyData(BaseModel):
    faculty_id: str
    name: str
    department: str
    image: str

class AttendanceHistory(BaseModel):
    faculty_id: str
    start_date: str
    end_date: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL="https://ohzkpmpwzfklznxgficm.supabase.co"
SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9oemtwbXB3emZrbHpueGdmaWNtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzQ0NTYxNzAsImV4cCI6MjA1MDAzMjE3MH0.9JMQzYW3CGfPEV6Abp_Cm7tJfVnLgs9Xz6n4ArYUVmU"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

geolocator = Nominatim(user_agent="faculty_attendance_app")

def initialize_db():
    if not os.path.exists("faculty_photos_db"):
        os.makedirs("faculty_photos_db")
        logger.info("Created faculty_photos_db directory")

def verify_face_in_image(image_path: str) -> bool:
    try:
        face = DeepFace.extract_faces(
            img_path=image_path,
            enforce_detection=True,
            detector_backend='opencv'
        )
        return len(face) > 0
    except Exception as e:
        logger.error(f"Face detection failed: {str(e)}")
        return False

@app.post("/faculty_attendance", response_model=Response)
async def faculty_attendance(location_data: LocationData):
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            image_data = base64.b64decode(location_data.image)
            temp_file.write(image_data)
            temp_file_path = temp_file.name

        if not verify_face_in_image(temp_file_path):
            return Response(
                success=False,
                message="No clear face detected. Please try again.",
                data={}
            )
        
        try:
            location = geolocator.reverse(f"{location_data.latitude}, {location_data.longitude}")
            address = location.address if location else "Unknown Location"
        except Exception as e:
            logger.error(f"Location lookup failed: {str(e)}")
            address = "Location lookup failed"
        
        try:
            result = DeepFace.find(
                img_path=temp_file_path,
                db_path="faculty_photos_db",
                enforce_detection=True,
                model_name="VGG-Face",
                distance_metric="cosine",
                detector_backend='opencv'
            )
            
            if isinstance(result, list) and len(result) > 0 and not result[0].empty:
                df_result = result[0]
                df_result = df_result.sort_values('distance')
                best_match = df_result.iloc[0]
                
                identity_path = best_match['identity']
                faculty_id = os.path.splitext(os.path.basename(identity_path))[0]
                confidence = 1 - float(best_match['distance'])
                
                if confidence > 0.7:
                    faculty_info_query = supabase.table('faculty_info').select("*").eq('faculty_id', faculty_id).execute()
                    
                    if not faculty_info_query.data:
                        return Response(
                            success=False,
                            message="Faculty information not found in database",
                            data={}
                        )
                    
                    faculty_info = faculty_info_query.data[0]
                    current_time = datetime.datetime.now()
                    today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                    
                    attendance_data = {
                        'faculty_id': faculty_id,
                        'name': faculty_info['name'],
                        'department': faculty_info['department'],
                        'latitude': location_data.latitude,
                        'longitude': location_data.longitude,
                        'location_address': address,
                        'confidence': confidence
                    }

                    # Check existing attendance
                    existing_attendance = supabase.table('faculty_attendance')\
                        .select("*")\
                        .eq('faculty_id', faculty_id)\
                        .gte('check_in_time', today_start.isoformat())\
                        .is_('check_out_time', 'null')\
                        .execute()

                    if location_data.is_checkout:
                        if not existing_attendance.data:
                            return Response(
                                success=False,
                                message="No active check-in found. Please check-in first.",
                                data={}
                            )
                        
                        # Update check-out information
                        update_data = {
                            'check_out_time': current_time.isoformat(),
                            'check_out_latitude': location_data.latitude,
                            'check_out_longitude': location_data.longitude,
                            'check_out_address': address
                        }
                        
                        supabase.table('faculty_attendance')\
                            .update(update_data)\
                            .eq('id', existing_attendance.data[0]['id'])\
                            .execute()
                        
                        return Response(
                            success=True,
                            message=f"Goodbye {faculty_info['name']}! Check-out recorded successfully.",
                            data=attendance_data
                        )
                    else:
                        if existing_attendance.data:
                            return Response(
                                success=False,
                                message="Already checked in. Please check-out first.",
                                data={}
                            )
                        
                        # Create new check-in record
                        attendance_data['check_in_time'] = current_time.isoformat()
                        supabase.table('faculty_attendance').insert(attendance_data).execute()
                        
                        return Response(
                            success=True,
                            message=f"Welcome {faculty_info['name']}! Check-in recorded successfully.",
                            data=attendance_data
                        )
                
                return Response(
                    success=False,
                    message="Face match confidence too low. Please try again.",
                    data={}
                )
            
            return Response(
                success=False,
                message="No matching faculty found. Please try again.",
                data={}
            )
            
        except Exception as e:
            logger.error(f"Face recognition error: {traceback.format_exc()}")
            return Response(
                success=False,
                message=f"Face recognition error: {str(e)}. Please try again.",
                data={}
            )
            
    except Exception as e:
        logger.error(f"Attendance error: {traceback.format_exc()}")
        return Response(
            success=False,
            message=str(e),
            data={}
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {str(e)}")

@app.post("/register_faculty", response_model=Response)
async def register_faculty(faculty_data: FacultyData):
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            image_data = base64.b64decode(faculty_data.image)
            temp_file.write(image_data)
            temp_file_path = temp_file.name

        if not verify_face_in_image(temp_file_path):
            os.unlink(temp_file_path)
            return Response(
                success=False,
                message="No clear face detected in image. Please try again.",
                data={}
            )

        faculty_id = faculty_data.faculty_id
        photo_path = f"faculty_photos_db/{faculty_id}.jpg"
        
        shutil.move(temp_file_path, photo_path)
        
        faculty_info = {
            'faculty_id': faculty_id,
            'name': faculty_data.name,
            'department': faculty_data.department,
            'photo_path': photo_path
        }
        
        result = supabase.table('faculty_info').insert(faculty_info).execute()
        
        try:
            DeepFace.find(
                img_path=photo_path,
                db_path="faculty_photos_db",
                enforce_detection=False,
                model_name="VGG-Face",
            )
            logger.info(f"Registered new faculty: {faculty_id}")
        except Exception as e:
            logger.error(f"Error initializing face database: {str(e)}")
        
        return Response(
            success=True,
            message="Faculty registered successfully",
            data=faculty_info
        )
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return Response(
            success=False,
            message=f"Registration failed: {str(e)}",
            data={}
        )

@app.post("/get_attendance_history", response_model=Response)
async def get_attendance_history(request: AttendanceHistory):
    try:
        attendance_records = supabase.table('faculty_attendance')\
            .select("*")\
            .eq('faculty_id', request.faculty_id)\
            .gte('check_in_time', request.start_date)\
            .lte('check_in_time', request.end_date)\
            .execute()

        if not attendance_records.data:
            return Response(
                success=True,
                message="No attendance records found for the specified period",
                data={'records': []}
            )

        formatted_records = []
        for record in attendance_records.data:
            formatted_record = {
                'date': record['check_in_time'].split('T')[0],
                'check_in_time': record['check_in_time'],
                'check_in_location': record['location_address'],
                'check_out_time': record['check_out_time'],
                'check_out_location': record.get('check_out_address', 'Not checked out'),
                'department': record['department'],
                'name': record['name']
            }
            formatted_records.append(formatted_record)

        return Response(
            success=True,
            message="Attendance history retrieved successfully",
            data={'records': formatted_records}
        )

    except Exception as e:
        logger.error(f"Error retrieving attendance history: {str(e)}")
        return Response(
            success=False,
            message=f"Error retrieving attendance history: {str(e)}",
            data={}
        )

@app.on_event("startup")
async def startup_event():
    initialize_db()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
