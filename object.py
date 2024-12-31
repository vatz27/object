import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from deepface import DeepFace
from supabase import create_client, Client
import datetime
from typing import Dict, List
from dataclasses import dataclass
import base64
from geopy.geocoders import Nominatim
import tensorflow as tf
import tempfile
import logging
import shutil
import traceback
from PIL import Image
import io

# Suppress TensorFlow warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LocationData:
    latitude: float
    longitude: float
    image: str
    is_checkout: bool = False
    
    @staticmethod
    def validate_latitude(v: float) -> float:
        if not -90 <= v <= 90:
            raise ValueError('Invalid latitude value')
        return v
    
    @staticmethod
    def validate_longitude(v: float) -> float:
        if not -180 <= v <= 180:
            raise ValueError('Invalid longitude value')
        return v
    
    @staticmethod
    def validate_image(v: str) -> str:
        try:
            base64.b64decode(v)
        except:
            raise ValueError('Invalid base64 image')
        return v

@dataclass
class FacultyData:
    faculty_id: str
    name: str
    department: str
    image: str

@dataclass
class AttendanceHistory:
    faculty_id: str
    start_date: str
    end_date: str

app = Flask(__name__)
CORS(app)

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
        backends = ['retinaface', 'mtcnn', 'opencv', 'ssd']
        
        for backend in backends:
            try:
                faces = DeepFace.extract_faces(
                    img_path=image_path,
                    enforce_detection=True,
                    detector_backend=backend
                )
                if len(faces) > 0:
                    img = cv2.imread(image_path)
                    face = faces[0]
                    face_region = img[
                        face['facial_area']['y']:face['facial_area']['y'] + face['facial_area']['h'],
                        face['facial_area']['x']:face['facial_area']['x'] + face['facial_area']['w']
                    ]
                    
                    if face['facial_area']['w'] < 100 or face['facial_area']['h'] < 100:
                        continue
                        
                    if face.get('confidence', 0) < 0.9:
                        continue
                        
                    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                    blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if blur_value < 100:
                        continue
                        
                    return True
            except:
                continue
                
        return False
    except Exception as e:
        logger.error(f"Face detection failed: {str(e)}")
        return False

def preprocess_image(image_path: str) -> str:
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced)
        edges = cv2.Canny(denoised, 50, 150)
        enhanced_edges = cv2.addWeighted(denoised, 0.7, edges, 0.3, 0)
        enhanced_bgr = cv2.cvtColor(enhanced_edges, cv2.COLOR_GRAY2BGR)
        
        alpha = 1.2
        beta = 10
        adjusted = cv2.convertScaleAbs(enhanced_bgr, alpha=alpha, beta=beta)
        
        preprocessed_path = image_path.replace('.jpg', '_preprocessed.jpg')
        cv2.imwrite(preprocessed_path, adjusted)
        
        return preprocessed_path
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        return image_path

def find_matching_face(img_path: str, confidence_threshold: float = 0.6) -> tuple:
    try:
        models = ['VGG-Face', 'Facenet', 'OpenFace']
        metrics = ['cosine', 'euclidean']
        
        best_match = None
        best_confidence = 0
        
        for model in models:
            for metric in metrics:
                try:
                    result = DeepFace.find(
                        img_path=img_path,
                        db_path="faculty_photos_db",
                        enforce_detection=False,
                        model_name=model,
                        distance_metric=metric,
                        detector_backend='retinaface'
                    )
                    
                    if isinstance(result, list) and len(result) > 0 and not result[0].empty:
                        df_result = result[0]
                        df_result = df_result.sort_values('distance')
                        match = df_result.iloc[0]
                        
                        confidence = 1 - float(match['distance'])
                        if confidence > best_confidence:
                            identity_path = match['identity']
                            faculty_id = os.path.splitext(os.path.basename(identity_path))[0]
                            best_match = faculty_id
                            best_confidence = confidence
                            
                except Exception as e:
                    logger.error(f"Error with model {model}, metric {metric}: {str(e)}")
                    continue
                    
        if best_confidence > confidence_threshold:
            return best_match, best_confidence
            
        return None, 0
        
    except Exception as e:
        logger.error(f"Face matching failed: {str(e)}")
        return None, 0

@app.route("/faculty_attendance", methods=['POST'])
def faculty_attendance():
    temp_file_path = None
    preprocessed_path = None
    try:
        data = request.get_json()
        location_data = LocationData(**data)
        
        # Validate location data
        location_data.validate_latitude(location_data.latitude)
        location_data.validate_longitude(location_data.longitude)
        location_data.validate_image(location_data.image)
        
        # Save image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            image_data = base64.b64decode(location_data.image)
            temp_file.write(image_data)
            temp_file_path = temp_file.name
            
        preprocessed_path = preprocess_image(temp_file_path)
        
        if not verify_face_in_image(preprocessed_path):
            return jsonify({
                'success': False,
                'message': "No clear face detected. Please ensure good lighting and try again.",
                'data': {}
            })
            
        try:
            location = geolocator.reverse(f"{location_data.latitude}, {location_data.longitude}")
            address = location.address if location else "Unknown Location"
        except Exception as e:
            logger.error(f"Location lookup failed: {str(e)}")
            address = "Location lookup failed"
            
        faculty_id, confidence = find_matching_face(preprocessed_path)
        
        if faculty_id:
            faculty_info_query = supabase.table('faculty_info').select("*").eq('faculty_id', faculty_id).execute()
            
            if not faculty_info_query.data:
                return jsonify({
                    'success': False,
                    'message': "Faculty information not found in database",
                    'data': {}
                })
                
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
            
            existing_attendance = supabase.table('faculty_attendance')\
                .select("*")\
                .eq('faculty_id', faculty_id)\
                .gte('check_in_time', today_start.isoformat())\
                .is_('check_out_time', 'null')\
                .execute()
                
            if location_data.is_checkout:
                if not existing_attendance.data:
                    return jsonify({
                        'success': False,
                        'message': "No active check-in found. Please check-in first.",
                        'data': {}
                    })
                    
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
                    
                return jsonify({
                    'success': True,
                    'message': f"Goodbye {faculty_info['name']}! Check-out recorded successfully.",
                    'data': attendance_data
                })
            else:
                if existing_attendance.data:
                    return jsonify({
                        'success': False,
                        'message': "Already checked in. Please check-out first.",
                        'data': {}
                    })
                    
                attendance_data['check_in_time'] = current_time.isoformat()
                supabase.table('faculty_attendance').insert(attendance_data).execute()
                
                return jsonify({
                    'success': True,
                    'message': f"Welcome {faculty_info['name']}! Check-in recorded successfully.",
                    'data': attendance_data
                })
                
        return jsonify({
            'success': False,
            'message': "No matching faculty found. Please try again or contact administrator.",
            'data': {}
        })
        
    except Exception as e:
        logger.error(f"Attendance error: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'message': str(e),
            'data': {}
        })
    finally:
        for path in [temp_file_path, preprocessed_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    logger.error(f"Error cleaning up temporary file: {str(e)}")

@app.route("/register_faculty", methods=['POST'])
def register_faculty():
    try:
        data = request.get_json()
        faculty_data = FacultyData(**data)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            image_data = base64.b64decode(faculty_data.image)
            temp_file.write(image_data)
            temp_file_path = temp_file.name

        if not verify_face_in_image(temp_file_path):
            os.unlink(temp_file_path)
            return jsonify({
                'success': False,
                'message': "No clear face detected in image. Please try again.",
                'data': {}
            })

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
        
        return jsonify({
            'success': True,
            'message': "Faculty registered successfully",
            'data': faculty_info
        })
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"Registration failed: {str(e)}",
            'data': {}
        })

@app.route("/get_attendance_history", methods=['POST'])
def get_attendance_history():
    try:
        data = request.get_json()
        history_request = AttendanceHistory(**data)
        
        attendance_records = supabase.table('faculty_attendance')\
            .select("*")\
            .eq('faculty_id', history_request.faculty_id)\
            .gte('check_in_time', history_request.start_date)\
            .lte('check_in_time', history_request.end_date)\
            .execute()

        if not attendance_records.data:
            return jsonify({
                'success': True,
                'message': "No attendance records found for the specified period",
                'data': {'records': []}
            })

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

        return jsonify({
            'success': True,
            'message': "Attendance history retrieved successfully",
            'data': {'records': formatted_records}
        })

    except Exception as e:
        logger.error(f"Error retrieving attendance history: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"Error retrieving attendance history: {str(e)}",
            'data': {}
        })

# Initialize the database when creating the app
initialize_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
