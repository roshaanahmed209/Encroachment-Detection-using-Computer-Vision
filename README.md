# 🛰️ Smart Land Encroachment Analysis System

A web-based AI-powered system for detecting land encroachment by comparing user-submitted Google Maps images with official approved sitemaps. This project leverages **YOLOv8m segmentation** and **template matching** to automatically highlight unauthorized construction and assist urban planners with scalable, accurate analysis.

---

## 🚀 Project Overview

This system allows users to upload satellite imagery (e.g., from Google Maps) and checks it against the corresponding approved government sitemap. Unauthorized land use is identified and visually marked using segmentation masks and similarity-based comparison.

It’s built to help automate what would traditionally be a time-consuming, manual process — offering speed, accuracy, and insight for urban planning authorities.

---

## 💡 Key Features

✅ Upload Google Maps image and official sitemap  
✅ Semantic segmentation with **YOLOv8m**  
✅ **Template matching** and **cosine similarity** for mask comparison  
✅ Automatic detection and highlight of encroachments  
✅ Visual comparison and output rendering  
✅ **Login/Authentication system** with SQLite  
✅ Admin/user separation for secure access and report generation

---

## 🛠️ Tech Stack

- **Backend**: Python  
- **Segmentation**: [Ultralytics YOLOv8m](https://github.com/ultralytics/ultralytics)  
- **Dataset Annotation**: [Roboflow](https://roboflow.com/)  
- **Frontend**: HTML, CSS, JavaScript  
- **Database**: SQLite (for login credential validation)  
- **Image Comparison**: Template Matching + Cosine Similarity

---

## 📈 Performance Metrics

- 🧠 **Segmentation mAP**: 89%  
- 🎯 **Precision**: 87% | **Recall**: 85.6% | **F1 Score**: 86%  
- 🧩 **Template Matching Accuracy**: 90%  
- 📊 **Overall System Accuracy**: 89%

---

## 📁 Project Structure

smart_land_encroachment/
├── static/ # CSS, JS, assets
├── templates/ # HTML templates
├── app.py # Flask application
├── yolov8_segmentation.py # YOLO model integration
├── matcher.py # Template matching logic
├── database.db # SQLite database for login
├── roboflow_config.json # Dataset config (optional)
├── uploads/ # User-uploaded images
├── requirements.txt
└── README.md



---

## 🧪 How It Works

1. A user uploads a satellite image and the corresponding approved sitemap.
2. **YOLOv8m** generates masks for key regions in both images.
3. The system uses **template matching** to compare and identify mismatches.
4. Encroached areas are highlighted in the result image.
5. Reports are stored and visual results are shown to the user/admin.

---

## 📦 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/roshaanahmed209/Encroachment-Detection-using-Computer-Vision
cd Smart-Land-Encroachment-Analysis
```

2. Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Run the application
python app.py
