# Parkinson Detection – Deep Learning  

This project predicts whether a patient is **Normal** or has **Parkinson’s Disease** using a **deep learning model (CNN with TensorFlow/Keras)**.  
It includes a **Flask-based web application** for user-friendly medical image prediction.  

---

## 🚀 Features  
- Predicts **Parkinson’s Disease vs Normal** from uploaded medical images.  
- Preprocesses images using **OpenCV** and **TensorFlow image utilities**.  
- Uses a trained **Keras `.h5` model** for classification.  
- Flask web app (`app.py` / `health.py`) for interactive prediction.  
- REST API endpoint (`/upload`) for real-time prediction.  
- Saves uploaded images in the `uploads/` folder for reference.  

---

## 🛠️ Tech Stack  
- Python  
- Flask  
- TensorFlow / Keras  
- OpenCV  
- NumPy  

---

## 📊 Model  
- **Input**: Medical image (e.g., handwriting, drawing, or scan used in Parkinson’s detection research)  
- **Preprocessing**: Resize → Normalize → Convert to RGB  
- **Output**: Binary classification → **Parkinson / Normal**  
- **Threshold values**:  
  - In `app.py`: `prediction >= 0.049` → Parkinson  
  - In `health.py`: `prediction >= 0.1` → Parkinson  

---

## 🔧 How to Run  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/parkinson-detection-webapp.git
   cd parkinson-detection-webapp
