# 🧍‍♂️📸 Human Activity Recognition (HAR) from Images

This project performs **Human Activity Recognition** using **image data**. It uses machine learning (or deep learning) to classify images into categories like walking, sitting, standing, etc., based on visual features.

## 📦 Project Structure

├── data/ # Folder for image datasets (train/val/test)
├── models/ # Trained model files
├── app.py # Streamlit app (visual demo)
├── main.py # Model training and evaluation
├── utils.py # Helper functions
├── requirements.txt # Python dependencies
└── README.md # You're reading this :)


## 🧠 Activities Recognized

- 🚶‍♂️ WALKING  
- 🧍 STANDING  
- 🪑 SITTING  
- 🛌 LAYING  
- 🧗 WALKING_UPSTAIRS  
- ⬇️ WALKING_DOWNSTAIRS  

## 🖼️ Dataset

We use a dataset of labeled human activity images. Examples may come from public sources or be custom-collected.

> **Alternative**: You can also adapt sensor-based datasets (e.g., UCI HAR) to synthetic images using plotting or pose estimation models.

## 📊 Model

- CNN-based image classifier (e.g., `ResNet`, `MobileNet`, or custom CNN)
- Trained using PyTorch or TensorFlow
- Metrics: accuracy, precision, recall, F1-score

## 🚀 How to Run

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
2. **Train the model** 
python main.py

3. **Launch Streamlit app:**
streamlit run app.py


💡 Features

    Upload your own activity images and get predictions in real time

    Visualize model performance

    Extendable for more activity classes

