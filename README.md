# Image Classification Platform using Neural Networks

## 👥 Group Members
- **Donfack Tsopack Yves Dylane**
- **Eba Bokalli Anna**
- **Hamadou Ivang Kembeng**
- **Djomegni Tanteh Eric Lionel**

### 🔗 [GitHub Repository](https://github.com/yvesdylane/Image-Classification-Platform-using-Neural-Network/tree/main) | [Docker Repository](https://hub.docker.com/repository/docker/yvesdylane/g7-image-recognition-api/general)

## 📌 Introduction
This project is an end-to-end image classification platform built using a neural network. The primary objective was to integrate various technologies to construct a complete pipeline that enables users to classify images through a trained model, an API, and a web interface. The lab covered essential concepts, including:
- 📊 Model training
- 🌐 API development
- 🎨 Web interface design
- 📦 Containerization with Docker
- 🖥️ Local deployment simulation

By dividing the project into multiple stages, the team gained practical experience in real-world machine learning application development.

## 🎯 Objective
To design and develop an end-to-end platform for classifying images using a neural network in a collaborative group setting.

## 🛠️ Tools & Technologies
- 🐍 **Python** (VSCode as IDE)
- 🖼️ **Trained image recognition model**
- 🚀 **Flask** for serving the model
- 🐳 **Docker** for containerizing the application
- 📷 **OpenCV** for image preprocessing
- 🌍 **Local server** for deployment simulation
- 🧪 **Postman/cURL** for API testing

### 📂 Dataset
We utilized the pre-trained model from **Fashionminst dataset** for image recognition.

---

## 📝 Lab Steps and Execution
### **🔹 Step 1: Prepare the Image Recognition Model**
#### 👤 **Work by: Donfack Tsopack Yves Dylane**

#### 📚 **Training the Model**
A class was created to manage the neural network model, allowing for initialization, training, and saving for future predictions. The class included key methods such as:
- 🔧 `__init__`: Initializes clustering parameters, feature storage, and preprocessing components.
- 📂 `load_dataset()`: Loads the Fashion MNIST dataset.
- 🖼️ `display_sample_images()`: Displays sample images for visualization.
- 🔍 `preprocess_data()`: Standardizes and reduces dataset complexity.
- 🏷️ `perform_clustering()`: Groups images using a clustering algorithm.
- 📊 `visualize_clusters()`: Generates visual representations of the clustered groups.
- 🔑 `get_cluster_representatives()`: Identifies representative images for each cluster.
- 🖼️ `visualize_cluster_representatives()`: Displays key cluster representatives.
- 🏷️ `predict_new_image()`: Assigns a new image to the most similar cluster.
- 💾 `save_model()`: Saves the trained model in **.joblib** format for easy loading.
- 🔄 `load_model()`: Reloads the model for further use.

The model successfully identified two distinct clusters in the dataset.

---

### **🔹 Step 2: Create the API Using Flask**
#### 👤 **Work by: Djomegni Tanteh Eric Lionel**

#### 🌍 **API Development**
- 🛠️ Flask was chosen due to its simplicity and ease of use for beginners.
- 📦 Required libraries (Flask, OpenCV, joblib) were installed for handling model loading and image preprocessing.

#### 🌐 **API Endpoints**
- 🔹 `/predict` (POST): Accepts image files for classification.
- 📏 Implements image resizing and normalization before passing the image to the model.
- 🏷️ Returns predicted class labels in JSON format.

---

### **🔹 Step 2.1: Web Interface Development**
#### 👤 **Work by: Eba Bokalli Anna**

A simple **HTML/CSS/JavaScript** web interface was built to allow users to upload images and receive classification results. Features include:
- 📤 An **upload button** to submit images.
- 📊 A **results display** below the image.
- 🖥️ A user-friendly graphical interface for those unfamiliar with command-line tools like cURL or Postman.

---

### **🔹 Step 3: Dockerizing the Application**
#### 👤 **Work by: Hamadou Ivang Kembeng**

#### 📦 **Dockerfile Creation**
- The **Dockerfile** sets up the environment with required dependencies.
- 📂 Copies the application code into the container.
- 🔌 Exposes necessary ports for communication.
- 🚀 Defines the command to run the Flask application.

#### 🛠️ **Building and Running the Docker Image**
- 🏗️ Built using: `docker build -t image-classification-api .`
- 🚀 Ran using: `docker run -p 8000:8000 image-classification-api`
- ✅ Deployed successfully, with logs confirming proper execution.

---

### **🔹 Step 4: Local Deployment**
- 🐳 The **Docker container** was deployed on a local machine for simulation.
- 🛠️ Docker was installed and configured for local testing.
- 🔍 The API was tested via a web interface by submitting images and receiving predictions.

#### 🚀 **Final Deployment**
- 📤 The completed application was pushed to a Docker repository for public accessibility.

---

## 🏛️ System Architecture
The system follows a **three-layered architecture**:

1. 🏷️ **Application Layer**
   - Users interact with the system via a web interface or API from any device (mobile, desktop, etc.).

2. 🔄 **Data Processing Layer**
   - The Flask API handles incoming image data, preprocesses it, and sends it to the model for classification.

3. 🖥️ **Model Layer**
   - The trained model processes the image and returns the classification result to the API, which then sends it back to the user.

---

## ✅ Conclusion
This project successfully demonstrated the development of an end-to-end image classification platform. The team gained valuable hands-on experience in:
- 🎓 Training machine learning models.
- 🌐 Developing and deploying APIs.
- 🎨 Creating web-based user interfaces.
- 📦 Containerizing applications with Docker.

🔮 **Future improvements may include:**
- 🤖 Integration of more advanced neural network models.
- ⚡ Optimizing performance for real-time classification.
- ☁️ Deploying on cloud platforms for scalability.

This project serves as a strong foundation for further exploration into AI-powered applications.

---

## 👨‍💻 Contributors
- **Donfack Tsopack Yves Dylane** – 🖥️ Model Training
- **Djomegni Tanteh Eric Lionel** – 🌍 API Development
- **Eba Bokalli Anna** – 🎨 Web Interface Development
- **Hamadou Ivang Kembeng** – 📦 Dockerization & Deployment

For further inquiries, feel free to contact any of the group members. 🚀
