🔢 Handwritten Digit Recognition using Flask and Machine Learning
🧠 Overview

This project demonstrates how to deploy a machine learning model using Flask to predict handwritten digits (0–9).
The model is trained using the Scikit-learn Digits Dataset, which contains 8×8 grayscale images of handwritten numbers.

The trained model is saved as digit.pkl, and a simple web interface allows users to input pixel intensity values and get the predicted digit in real time.

📂 Project Structure
digit_project/
│
├── app.py               # Flask backend script
├── digit.pkl            # Trained model file (pickle format)
├── requirements.txt     # Python dependencies
└── templates/
    └── home.html        # Frontend HTML template

🧩 How It Works
1️⃣ Dataset — load_digits()

Built-in Scikit-learn dataset of handwritten digits.

Each image = 8×8 pixels, total 64 features per sample.

Pixel intensity range = 0–16.

Total 1797 samples, each labeled as digit 0–9.

2️⃣ Model Training

A Logistic Regression model is trained on this dataset and then serialized (saved) using Python’s pickle module.

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
import pickle

digits = load_digits()
X, y = digits.data, digits.target

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

with open("digit.pkl", "wb") as file:
    pickle.dump(model, file)

3️⃣ Flask Deployment

The Flask app:

Loads the trained model (digit.pkl)

Accepts 64 comma-separated pixel values (0–16)

Predicts the most likely digit using model.predict()

Displays the result on the web page

🖥️ User Interface (home.html)

The HTML interface provides a simple input form:

Field	Description
Pixels Input	64 pixel intensity values (comma separated)
Predict Button	Sends data to Flask backend
Output	Displays predicted digit (0–9)

Example Input:

0,0,10,14,8,1,0,0,0,2,16,14,6,15,2,0,0,0,11,16,16,16,8,0,...


Output:

Predicted Digit: 3

⚙️ Installation and Setup
1️⃣ Clone the Repository
git clone https://github.com/Ashok8601/project5
cd project5

2️⃣ Create a Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate      # Windows
# or
source venv/bin/activate   # Linux/Mac

3️⃣ Install Dependencies

Create a requirements.txt file:

flask
numpy
scikit-learn
pickle-mixin


Then install:

pip install -r requirements.txt

4️⃣ Run the Application
python app.py


Then open your browser and visit:
👉 http://127.0.0.1:5000/

🧪 Example Prediction
Sample	Actual Digit	Model Prediction
790th Image	3	✅ 3
1205th Image	7	✅ 7
1786th Image	0	✅ 0
💻 Technologies Used
Component	Description
Python 3.x	Programming language
Flask	Web framework for deployment
Scikit-learn	Machine Learning model training
Pickle	Model serialization
HTML & CSS	Frontend user interface
🧑‍🏫 Learning Outcomes

Train and serialize a machine learning model using pickle

Deploy the model using Flask

Accept and process user input through web forms

Bridge the gap between machine learning and web development

🌐 Future Enhancements

Add 8×8 grid input UI instead of text input (for pixel visualization)

Allow image upload for digit recognition using OpenCV

Deploy on Render / Railway / Heroku for public access

🏁 Conclusion

This project is a perfect starting point to understand how machine learning models can be integrated with Flask web apps.



                                                                                                           Author :
                                                                                                       Ashok Kumar Yadav
It shows the complete workflow — from model training to real-time prediction via browser interface.

🚀 Bridging AI with the Web — one project at a time!
