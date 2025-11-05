from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("digit.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        # Get 64 pixel intensity values (comma separated)
        pixels = request.form["pixels"]
        
        # Convert to list of floats
        pixel_list = [float(x) for x in pixels.split(",")]
        
        # Ensure correct length (8x8 = 64)
        if len(pixel_list) != 64:
            prediction = "Please enter exactly 64 pixel values!"
        else:
            # Reshape and predict
            data = np.array(pixel_list).reshape(1, -1)
            result = model.predict(data)
            prediction = f"Predicted Digit: {result[0]}"

    return render_template("home.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
