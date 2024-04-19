from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
import numpy as np
from PIL import Image
import os 
from werkzeug.utils import secure_filename
from disease_info import DISEASE_INFO 
from keras.losses import SparseCategoricalCrossentropy

# Load the model and specify the loss function

app = Flask(__name__)


@app.route('/index1')
def index1():
    # Your identification logic here
    return render_template('index1.html')
@app.route('/identify')
def identify(): 
    # Your identification logic here
    return render_template('identify.html')
# Specifying Paths for model
MODEL_PATHS = {
    "apple": "applee.h5",
    "bell pepper": "mergebellypepper.h5",
    "corn": "Corn.h5",
    "cherry": "mergecherry.h5",
    "peach": "mergepeach.h5",
    "potato": "Potato.h5",
    "strawberry": "mergestrawberry.h5",
    "tomato": "mergetomato.h5",
    "guava":"guava.h5"
}

# Specifying Class labels for each model
CLASS_LABELS = {
    "potato": ['Potato___Early_blight', 'Potato___Late_blight', 'Healthy'],
    "apple": ['apple scab', 'black rot'],
    "corn": ['common rust', 'Healthy', 'northern leaf blight'],
    "strawberry": ['Healthy', 'leaf scorch'],
    "cherry":['Healthy', 'Powdery Mildew'],
    "tomato":['Bacterial Spot','Early Blight','Healthy','Late Blight','Leaf Mold','Mosaic Virus','Septoria Leaf Spot','Spider Mites','Target Spot','Yellow Leaf Curl Virus',],
    "peach":['bacterial spot',' Healthy'],
    "bell pepper":['bacterial-spot','Healthy'],
    "guava": ["Healthy Guava","Guava Phytopthora","Guava Red Rust,","Guava Scab","Guava Styler and Root"],
}
'''
DISEASE_INFO =  { file made
}'''
# Load the models and class labels outside of the route function
loaded_models = {}
model_class_labels = {}
for model_name, model_path in MODEL_PATHS.items():
    loaded_models[model_name] = load_model(model_path)
    model_class_labels[model_name] = CLASS_LABELS[model_name]
    loaded_model = load_model(model_path, custom_objects={'SparseCategoricalCrossentropy': SparseCategoricalCrossentropy()})
    from keras.models import load_model
from keras.losses import SparseCategoricalCrossentropy

# Load the saved model architecture
loaded_model = load_model(model_path)

# Get the configuration of the loaded model
model_config = loaded_model.get_config()

# Recreate the model architecture using the configuration
recreated_model = keras.Model.from_config(model_config) # type: ignore

# Compile the recreated model with the appropriate loss function
recreated_model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Load the weights of the saved model into the recreated model
recreated_model.load_weights(model_path)

# Now, the recreated_model should be ready for use




@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    selected_model = "potato"  # Default model
    uploaded_image_url = None  # Initialize uploaded image URL
    
    # load required model based on drop down
    if request.method == 'POST':
        if 'image' in request.files:
            image_file = request.files['image']
            selected_model = request.form['plant_type']

            if image_file.filename != '':
                if selected_model and selected_model != "select from plant":
                    img = Image.open(image_file)
                    img = img.convert('RGB')
                    img = img.resize((180, 180))
                    img = np.array(img)

                    model = loaded_models.get(selected_model)
                    class_labels = model_class_labels.get(selected_model)

                    if model and class_labels:
                        prediction = model.predict(np.expand_dims(img, axis=0))
                        predicted_class_index = np.argmax(prediction[0])
                        predicted_class = class_labels[predicted_class_index]
                        result = predicted_class

                        # Save the uploaded image
                        filename = secure_filename(image_file.filename)
                        upload_folder = os.path.join(app.root_path, 'static', 'uploads')
                        os.makedirs(upload_folder, exist_ok=True)  # Create the directory if it doesn't exist
                        uploaded_image_path = os.path.join(upload_folder, filename)
                        uploaded_image_url = url_for('static', filename=f'uploads/{filename}')
                        image_file.save(uploaded_image_path)
                        disease_info = DISEASE_INFO.get(predicted_class, {})
                        print("Disease Info:", disease_info) 
                        return render_template('result.html', result=result, disease_info=disease_info)

    return render_template('index1.html', result=result, selected_model=selected_model, uploaded_image_url=uploaded_image_url)

if __name__ == '__main__':
    app.run(debug=True)
