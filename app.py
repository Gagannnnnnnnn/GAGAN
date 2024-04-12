from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os





app = Flask(__name__, template_folder='C:/Users/Gagan/Downloads/gagan/FLASK/templates')


# knn #

# Function to process the uploaded CSV file
def process_csv(file):
    # Load the dataset
    data = pd.read_csv(file)

    # Infer feature and target columns
    target_column = data.columns[-1]  # Assume the last column is the target column
    feature_columns = data.columns[:-1].tolist()  # All columns except the last one are features

    X = data[feature_columns]
    y = data[target_column]

    # Identify categorical and numerical columns
    categorical_columns = [col for col in X.columns if X[col].dtype == 'object']
    numeric_columns = [col for col in X.columns if col not in categorical_columns]

    # One-hot encode categorical columns
    if categorical_columns:
        encoder = OneHotEncoder(drop='first', sparse=False)
        X_encoded = encoder.fit_transform(X[categorical_columns])

        # PCA for dimensionality reduction
        pca = PCA(n_components=min(100, X_encoded.shape[1]))  # Limit components to avoid high dimensionality
        X_pca = pca.fit_transform(X_encoded)
    else:
        # No categorical columns, proceed with numerical columns only
        X_pca = X

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=2, random_state=42)  # Adjust the number of clusters based on your problem
    y_pred = kmeans.fit_predict(X_test)

    # Convert cluster labels to binary (0 or 1)
    y_pred_binary = (y_pred == 1).astype(int)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred_binary)

    # Compute classification report
    classification_rep = classification_report(y_test, y_pred_binary)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    
    
    
    

    return accuracy, classification_rep, cm, categorical_columns












# RF #





# Function to process the uploaded CSV file
def process_csv1(file):
    # Load the dataset
    data = pd.read_csv(file)

    # Infer feature and target columns
    target_column = data.columns[-1]  # Assume the last column is the target column
    feature_columns = data.columns[:-1].tolist()  # All columns except the last one are features

    X = data[feature_columns]
    y = data[target_column]

    # Identify categorical and numerical columns
    categorical_columns = [col for col in X.columns if X[col].dtype == 'object']
    numeric_columns = [col for col in X.columns if col not in categorical_columns]

    # One-hot encode categorical columns
    if categorical_columns:
        encoder = OneHotEncoder(drop='first', sparse=False)
        X_encoded = encoder.fit_transform(X[categorical_columns])

        # PCA for dimensionality reduction
        pca = PCA(n_components=min(100, X_encoded.shape[1]))  # Limit components to avoid high dimensionality
        X_pca = pca.fit_transform(X_encoded)
    else:
        # No categorical columns, proceed with numerical columns only
        X_pca = X

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Apply RF #
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Convert cluster labels to binary (0 or 1)
    y_pred_binary = (y_pred == 1).astype(int)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred_binary)

    # Compute classification report
    classification_rep = classification_report(y_test, y_pred_binary)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    
    
    
    

    return accuracy, classification_rep, cm, categorical_columns









# SVC #





# Function to process the uploaded CSV file
def process_csvsvc(file):
    # Load the dataset
    data = pd.read_csv(file)

    # Infer feature and target columns
    target_column = data.columns[-1]  # Assume the last column is the target column
    feature_columns = data.columns[:-1].tolist()  # All columns except the last one are features

    X = data[feature_columns]
    y = data[target_column]

    # Identify categorical and numerical columns
    categorical_columns = [col for col in X.columns if X[col].dtype == 'object']
    numeric_columns = [col for col in X.columns if col not in categorical_columns]

    # One-hot encode categorical columns
    if categorical_columns:
        encoder = OneHotEncoder(drop='first', sparse=False)
        X_encoded = encoder.fit_transform(X[categorical_columns])

        # PCA for dimensionality reduction
        pca = PCA(n_components=min(100, X_encoded.shape[1]))  # Limit components to avoid high dimensionality
        X_pca = pca.fit_transform(X_encoded)
    else:
        # No categorical columns, proceed with numerical columns only
        X_pca = X

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Apply SVC #
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Convert cluster labels to binary (0 or 1)
    y_pred_binary = (y_pred == 1).astype(int)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred_binary)

    # Compute classification report
    classification_rep = classification_report(y_test, y_pred_binary)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    
    
    
    

    return accuracy, classification_rep, cm, categorical_columns








@app.route('/')
def index():
    return render_template("index.html")


@app.route('/USER.html')  # Route for sec.html
def user():
    return render_template("USER.html")

@app.route('/ADMIN.html')  # Route for sec.html
def admin():
    return render_template("ADMIN.html")


@app.route('/HOME.html')  # Route for sec.html
def home():
    return render_template("HOME.html")


@app.route('/RESOURCES.html')  # Route for sec.html
def resources():
    return render_template("RESOURCES.html")


@app.route('/RESULTS.html')  # Route for sec.html
def results():
    return render_template("RESULTS.html")


@app.route('/ABOUT_US.html')  # Route for sec.html
def about():
    return render_template("ABOUT.html")


@app.route('/CONTACT_US.html')  # Route for sec.html
def contact():
    return render_template("CONTACT_US.html")


@app.route('/DOCTORS_PORTAL.html')  # Route for sec.html
def doctors():
    return render_template("DOCTORS_PORTAL.html")


@app.route('/FATURES.html')  # Route for sec.html
def features():
    return render_template("FEATURES.html")


@app.route('/TOOLS.html')  # Route for sec.html
def tools():
    return render_template("TOOLS.html")


@app.route('/knnn.html')  # Route for sec.html
def knnn():
    return render_template("knnn.html")


@app.route('/random.html')  # Route for sec.html
def random():
    return render_template("random.html")



@app.route('/svc.html')  # Route for sec.html
def svc():
    return render_template("svc.html")


@app.route('/knnres.html', methods=['POST'])
def knnres():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Process the uploaded CSV file
        accuracy, classification_rep, cm, categorical_columns = process_csv(file_path)
        
        
        
        
        # Debugging: Print the processed data
        print("Accuracy:", accuracy)
        print("Classification Report:", classification_rep)
        print("Confusion Matrix:", cm)
        print("Categorical Columns:", categorical_columns)
        
        
        
        

        # Render knnres.html with results
        return render_template("knnres.html", accuracy=accuracy, classification_rep=classification_rep, cm=cm,
                               categorical_columns=categorical_columns)



    
    
    
    
    
@app.route('/rfres.html', methods=['POST'])
def rfres():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Process the uploaded CSV file
        accuracy, classification_rep, cm, categorical_columns = process_csv1(file_path)
        
        
        
        
        # Debugging: Print the processed data
        print("Accuracy:", accuracy)
        print("Classification Report:", classification_rep)
        print("Confusion Matrix:", cm)
        print("Categorical Columns:", categorical_columns)
        
        
        
        

        # Render knnres.html with results
        return render_template("rfres.html", accuracy=accuracy, classification_rep=classification_rep, cm=cm,
                               categorical_columns=categorical_columns)    
    

    
    
    
    
@app.route('/svcres.html', methods=['POST'])
def svcres():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Process the uploaded CSV file
        accuracy, classification_rep, cm, categorical_columns = process_csvsvc(file_path)
        
        
        
        
        # Debugging: Print the processed data
        print("Accuracy:", accuracy)
        print("Classification Report:", classification_rep)
        print("Confusion Matrix:", cm)
        print("Categorical Columns:", categorical_columns)
        
        
        
        

        # Render knnres.html with results
        return render_template("svcres.html", accuracy=accuracy, classification_rep=classification_rep, cm=cm,
                               categorical_columns=categorical_columns)    
    
    
    


    
    
    
    



if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'C:/Users/Gagan/Downloads/gagan/FLASK/uploads'
    app.run(debug=False,host='0.0.0.0')

    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


