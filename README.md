Open the project directory in VS Code and check if it has the following
  i) A dataset named “heart.csv” 
  ii) A python file named “ml_model.py” (this code fetches data from the dataset, trains the ML model and dumps the data in 2 files named as “model.pkl” and “scaler.pkl”)
  iii) A python file named “web_app.py” (this code fetches data from the 2 .pkl files and connects with Flask server
  iv) A folder named “templates” containing a file named “index.html” (the html code creates the web page)
  v) A folder named “static” containing:
      a) a file named “style.css” (this code adds details to the web page)
      b) An image file named “heart_bg.jpeg”
Create a virtual environment with Python 3.14.2 as the interpreter and activate it.
Install required libraries in the virtual environment by running this command in the terminal of VS Code: “pip install numpy pandas matplotlib scikit-learn xgboost flask joblib seaborn”
Run the “ml_model.py” and “web_app.py” files one by one in the virtual environment using commands: “python –m ml_model” and “python –m web_app” respectively.
Open the web page by going to the displayed URL.
Enter all details in the form of the web page and click on “GET PREDICTION” button to view the prediction.

