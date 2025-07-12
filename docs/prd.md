### **Connesis Machine Learning Web**

Build a complete, multi-page machine learning web application using Python and Streamlit. The application should guide a user through the entire workflow: uploading data, preprocessing it, training a model with hyperparameter tuning, evaluating the results, and finally, using the trained model for live predictions.

The application should be structured as a single Python script and should use `st.session_state` to manage the state between steps.

**Technology Stack:**
* **Web Framework:** Streamlit
* **Data Handling:** Pandas
* **Machine Learning & Preprocessing:** Scikit-learn
* **Visualization:** Plotly
* **Model Persistence:** Joblib
* **Development Environment:** Python 3.11 & uv

---

### **Application Structure & Workflow**

The app will have two main "pages" controlled by a sidebar radio button: **"Model Builder"** and **"Prediction Interface"**.

#### **Page 1: Model Builder**

This page is for the entire workflow from data upload to model evaluation. The controls should be in the sidebar (`st.sidebar`).

**1.  Sidebar Controls:**

* **File Upload:** Use `st.file_uploader` to accept CSV or Excel files. If no file is uploaded, the app should show a welcome message. Once a file is uploaded, store the DataFrame in `st.session_state.df` and display all subsequent controls.
* **Preprocessing Settings:**
    * A `st.selectbox` for the **Target Variable (Y)**.
    * A `st.multiselect` for the **Feature Variables (X)**.
    * A `st.selectbox` for **Missing Value Treatment** ("Delete Rows", "Impute").
    * A `st.selectbox` for **Categorical Encoding** ("One-Hot Encoding", "Label Encoding").
    * A `st.selectbox` for **Numerical Scaling** ("StandardScaler", "MinMaxScaler").
    * An `st.button` labeled "**Apply Preprocessing**". When clicked, run all preprocessing steps and store the final preprocessor object and the processed X/y dataframes in `st.session_state`.
* **Training Settings:**
    * A `st.slider` for **Test Set Split Ratio** (0.1 to 0.5).
    * A `st.number_input` for **Random Seed**.
* **Model Selection & Tuning:**
    * A `st.selectbox` to **Select Classification Model** ("Decision Tree", "KNN", "Random Forest", "SVM", "Logistic Regression", "Gradient Boosting").
    * A **Dynamic Hyperparameter Tuning** section. Use `if/elif` statements to show different widgets based on the model selected. For example:
        * **If "Decision Tree"**: Show sliders for "Maximum Depth" and "Minimum Samples for Split".
        * **If "KNN"**: Show a slider for "Number of Neighbors (k)" and a dropdown for "Weights".
* **Advanced Training Options:**
    * A `st.checkbox` for "**Enable Class Weight 'balanced'**".
    * A `st.checkbox` for "**Enable Hyperparameter Tuning (Grid Search)**".
* **Final Launch Button:**
    * A `st.button` labeled "**Start Training** 🚀".

**2.  Main Content Area (Model Builder):**

* **Before Training:** Display a preview of the data (`st.dataframe`).
* **After Training:** The main area becomes a **Results Dashboard** using `st.tabs`:
    * **Tab 1: Evaluation Metrics:** Display the overall **Accuracy** using `st.metric`. Show the **Confusion Matrix** using `st.plotly_chart` and the **Classification Report** using `st.dataframe`.
    * **Tab 2: Feature Importance:** Show a Plotly bar chart of feature importances (if applicable to the model).
    * **Tab 3: ROC Curve:** Show a Plotly line chart of the ROC curve and display the AUC score.
    * **Tab 4: Feature Correlation Heatmap:** Display a Plotly heatmap showing the correlation matrix between all numerical features used in training. This helps identify feature relationships and potential multicollinearity issues.
    * **Tab 5: Model-Specific View:** For a Decision Tree, show the tree structure using `st.text_area`.
* **Deploy Button:** After a model is successfully trained, display a `st.button` labeled "**Deploy Model**". When clicked, use `joblib.dump` to save the trained model object and the preprocessor object to disk.

---

#### **Page 2: Prediction Interface**

This page is for using the deployed model. It should only be accessible if a model has been saved.

**1.  Prediction Method:**

* Use `st.radio` to select between **"Manual Input"** and **"Batch Prediction (File Upload)"**.

**2.  Manual Input:**

* Dynamically create a form with one input widget for each feature the model was trained on. Use `st.number_input` for numerical features and `st.selectbox` (populated with categories from the training data) for categorical features.
* A `st.button` labeled "**Get Prediction**". When clicked:
    1.  Load the saved preprocessor and model using `joblib.load`.
    2.  Transform the user's input using the loaded preprocessor.
    3.  Make a prediction using the loaded model.
    4.  Display the result using `st.success`.

**3.  Batch Prediction:**

* Use `st.file_uploader` to accept a new file.
* A `st.button` to run predictions on the entire file and offer a `st.download_button` for the results as a CSV.

Please provide the complete, single-script Python code for this Streamlit application, including all necessary functions and UI logic.