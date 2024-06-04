from shiny import App, ui, render, reactive
import pandas as pd
import joblib


# Load the sample dataset to get unique values for dropdowns
data = pd.read_csv(r'C:\Users\tngku\OneDrive\Desktop\project1\src\pyfile\afiqhensem.csv')

# Load the trained model and columns used during training
model = joblib.load(r'C:\Users\tngku\OneDrive\Desktop\project1\src\pyfile\mood_swings_model.pkl')
columns = joblib.load(r'C:\Users\tngku\OneDrive\Desktop\project1\src\pyfile\columns.pkl')

# Get unique values for each feature
gender_values = ['Female', 'Male']
country_values = ['United States', 'United Kingdom', 'Australia', 'Canada', 'Germany', 'France', 'India', 'Others']
occupation_values = ['Corporate', 'Student', 'Housewife', 'Others']
self_employed_values = ['No', 'Yes']
family_history_values = ['No', 'Yes']
treatment_values = ['No', 'Yes']
days_indoors_values = ['1-14 days', '15-30 days', '31-60 days', 'Go out Every day', 'More than 2 months']
growing_stress_values = ['No', 'Yes', 'Maybe']
changes_habits_values = ['No', 'Yes', 'Maybe']
mental_health_history_values = ['No', 'Yes', 'Maybe']
coping_struggles_values = ['No', 'Yes']
work_interest_values = ['No', 'Yes', 'Maybe']
social_weakness_values = ['No', 'Yes', 'Maybe']

# Define custom CSS for better styling
custom_css = """
body {
    font-family: Arial, sans-serif;
    background-color: #e0f7fa;
}
.container {
    max-width: 900px;
    margin: auto;
    padding: 20px;
    background-color: white;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    border-radius: 10px;
}
.header {
    text-align: center;
    margin-bottom: 20px;
}
.header h1 {
    margin: 0;
    font-size: 2.5em;
    color: #00796b;
    font-weight: bold;
}
.header p {
    font-size: 1.2em;
    color: #004d40;
}
.section {
    margin-bottom: 20px;
}
.section h2 {
    font-size: 1.5em;
    margin-bottom: 10px;
    color: #004d40;
    font-weight: bold;
}
.section .input-group {
    margin-bottom: 10px;
}
.input-group label {
    font-weight: bold;
    color: #004d40;
}
.input-group select {
    width: 100%;
    padding: 5px;
    border: 1px solid #00796b;
    border-radius: 5px;
}
.predict-button {
    display: block;
    width: 100%;
    padding: 10px;
    font-size: 1.2em;
    background-color: #00796b;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-weight: bold;
}
.predict-button:hover {
    background-color: #004d40;
}
.result {
    margin-top: 20px;
    padding: 20px;
    background-color: #b2dfdb;
    border: 1px solid #004d40;
    border-radius: 10px;
}
.result h3 {
    margin-top: 0;
    font-size: 1.5em;
    color: #00796b;
}
"""

# Define the UI
app_ui = ui.page_fluid(
    ui.tags.style(custom_css),
    ui.div(
        {"class": "container"},
        ui.div(
            {"class": "header"},
            ui.h1("Mood Swings Prediction"),
            ui.p("Predict the likelihood of experiencing mood swings based on your inputs.")
        ),
        ui.div(
            {"class": "section"},
            ui.h2("Personal Information"),
            ui.div({"class": "input-group"}, ui.input_select("gender", "Select Gender:", gender_values)),
            ui.div({"class": "input-group"}, ui.input_select("country", "Select Country:", country_values)),
            ui.div({"class": "input-group"}, ui.input_select("occupation", "Select Occupation:", occupation_values)),
            ui.div({"class": "input-group"}, ui.input_select("self_employed", "Select Self-Employed Status:", self_employed_values)),
        ),
        ui.div(
            {"class": "section"},
            ui.h2("Mental Health History"),
            ui.div({"class": "input-group"}, ui.input_select("family_history", "Select Family History:", family_history_values)),
            ui.div({"class": "input-group"}, ui.input_select("treatment", "Select Treatment Status:", treatment_values)),
            ui.div({"class": "input-group"}, ui.input_select("mental_health_history", "Select Mental Health History:", mental_health_history_values)),
        ),
        ui.div(
            {"class": "section"},
            ui.h2("Lifestyle and Habits"),
            ui.div({"class": "input-group"}, ui.input_select("days_indoors", "Select Days Indoors:", days_indoors_values)),
            ui.div({"class": "input-group"}, ui.input_select("growing_stress", "Select Growing Stress Level:", growing_stress_values)),
            ui.div({"class": "input-group"}, ui.input_select("changes_habits", "Select Changes in Habits:", changes_habits_values)),
        ),
        ui.div(
            {"class": "section"},
            ui.h2("Coping and Social"),
            ui.div({"class": "input-group"}, ui.input_select("coping_struggles", "Select Coping Struggles:", coping_struggles_values)),
            ui.div({"class": "input-group"}, ui.input_select("work_interest", "Select Work Interest Level:", work_interest_values)),
            ui.div({"class": "input-group"}, ui.input_select("social_weakness", "Select Social Weakness:", social_weakness_values)),
        ),
        ui.div(
            {"class": "section"},
            ui.input_action_button("predict", "Predict Mood Swings", class_="predict-button")
        ),
        ui.div({"class": "result"},
            ui.output_text("prediction_output")
        )
    )
)

# Define the server logic
def server(input, output, session):
    # Reactive trigger for prediction
    trigger = reactive.Value(0)

    # Function to convert user inputs to one-hot encoded format
    def transform_input(input):
        user_input = {
            'Gender': [input.gender()],
            'Country': [input.country()],
            'Occupation': [input.occupation()],
            'self_employed': [input.self_employed()],
            'family_history': [input.family_history()],
            'treatment': [input.treatment()],
            'Days_Indoors': [input.days_indoors()],
            'Growing_Stress': [input.growing_stress()],
            'Changes_Habits': [input.changes_habits()],
            'Mental_Health_History': [input.mental_health_history()],
            'Coping_Struggles': [input.coping_struggles()],
            'Work_Interest': [input.work_interest()],
            'Social_Weakness': [input.social_weakness()]
        }
        user_input_df = pd.DataFrame(user_input)
        user_input_df = pd.get_dummies(user_input_df)
        
        # Ensure all columns are present
        for col in columns:
            if col not in user_input_df.columns:
                user_input_df[col] = 0
        
        # Reorder columns to match the training data
        user_input_df = user_input_df[columns]
        
        return user_input_df

    @reactive.Effect
    @reactive.event(input.predict)
    def make_prediction():
        trigger.set(trigger() + 1)

    @output
    @render.text
    def prediction_output():
        if trigger():
            # Transform user input
            user_input_df = transform_input(input)
            
            # Debugging: print transformed user input with full columns
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print("Transformed user input:\n", user_input_df)
            
            # Perform prediction
            prediction = model.predict(user_input_df)
            
            # Debugging: print model prediction
            print("Model prediction:", prediction)
            
            # Interpret the prediction based on the model output
            if prediction[0] == 0:  # Assuming 0 corresponds to "Medium" or "Bearable Level"
                return "Mood Swings Prediction: Bearable Level"
            else:
                return "Mood Swings Prediction: High Level"
        else:
            return "Click 'Predict Mood Swings' to see the result."

# Create the Shiny app
app = App(app_ui, server)

# Run the app
app.run()
