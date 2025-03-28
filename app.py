import pandas as pd
import numpy as np
import joblib
import gradio as gr

# Load the saved model and encoders
model = joblib.load('yoga_recommendation_model.pkl')
le_level = joblib.load('le_level.pkl')
le_target_areas = joblib.load('le_target_areas.pkl')
le_weight_goal = joblib.load('le_weight_goal.pkl')
mlb_mental = joblib.load('mlb_mental.pkl')
mlb_physical = joblib.load('mlb_physical.pkl')

# Load the dataset
df = pd.read_excel('data.xlsx')

# Load pose features (assuming they were saved or can be regenerated)
pose_features = pd.concat([
    df[['Level', 'Target Areas', 'Weight Goal Alignment']],
    pd.DataFrame(mlb_mental.transform(df['Targeted Mental Problems'].str.split(', ')), columns=mlb_mental.classes_),
    pd.DataFrame(mlb_physical.transform(df['Targeted Physical Problems'].str.split(', ')), columns=mlb_physical.classes_)
], axis=1)

# Recommendation function
def recommend_poses(height, weight, age, target_areas_arms, target_areas_legs, target_areas_core,
                    target_areas_back, target_areas_flexibility, target_areas_balance,
                    weight_goal, health_type, specific_problem='none'):
    if specific_problem == '':
        specific_problem = 'none'

    target_areas_list = []
    if target_areas_arms: target_areas_list.append('arms')
    if target_areas_legs: target_areas_list.append('legs')
    if target_areas_core: target_areas_list.append('core')
    if target_areas_back: target_areas_list.append('back')
    if target_areas_flexibility: target_areas_list.append('flexibility')
    if target_areas_balance: target_areas_list.append('balance')

    user_features = [
        height, weight, age,
        1 if health_type.lower() == 'mental' else 0,
        1 if specific_problem.lower() in (df['Targeted Mental Problems'] + df['Targeted Physical Problems']).str.lower() else 0,
        1 if 'arms' in target_areas_list else 0,
        1 if 'legs' in target_areas_list else 0,
        1 if 'core' in target_areas_list else 0,
        1 if 'back' in target_areas_list else 0,
        1 if 'flexibility' in target_areas_list else 0,
        1 if 'balance' in target_areas_list else 0,
        1 if weight_goal.lower() == 'lose weight' else 0
    ]

    X_pred = [np.concatenate([user_features, pose_features.iloc[idx].values]) for idx in range(len(df))]
    probs = model.predict_proba(np.array(X_pred))[:, 1]
    top_indices = np.argsort(probs)[::-1][:5]

    recommendations = df.iloc[top_indices][['AName', 'Description', 'Benefits', 'Contraindications', 'Level', 'Target Areas']]
    recommendations['Level'] = le_level.inverse_transform(recommendations['Level'])
    recommendations['Target Areas'] = le_target_areas.inverse_transform(recommendations['Target Areas'])
    return recommendations.to_html(index=False)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Personalized Yoga Recommendations")
    with gr.Row():
        height = gr.Slider(100, 250, value=170, label="Height (cm)", step=1)
        weight = gr.Slider(30, 150, value=70, label="Weight (kg)", step=1)
        age = gr.Slider(10, 100, value=30, label="Age", step=1)
    gr.Markdown("### Target Areas")
    target_areas_arms = gr.Checkbox(label="Arms")
    target_areas_legs = gr.Checkbox(label="Legs")
    target_areas_core = gr.Checkbox(label="Core")
    target_areas_back = gr.Checkbox(label="Back")
    target_areas_flexibility = gr.Checkbox(label="Flexibility")
    target_areas_balance = gr.Checkbox(label="Balance")
    gr.Markdown("### Weight Goal")
    weight_goal = gr.Radio(choices=["Lose Weight", "Gain Muscle"], value="Lose Weight", label="Weight Goal")
    gr.Markdown("### Health Focus")
    health_type = gr.Radio(choices=["Mental", "Physical"], value="Physical", label="Mental or Physical Health")
    gr.Markdown("### Specific Problem (Optional)")
    specific_problem = gr.Textbox(label="Specific Problem (e.g., stress relief, high bp)", placeholder="Enter a specific problem or leave blank")
    submit_button = gr.Button("Get Recommendations")
    output = gr.HTML(label="Recommended Yoga Poses")
    submit_button.click(
        fn=recommend_poses,
        inputs=[height, weight, age,
                target_areas_arms, target_areas_legs, target_areas_core,
                target_areas_back, target_areas_flexibility, target_areas_balance,
                weight_goal, health_type, specific_problem],
        outputs=output
    )

demo.launch()