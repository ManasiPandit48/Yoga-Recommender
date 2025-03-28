# Yoga Recommender

**Yoga Recommender** is a machine learning-based web app that suggests personalized yoga poses based on user inputs like height, weight, age, target areas, weight goals, and health focus. Built with Python and a Random Forest Classifier, it’s deployed on Hugging Face Spaces using Gradio.

## Features

- **Personalized Yoga Poses**: Tailored recommendations for each user.
- **Simple Interface**: Sliders, checkboxes, and text inputs for easy use.
- **Online Access**: Hosted on Hugging Face Spaces—no local setup needed.

## Live Demo

Try the app on hugginface Spaces
- [Yoga Recommender on Hugging Face Spaces](https://huggingface.co/spaces/ManasiPandit/Yoga_Recommender)  

## How It Works

1. Users enter their details (e.g., height, age, target areas).  
2. A pre-trained model predicts the best yoga poses.  
3. The app displays the top 5 poses with descriptions and benefits.

## Setup

### Prerequisites

- Python 3.8+  
- Libraries: `pandas`, `numpy`, `scikit-learn`, `gradio`, `joblib`, `openpyxl`

### Local Installation

1. Clone the repo:  
   ```bash
   git clone https://github.com/yourusername/yoga_recommender.git
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:  
   ```bash
   python app.py
   ```

## Deployment

Hosted on Hugging Face Spaces. To deploy your own:  
1. Create a Space on Hugging Face.  
2. Upload `app.py`, `requirements.txt`, dataset, and model files.  
3. Select Gradio as the SDK.

## License

MIT License

---
