# Mental Health Analysis Chatbot

A deep learning-based chatbot designed to analyze mental health conditions based on user inputs. The model detects emotions such as anxiety, depression, stress, and more using NLP techniques.

## Features
- **Emotion Detection:** Identifies various mental health states from text input.
- **AI-Powered Chatbot:** Provides supportive and engaging responses based on detected emotions.
- **RoBERTa-Based Model:** Fine-tuned on mental health-related datasets for accurate predictions.
- **Streamlit UI:** Simple web interface for easy interaction.
- **Google Drive Integration:** Loads models directly from Google Drive without storing them on GitHub.

## Built With
- Python  
- Hugging Face Transformers  
- PyTorch  
- Google Drive API  
- gdown  
- Streamlit  
- GitHub  
- Streamlit Cloud  
- NLTK  
- SpaCy  

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/LUFFY-KingofCoder/mental_health_chatbot.git
   cd mental_health_chatbot
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Run the chatbot:
   ```sh
   streamlit run chatbot.py
   ```

## Model Setup
Since GitHub has size limits, the model is stored in Google Drive. To use it:

1. Ensure `gdown` is installed:
   ```sh
   pip install gdown
   ```
2. Update `chatbot.py` with the correct Google Drive folder link.

## Usage
- Open the chatbot in a browser using the Streamlit interface.
- Enter text, and the chatbot will analyze the emotional state.

## Challenges Faced
- Managing large model files with GitHub LFS limitations.
- Fine-tuning RoBERTa for multi-class emotion classification.
- Ensuring real-time response efficiency on Streamlit Cloud.

## Future Plans
- Enhance chatbot response personalization.
- Improve model accuracy with more mental health datasets.
- Deploy on a dedicated server for better performance.


## Author
**Shashank Ghosh**  
GitHub: [LUFFY-KingofCoder](https://github.com/LUFFY-KingofCoder)  
LinkedIn: [Shashank Ghosh](https://www.linkedin.com/in/shashank-ghosh-915b14169/)

