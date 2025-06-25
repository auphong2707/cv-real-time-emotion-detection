# cv-real-time-emotion-detection

## Installation

1. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Check the `deploy_models` folder:
    - If there are no models in the folder, download them from [Hugging Face](https://huggingface.co/auphong2707/cv-real-time-emotion-detection/tree/main).
    - The models are located in each folder with the format `model_name_best.pth`. Download and save them in the `deploy_models` folder.

## Running the Application

1. Start the application:
    ```bash
    python emotion_recognition_web.py
    ```

2. Open your browser and navigate to `http://localhost:8000` to use the app.

## Training the Model

If you want to train the models:

1. Modify the parameters in `constants.py` to suit your training requirements.

2. Set up the necessary environment variables:
    - `HUGGINGFACE_TOKEN`
    - `WANDB_API_KEY`
    - `WANDB_PROJECT`

3. Run the corresponding Python script located in the `scripts` folder. For example:
    ```bash
    python scripts/train_emotion_cnn.py
    ```