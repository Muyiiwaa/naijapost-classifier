# Nigerian Post Classifier.

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20Streamlit-green.svg)](https://fastapi.tiangolo.com/)
[![Model](https://img.shields.io/badge/Model-DistilBERT-yellow.svg)](https://huggingface.co/distilbert-base-uncased)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)

This repository contains a machine learning project for classifying Nigerian posts. It uses a fine-tuned transformer model (DistilBERT) to categorize posts from nairaland (a popular nigerian microblogging website) into predefined labels. The project includes scripts for training, preprocessing tools, A production ready ZenML + MLFlow pipeline, a REST API built with FastAPI, and an interactive demo using Streamlit.

---

## Live Demo

Here is a quick look at the Streamlit application in action. The app provides a simple interface to input text and receive a classification prediction from the model served by the FastAPI backend.

**Application Screenshot:**
*You can replace the placeholder below with a screenshot of your app.*
![Streamlit App Demo](path/to/your/streamlit_demo.png)

**Video Walkthrough:**
*To embed a video, you can create a GIF and use standard image markdown. For a full video, you might link to a YouTube or Loom recording.*

`[![Project Demo Video](path/to/your/video_thumbnail.jpg)](https://www.youtube.com/watch?v=your_video_id)`

---

## Features

* **High-Performance Model**: Leverages a fine-tuned **DistilBERT** model for accurate and efficient text classification.
* **Nigerian Language Support**: Specifically trained on the **Naijaweb dataset** by Saheed Niyi, covering multiple Nigerian languages and Pidgin.
* **REST API**: A robust backend API built with **FastAPI** to serve the model and handle inference requests.
* **Interactive Demo**: A user-friendly web interface created with **Streamlit** to demonstrate the model's capabilities.
* **Modern Tooling**: Uses **UV** for fast and reliable Python package management.

---

## Tech Stack

* **Model**: Hugging Face Transformers (DistilBERT)
* **Backend**: FastAPI
* **Frontend Demo**: Streamlit
* **Data Handling and Fine-Tuning**: Pandas, Scikit-learn, PyTorch
* **Package Management**: UV
* **PipeLine**: ZenML
* **Experiment Tracking**: MLFlow

---

## ⚙️ Setup and Installation

To get this project running locally, follow these steps.

**1. Clone the Repository:**
```bash
git clone [https://github.com/Muyiiwaa/naijapost-classifier.git](https://github.com/Muyiiwaa/naijapost-classifier.git)
cd naijapost-classifier
````

**2. Environment Variables (.env_example):**
This project relies on environment variables for configuration and sensitive information (like API keys). An example file, `.env_example`, is provided to guide you.

  * Copy the `.env_example` file and rename it to `.env`:
    ```bash
    cp .env_example .env
    ```
  * Open the newly created `.env` file and update the placeholder values (especially `HUGGING_FACE_API_KEY`) with your actual credentials.
  * **Important:** make sure `.env` is added to your `.gitignore` file to prevent sensitive information from being committed to version control.

**3. Create and Activate a Virtual Environment:**
This project uses **UV** for package management. First, run `uv sync`, which creates a virtual environment with all the libraries available.

```bash
uv sync
```

Then, activate it.

  * On macOS and Linux:
    ```bash
    source .venv/bin/activate
    ```
  * On Windows:
    ```bash
    .venv\Scripts\activate
    ```

**4. Install Dependencies:**
Install all the required packages from `pyproject.toml`.

```bash
uv pip install -e .
```

-----

## How to Run

After setting up the project, you can run the individual components.

**1. Run the API Server:**
The FastAPI server exposes the model's prediction endpoint. Start the server using Uvicorn.

```bash
uvicorn src.main:app --reload --port 8005
```

You can now access the API documentation at `http://127.0.0.1:8005/docs` and `http://127.0.0.1:8005/redoc`.

**2. Launch the Streamlit Demo:**
Run the Streamlit app in a separate terminal. The app will connect to the FastAPI server running in the background.

```bash
streamlit run streamlit_app.py
```

Open your browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

**3. Model Training:**
The notebook `train.ipynb` contains the complete workflow for data preprocessing, model fine-tuning, and evaluation. You can explore or re-run it using a Jupyter environment.

-----

## 📂 Project Structure

Here is an overview of the repository's structure:

```
.
├── src/                # Source code for the application logic
├── .python-version     # Specifies the Python version for the project
├── config.py           # Configuration variables and settings
├── main.py             # FastAPI application entry point
├── pyproject.toml      # Project metadata and dependencies for UV
├── streamlit_app.py    # The Streamlit demo application script
├── train.ipynb         # Jupyter notebook for model training and experimentation
├── utils.py            # Utility functions for preprocessing, etc.
└── uv.lock             # Pinned dependencies managed by UV
```

-----

## Acknowledgments

  * A big S/O to **Saheed Niyi** for creating and sharing the [Naijaweb dataset](https://www.google.com/search?q=https://github.com/SaheedNIYI/NaijaWeb-Senti), which made this project possible.
  * The teams behind [Hugging Face Transformers](https://huggingface.co/), [FastAPI](https://fastapi.tiangolo.com/), and [Streamlit](https://streamlit.io/) for their amazing tools.

