# AI-Powered Cancer Support Suite

This repository contains a suite of AI-powered tools designed to assist with cancer-related information and risk assessment. It includes:

1.  **AI Oncologist Assistant Chatbot:** A conversational AI to answer medical queries about cancer using a Retrieval-Augmented Generation (RAG) approach.
2.  **Cancer Risk Prediction Model:** A machine learning model trained to predict the level of cancer risk based on patient data.

---

## Table of Contents

*   [AI Oncologist Assistant Chatbot](#ai-oncologist-assistant-chatbot)
    *   [Overview](#overview-chatbot)
    *   [Features](#features-chatbot)
    *   [Technology Stack](#technology-stack-chatbot)
    *   [Data Source](#data-source-chatbot)
    *   [Setup & Installation](#setup--installation-chatbot)
    *   [Usage](#usage-chatbot)
    *   [Notebook](#notebook-chatbot)
*   [Cancer Risk Prediction Model](#cancer-risk-prediction-model)
    *   [Overview](#overview-ml-model)
    *   [Features](#features-ml-model)
    *   [Technology Stack](#technology-stack-ml-model)
    *   [Data Source](#data-source-ml-model)
    *   [Model Performance](#model-performance)
    *   [Setup & Installation](#setup--installation-ml-model)
    *   [Usage](#usage-ml-model)
    *   [Notebook](#notebook-ml-model)
*   [API Endpoints](#api-endpoints)
*   [License](#license)
*   [Contributing](#contributing)

---

## AI Oncologist Assistant Chatbot

### Overview (Chatbot)

The Oncologist Assistant Doctor is an advanced AI chatbot designed to answer medical queries related to Cancer diseases. It leverages Groq's fast Large Language Model (LLM) inference capabilities (`llama-4-scout-17b-16e-instruct`) combined with a custom medical knowledge base built by scraping trusted medical websites. The system uses LangChain for orchestration and ChromaDB as a vector store for efficient information retrieval (RAG).

### Features (Chatbot)

*   Answers cancer-related questions in English or Arabic (detects input language).
*   Utilizes Retrieval-Augmented Generation (RAG) for contextually relevant answers based on scraped medical documents.
*   Maintains chat history for conversational context.
*   Provides sources or indicates when information is not available in its knowledge base.
*   Deployed as a FastAPI endpoint accessible via Ngrok.

### Technology Stack (Chatbot)

*   **LLM:** Groq (`meta-llama/llama-4-scout-17b-16e-instruct`)
*   **Framework:** LangChain (`langchain`, `langchain_groq`, `langchain_community`)
*   **Embeddings:** Hugging Face Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`)
*   **Vector Store:** ChromaDB (`chromadb`)
*   **Web Scraping:** `WebBaseLoader` (from Langchain)
*   **Language Detection:** `langdetect`
*   **API:** FastAPI (`fastapi`, `uvicorn`)
*   **Tunnelling:** Ngrok (`ngrok`, `pyngrok`)
*   **Environment:** Python, Jupyter Notebook (`.ipynb`)

### Data Source (Chatbot)

Medical information is scraped from the following trusted sources:

*   WebMD (`https://www.webmd.com/`)
*   Mayo Clinic (`https://www.mayoclinic.org/diseases-conditions/`)
*   MedlinePlus (`https://medlineplus.gov/`)
*   Healthline (`https://www.healthline.com/health`)
*   CDC (`https://www.cdc.gov/diseasesconditions/`)

### Setup & Installation (Chatbot)

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```
2.  **Install dependencies:**
    ```bash
    pip install -q gradio langchain_groq langchain_community chromadb langdetect ngrok pyngrok nest_asyncio fastapi uvicorn
    ```
3.  **Set up API Keys:**
    *   **Groq API Key:** Obtain a key from [GroqCloud Console](https://console.groq.com/keys) and replace `"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"` in the notebook.
    *   **Ngrok Authtoken:** Sign up at [Ngrok Dashboard](https://dashboard.ngrok.com/get-started/setup), get your authtoken, and configure it using the command in the notebook (replace `YOUR_NGROK_AUTH_TOKEN`):
        ```bash
        !ngrok config add-authtoken YOUR_NGROK_AUTH_TOKEN
        ```
    **Note:** Do not commit your actual API keys or tokens to version control. Use environment variables or a secure configuration method for production deployment.

### Usage (Chatbot)

1.  Open the `Oncologist_Assistant_Doctor.ipynb` notebook (e.g., in Google Colab or a local Jupyter environment).
2.  Ensure your API keys and Ngrok token are set correctly.
3.  Run all the cells in the notebook sequentially.
4.  The script will:
    *   Install necessary packages.
    *   Scrape web data, process it, and build the ChromaDB vector store (this may take some time on the first run).
    *   Initialize the LangChain components and the `OncologistChatbot` class.
    *   Start a FastAPI server.
    *   Launch an Ngrok tunnel and print the public URL (e.g., `https://<unique-id>.ngrok-free.app`).
5.  You can interact with the chatbot by sending POST requests to the `/ask` endpoint of the public Ngrok URL.

### Notebook (Chatbot)

*   [`Oncologist_Assistant_Doctor.ipynb`](./Oncologist_Assistant_Doctor.ipynb)

---

## Cancer Risk Prediction Model

### Overview (ML Model)

This project focuses on building and deploying a machine learning model to predict the risk level (Low, Medium, High) of lung cancer based on various patient attributes and lifestyle factors. The model is trained on a dataset of patient records and achieves high accuracy. The final selected model (RandomForestClassifier) is saved and deployed using FastAPI and Ngrok.

### Features (ML Model)

*   Data loading, exploration (EDA), and preprocessing (Label Encoding, Scaling).
*   Training and evaluation of multiple classification models (Logistic Regression, SVM, RandomForest).
*   Selection of the best-performing model (RandomForestClassifier).
*   Saving the trained pipeline (preprocessing + model) using Joblib.
*   Deployment of the model as a REST API using FastAPI.
*   Public accessibility via Ngrok tunneling.

### Technology Stack (ML Model)

*   **Data Handling:** Pandas (`pandas`), NumPy (`numpy`)
*   **Machine Learning:** Scikit-learn (`sklearn` - LogisticRegression, SVC, RandomForestClassifier, train_test_split, StandardScaler, LabelEncoder, Pipeline, metrics)
*   **Model Persistence:** Joblib (`joblib`)
*   **API:** FastAPI (`fastapi`, `uvicorn`), Pydantic (`pydantic` for input validation)
*   **Tunnelling:** Ngrok (`pyngrok`)
*   **Visualization (EDA):** Matplotlib (`matplotlib`), Seaborn (`seaborn`), Plotly (`plotly`)
*   **Environment:** Python, Jupyter Notebook (`.ipynb`)

### Data Source (ML Model)

*   The model is trained on the `cancer patient data sets.csv` file included (or referenced) in the repository. This dataset contains anonymized patient data with various features and a target 'Level' indicating cancer risk.
    *   **(Note:** Ensure this CSV file is correctly referenced or included in your repository structure).

### Model Performance

The final `RandomForestClassifier` model achieved high performance on the test set:

*   **Accuracy:** ~98.7% - 100% (depending on the exact run/split, the notebook shows excellent results)
*   Detailed performance metrics (precision, recall, F1-score) are available in the classification report generated within the notebook.

### Setup & Installation (ML Model)

1.  **Clone the repository:** (If not already done)
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```
2.  **Install dependencies:**
    ```bash
    pip install fastapi uvicorn joblib pandas pydantic pyngrok numpy plotly seaborn matplotlib scikit-learn
    ```
3.  **Ngrok Authtoken:** (If not already configured for the chatbot) Sign up at [Ngrok Dashboard](https://dashboard.ngrok.com/get-started/setup), get your authtoken, and configure it using the command in the notebook (replace `YOUR_NGROK_AUTH_TOKEN`):
    ```bash
    !ngrok config add-authtoken YOUR_NGROK_AUTH_TOKEN
    ```

### Usage (ML Model)

1.  Open the `High_Accuracies.ipynb` notebook.
2.  Ensure the `cancer patient data sets.csv` path is correct or the file is accessible.
3.  **Option 1: Training & Saving:** Run the cells up to the "Saving Model" section to perform EDA, train the models, evaluate them, and save the final pipeline (`canc_model.pkl`) and input columns (`model_inputs.pkl`).
4.  **Option 2: Running the API:**
    *   Ensure the `canc_model.pkl` file exists (either generated from Option 1 or provided).
    *   Run the cells in the "Test Model With API" section.
    *   This will:
        *   Load the saved model.
        *   Define the input data structure using Pydantic.
        *   Start a FastAPI server with the `/predict` endpoint.
        *   Launch an Ngrok tunnel and print the public URL.
5.  You can send POST requests with patient data (matching the `HealthInput` schema) to the `/predict` endpoint of the public Ngrok URL to get risk level predictions ("Positive" likely maps to High/Medium risk, "Negative" to Low - *confirm the mapping based on label encoding `0, 1, 2 -> Low, Medium, High`*).

### Notebook (ML Model)

*   [`High_Accuracies.ipynb`](./High_Accuracies.ipynb)

---

## API Endpoints

Once the notebooks are run and Ngrok tunnels are active:

*   **Chatbot:** `POST https://<chatbot-ngrok-url>/ask`
    *   **Request Body:** `{"question": "Your cancer question here"}`
    *   **Response Body:** `{"answer": "Chatbot's response", "language": "detected_language_code"}`
*   **Prediction Model:** `POST https://<ml-model-ngrok-url>/predict`
    *   **Request Body:** A JSON object matching the `HealthInput` Pydantic model defined in `High_Accuracies.ipynb`.
    *   **Response Body:** `{"prediction": "Positive/Negative"}` (Mapping to Low/Medium/High risk needs clarification based on implementation)


---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

---


 
 
