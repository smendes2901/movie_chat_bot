# Llama2 Movie Chatbot

## Overview
This repository contains the implementation of a fine-tuned Llama2 chatbot using QLoRA, tailored to provide detailed information and recommendations about movies. The model is fine-tuned on the IMDB dataset, enabling it to generate informed and contextually relevant responses.

## Features
- **Movie Information**: The chatbot can fetch and provide detailed information about a wide range of movies, including plot summaries, cast details, directorial information, and more.
- **Movie Recommendations**: Based on user preferences and querying style, the chatbot offers personalized movie recommendations.
- **Dynamic Context Handling**: The model uses a predefined context during inference to generate accurate and relevant responses. If the information required to answer a query is not available, the chatbot smartly responds with "I don't know," prompting further clarification or a different query from the user.

## Model Details
- **Base Model**: Llama2
- **Training Data**: IMDB dataset
- **Technique**: Fine-tuning with QLoRA for query-focused responses

## Installation
To set up the chatbot on your local machine, follow these steps:

1. **Clone the Repository**:
```
git clone https://github.com/yourgithubusername/llama2-movie-chatbot.git
```
2. **Install Dependencies**:
```
pip install -r requirements.txt
```
3. **Run the Application**:
```
python app.py
```

## Usage
To interact with the chatbot:

1. Start the application using the above instructions.
2. Access the chat interface through your local web server (usually `http://localhost:5000`).
3. Type your movie-related questions or ask for recommendations in the chat interface.

## Contributing
Contributions to enhance the functionality or efficiency of the Llama2 movie chatbot are welcome. Please follow these steps:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

Feel free to adjust the content to better fit your project's specifics or personal preferences!
