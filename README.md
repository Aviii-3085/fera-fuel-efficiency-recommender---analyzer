 Running the Project with Docker
Build the Docker image:
docker build -t fera .
Run the container:
docker run -p 8000:8000 fera

 Project Structure

fera-fuel-efficiency-recommender-analyzer/
│
├── Dockerfile
├── server.py
├── eff_model.joblib
├── health_model.joblib
├── run.bat
├── run_fast.bat
├── run_local.bat
└── static/



Introduction
FERA (Fuel Efficiency Recommender & Analyzer) is an intelligent system designed to
evaluate, predict, and improve fuel efficiency in internal combustion engines. The system
blends rule based logic with machine learning to provide real time insights and
recommendations for users.

Idea & Innovation
The core idea behind FERA is to provide an explainable, interactive, and user friendly
tool that helps users understand how various engine parameters affect fuel efficiency. The
innovation lies in combining ML based predictions, real time analysis, visualization, and
dynamic recommendations, presented within an accessible web interface. 

Research,Design & Ideology
The research behind FERA includes studying engine behavior, understanding key
parameters such as RPM, AFR, throttle, and load, and mapping their interactions with
efficiency. The system is designed using an explainable AI approach, ensuring users not
only get predictions but also understand the reasoning behind them. Economic Efficiency
FERA helps users maximize fuel savings by optimizing driving behavior and engine
characteristics. This contributes to reduced running costs and better long term engine
maintenance. 

Social Relevancy
Fuel efficiency is a significant concern in today’s environment. By promoting conscious
and efficient driving patterns, FERA supports energy conservation and environmental
sustainability. 

Impact on Society
The system encourages eco friendly driving, reduces carbon footprint, and spreads
awareness about efficiency optimization. It also empowers drivers with better control and
understanding of their vehicle performance. 

Technical Aspects
FERA uses:
• Frontend: HTML, CSS, JavaScript, Chart.js
• Backend: Python, FastAPI
• ML Stack: Scikit learn, joblib
• Tools: Docker, SQLite, Uvicorn
• Deployment: Containerized backend with static fronten
