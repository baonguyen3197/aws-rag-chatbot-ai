=====================================
## Setting Up the Project Environment
=====================================
```bash
pyhon -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

=====================================
## Initializing Reflex Project
=====================================
```bash
reflex init
```

=====================================
## Run Reflex App
=====================================
```bash
reflex run
```

=====================================
## Build Reflex App Docker Image
=====================================
```bash
docker build -t nhqb3197/aws_rag_chatbot_ai:latest .
docker push nhqb3197/aws_rag_chatbot_ai:latest