from fastapi import FastAPI
import joblib

app = FastAPI()

# Load the trained model and vectorizer
model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.pkl')

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/ask")
def ask(text: str):
    # Transform input text using the loaded vectorizer
    text_vectorized = vectorizer.transform([text])
    
    # Convert sparse matrix to dense array
    text_vectorized_dense = text_vectorized.toarray()

    # Make prediction
    prediction = model.predict(text_vectorized_dense)[0]
    
    return {"prediction": prediction}





