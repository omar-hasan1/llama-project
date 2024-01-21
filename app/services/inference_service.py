from app.models.llama_model import LlamaModelHandler

model_handler = LlamaModelHandler()

def get_inference(text):
    return model_handler.predict(text)
