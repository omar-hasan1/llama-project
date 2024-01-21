from llama_cpp import Llama
from app.config import MODEL_NAME

class LlamaModelHandler:
    def __init__(self):
        try:
            self.model = Llama(model_path="app/models/llama-2-7b-chat.ggmlv3.q8_0.bin")
            
        except Exception as e:
            print(f"Error loading Llama model: {str(e)}")

    def predict(self, text):
        
        text_bytes = text.encode("utf-8")
        
        tokens = self.model.tokenize(text_bytes)

        tokens = [int(token) for token in tokens]
        
        try:
            
            if tokens:
                
                response = self.model.eval(tokens)
                
                return response
            else:
                print("Tokens should be a list of integers.")
                return None
        
        except Exception as e:
            print(f"Error in predict: {str(e)}")
            return None