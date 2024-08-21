from sentence_transformers import SentenceTransformer
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

class Text_encoder():
    def __init__(self, device, model_name='weights/all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, sentences):
        return self.model.encode(sentences)

