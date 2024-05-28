from sentence_transformers import SentenceTransformer
import torch
from InstructorEmbedding import INSTRUCTOR


class Sentence_Embedder:

    def __init__(
            self, 
            # embedder_name='all-mpnet-base-v2', 
            embedder_name='hkunlp/instructor-large', 
            device='cuda:0', 
            lp=2, seed=123
        ):
        self.embedder_name=embedder_name
        if 'instructor' in embedder_name:
            self.model = INSTRUCTOR(embedder_name)
        else:
            self.model = SentenceTransformer(embedder_name)
        self.device=device
        self.lp=lp
        if device is not None:
            self.model=self.model.to(device)
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)     

    def get_embedding(self, sentence):
        if isinstance(sentence, list):
            text_instruction_pairs = [
                ["Represent the paragraph:", sub_sentence]
                for sub_sentence in sentence
            ]
        else:
            text_instruction_pairs = [
                ["Represent the paragraph:", sentence],
            ]
        embedding = self.model.encode(text_instruction_pairs, )
        if embedding.shape[0]==1:
            embedding = embedding.reshape(embedding.shape[1])
        return embedding
    
