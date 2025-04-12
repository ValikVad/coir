import torch
import numpy as np
import logging
import requests
from typing import List, Dict
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class YourCustomDEModel:
    def __init__(self, model_name="intfloat/e5-base-v2", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model_name = model_name
        self.tokenizer.add_eos_token = False

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def cls_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        # Extract the CLS token's embeddings (index 0) for each sequence in the batch
        cls_embeddings = token_embeddings[:, 0, :]
        return cls_embeddings

    def last_token_pool(self, model_output, attention_mask):
        last_hidden_states = model_output.last_hidden_state
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def encode_text(self, texts: List[str], batch_size: int = 12, max_length: int = 128) -> np.ndarray:
        logging.info(f"Encoding {len(texts)} texts...")

        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches", unit="batch"):
            batch_texts = texts[i:i+batch_size]
            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            batch_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings.append(batch_embeddings.cpu())

        embeddings = torch.cat(embeddings, dim=0)

        if embeddings is None:
            logging.error("Embeddings are None.")
        else:
            logging.info(f"Encoded {len(embeddings)} embeddings.")

        return embeddings.numpy()

    def encode_queries(self, queries: List[str], batch_size: int = 12, max_length: int = 512, **kwargs) -> np.ndarray:
        all_queries = ["query: "+ query for query in queries]
        return self.encode_text(all_queries, batch_size, max_length)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 12, max_length: int = 512, **kwargs) -> np.ndarray:
        all_texts = ["passage: "+ doc['text'] for doc in corpus]
        return self.encode_text(all_texts, batch_size, max_length)


class API_Encoder_Model:
    def __init__(self, config: Dict):
        self.api_url = config.get('api_url', 'http://example.com/api')
        self.api_key = config.get('api_key', '')
        self.batch_size = config.getint('batch_size', 12)
        self.max_length = config.getint('max_length', 128)
        logger.info(f"Initialized APIDEModel with API URL: {self.api_url}")

    def _send_request(self, texts: List[str]) -> List[np.ndarray]:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'texts': texts,
            'max_length': self.max_length
        }
        response = requests.post(self.api_url, json=data, headers=headers)
        if response.status_code != 200:
            logger.error(f"API request failed with status code {response.status_code}: {response.text}")
            raise Exception(f"API request failed with status code {response.status_code}")
        return response.json().get('embeddings', [])

    def encode_text(self, texts: List[str], batch_size: int = None, max_length: int = None) -> np.ndarray:
        if batch_size is None:
            batch_size = self.batch_size
        if max_length is None:
            max_length = self.max_length

        logging.info(f"Encoding {len(texts)} texts using API...")

        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches", unit="batch"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self._send_request(batch_texts)
            embeddings.extend(batch_embeddings)

        embeddings = np.array(embeddings)

        if embeddings is None or embeddings.size == 0:
            logging.error("Embeddings are None or empty.")
        else:
            logging.info(f"Encoded {len(embeddings)} embeddings.")

        return embeddings

    def encode_queries(self, queries: List[str], batch_size: int = None, max_length: int = None, **kwargs) -> np.ndarray:
        all_queries = ["query: "+ query for query in queries]
        return self.encode_text(all_queries, batch_size, max_length)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = None, max_length: int = None, **kwargs) -> np.ndarray:
        all_texts = ["passage: "+ doc['text'] for doc in corpus]
        return self.encode_text(all_texts, batch_size, max_length)


class APIGenerativeModel:
    def __init__(self, config: Dict):
        self.api_url = config.get('generative_api_url', 'http://example.com/generate')
        self.api_key = config.get('api_key', '')
        self.batch_size = config.getint('batch_size', 12)
        self.max_length = config.getint('max_length', 512)
        logger.info(f"Initialized APIGenerativeModel with API URL: {self.api_url}")

    def _send_request(self, prompts: List[str]) -> List[str]:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'prompts': prompts,
            'max_length': self.max_length
        }
        response = requests.post(self.api_url, json=data, headers=headers)
        if response.status_code != 200:
            logger.error(f"API request failed with status code {response.status_code}: {response.text}")
            raise Exception(f"API request failed with status code {response.status_code}")
        return response.json().get('generated_texts', [])

    def generate_text(self, prompts: List[str], batch_size: int = None, max_length: int = None) -> List[str]:
        if batch_size is None:
            batch_size = self.batch_size
        if max_length is None:
            max_length = self.max_length

        logging.info(f"Generating text for {len(prompts)} prompts using API...")

        generated_texts = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating batches", unit="batch"):
            batch_prompts = prompts[i:i+batch_size]
            batch_generated_texts = self._send_request(batch_prompts)
            generated_texts.extend(batch_generated_texts)

        if not generated_texts:
            logging.error("Generated texts are empty.")
        else:
            logging.info(f"Generated {len(generated_texts)} texts.")

        return generated_texts

