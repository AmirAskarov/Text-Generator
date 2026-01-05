from openai import OpenAI
from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
import base64
import os

