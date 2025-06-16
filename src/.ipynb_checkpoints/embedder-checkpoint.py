
from typing import List, Tuple, Union
from PIL import Image
from PIL.Image import Image as PIL_Image
import numpy as np
import pickle
from src.image_utils import pad_image, image2tensor
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch

class Embedder:
    def __init__(self, model):
        self.model = model

        config = resolve_data_config({}, model=model)
        print(f"model config {config}")
        self.transform = create_transform(**config)

    def save_features(self, embs, featpath: str):
        feature_file = open(featpath, 'wb')
        pickle.dump(embs, feature_file)
        feature_file.close()

    def load_features(self, featpath: str):
        feature_file = open(featpath, 'rb')
        embs = pickle.load(feature_file)
        feature_file.close()
        return embs

    def img2vec(self, img):
        if isinstance(img, str):
            img = Image.open(img)
        tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            with torch.no_grad():

                out = self.model(tensor)
            feats = out.detach().cpu().numpy()
            #feats = torch.nn.functional.normalize (feats[None], p = 2, dim = 1)
            feats = np.squeeze(feats)
        return feats

    def encode_image(self, img: Union[str, PIL_Image], model, preprocess, img_size:Tuple[int, int] = (224, 224), pad_mode: str="white"):
        if isinstance(img, str):
            img = Image.open(img)
        img = Image.fromarray(pad_image(np.array(img), img_size, pad_mode))
        with torch.no_grad():
            img = preprocess(img).unsqueeze(0)
            image_features = model.encode_image(img).numpy()
        image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
        return np.squeeze(image_features)

    def encode_filepaths(self, filepaths: List[str], model, preprocess, img_size:Tuple[int, int] = (224, 224), pad_mode: str="white", return_mean: bool=False):
        images = [Image.open(p).convert("RGB") for p in filepaths]
        images = [Image.fromarray(pad_image(np.array(t), img_size, pad_mode)) for t in images]
        with torch.no_grad():
            embs = [model.encode_image(preprocess(t).unsqueeze(0)) for t in images]
            embs = [emb.detach().numpy() for emb in embs]

        if return_mean:
            return self.embs_normalize(embs)
        else:
            return np.squeeze(np.array(embs))


    def embs_normalize(self, embs):
        embs_t = torch.from_numpy(np.array(embs))
        embs_t /= embs_t.norm(dim=-1, keepdim=True)
        embs_t = embs_t.mean(0)
        embs_t /= embs_t.norm()
        return embs_t.numpy()
