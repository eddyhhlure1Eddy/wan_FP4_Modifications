import torch
from safetensors import safe_open

class SVDModelLoader:
    
    @staticmethod
    def load_svd_compressed_model(model_path, device='cpu'):
        sd = {}
        compressed_keys = set()
        
        with safe_open(model_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            
            for key in keys:
                if key.endswith('.A'):
                    compressed_keys.add(key[:-2])
            
            print(f"Loading SVD compressed model: {len(compressed_keys)} compressed weights")
            
            for base_key in compressed_keys:
                A_key = f"{base_key}.A"
                B_key = f"{base_key}.B"
                shape_key = f"{base_key}.shape"
                
                if A_key in keys and B_key in keys:
                    A = f.get_tensor(A_key)
                    B = f.get_tensor(B_key)
                    original_dtype = A.dtype
                    
                    A_float = A.float().to(device)
                    B_float = B.float().to(device)
                    
                    reconstructed = torch.mm(A_float, B_float).cpu().to(original_dtype)
                    
                    if shape_key in keys:
                        original_shape = f.get_tensor(shape_key)
                        original_shape = tuple(original_shape.tolist())
                        if len(original_shape) > 2:
                            reconstructed = reconstructed.reshape(original_shape)
                    
                    sd[base_key] = reconstructed
                    
                    if device == 'cuda':
                        torch.cuda.empty_cache()
            
            for key in keys:
                if not (key.endswith('.A') or key.endswith('.B') or key.endswith('.shape')):
                    if key not in sd:
                        sd[key] = f.get_tensor(key)
        
        print(f"SVD model loaded: {len(sd)} total tensors")
        return sd

    @staticmethod
    def is_svd_compressed(model_path):
        try:
            with safe_open(model_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                return any(key.endswith('.A') for key in keys)
        except:
            return False

