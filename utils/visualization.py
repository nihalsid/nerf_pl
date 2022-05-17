import torchvision.transforms as T
import numpy as np
from PIL import Image
from matplotlib import cm

def visualize_depth(depth):
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8) # normalize to 0~1
    x_ = Image.fromarray((cm.get_cmap('jet')(x) * 255).astype(np.uint8))
    x_ = T.ToTensor()(x_)[:3, :, :]
    return x_
