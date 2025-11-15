import cv2, math, random, numpy as np
from tensorflow.keras.utils import Sequence

def safe_read_image(p, flags=cv2.IMREAD_COLOR):
    img = cv2.imread(p, flags)
    return img if img is not None else np.zeros((256,256,3), np.uint8)

class MultiTaskSequence(Sequence):
    def __init__(self, samples, mask_dir, batch_size, augment=False, tfm=None):
        self.samples = samples
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.augment = augment
        self.tfm = tfm
        random.shuffle(self.samples)

    def __len__(self):
        return math.ceil(len(self.samples) / self.batch_size)

    def __getitem__(self, idx):
        batch = self.samples[idx*self.batch_size: (idx+1)*self.batch_size]

        imgs, cls, scratch, masks = [], [], [], []
        for path, label in batch:

            img = safe_read_image(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))

            if label == 1:
                mask_path = f"{self.mask_dir}/{path.split('/')[-1]}"
                m = safe_read_image(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                m = np.zeros((256,256), np.uint8)

            m = cv2.resize(m, (256, 256))

            augmented = self.tfm(image=img, mask=m)
            img_t = augmented["image"] / 255.0
            m_t = augmented["mask"] / 255.0
            m_t = m_t[..., None]

            scratch_score = float(m_t.sum() / (256*256))

            imgs.append(img_t)
            cls.append([label])
            scratch.append([scratch_score])
            masks.append(m_t)

        return np.array(imgs), {
            "class_output": np.array(cls),
            "scratch_output": np.array(scratch),
            "mask_output": np.array(masks)
        }
