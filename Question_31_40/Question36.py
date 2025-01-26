import cv2
import numpy as np
import matplotlib.pyplot as plt

# DCT hyoer-parameter
T = 8
K = 8
channel = 3

# DCT weight
def calc_weight(x, y, u, v):
    cu = 1.
    cv = 1.
    if u == 0:
        cu /= np.sqrt(2)
    if v == 0:
        cv /= np.sqrt(2)
    theta = np.pi / (2 * T)
    return ((2*cu*cv/T) * np.cos((2*x+1)*u*theta) * np.cos((2*y+1)*v*theta))

# DCT
def dct(img):
    H, W, _ = img.shape

    F = np.zeros((H, W, channel), dtype=np.float32)

    for c in range(channel):
        for xi in range(0, H, T):
            for yi in range(0, W, T):
                for u in range(T):
                    for v in range(T):
                        for x in range(T):
                            for y in range(T):
                                F[u+xi, v+yi, c] += img[x+xi, y+yi, c] * calc_weight(x, y, u, v)

    return F


# IDCT
def idct(F):
    H, W, _ = F.shape

    out = np.zeros((H, W, channel), dtype=np.float32)

    for c in range(channel):
        for xi in range(0, H, T):
            for yi in range(0, W, T):
                for x in range(T):
                    for y in range(T):
                        for u in range(K):
                            for v in range(K):
                                out[x+xi, y+yi, c] += F[u+xi, v+yi, c] * calc_weight(x, y, u, v)

    out = np.clip(out, 0, 255)
    out = np.round(out).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    return out


path = "/work/Gasyori100knock-1/Tutorial/assets/imori.jpg"
img = cv2.imread(path)
F = dct(img)
out = idct(F)

plt.imshow(out)
plt.show()
