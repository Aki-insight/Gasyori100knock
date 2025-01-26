import matplotlib.pyplot as plt
import numpy as np
import cv2


def bgr2gray(img):
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray

# DFT
def dft(img, K=128, L=128, channel=3):
    # prepare out image
    H, W, _ = img.shape

    # Prepare DFT coefficient
    G = np.zeros((L, K, channel), dtype=np.complex128)

    # prepare processed index corresponding to original image positions
    # 画像処理の場合、xが行番号（縦軸）、yが列番号（横軸）を表す。
    x = np.arange(H).repeat(W).reshape(H, -1)
    y = np.tile(np.arange(W), (H, 1))

    # dft
    for c in range(channel):
        for l in range(L):
            for k in range(K):
                G[l, k, c] = np.sum(img[..., c] * np.exp(-2j * np.pi * (x * k / K + y * l / L))) / np.sqrt(K * L)
                #for n in range(N):
                #    for m in range(M):
                #        v += gray[n, m] * np.exp(-2j * np.pi * (m * k / M + n * l / N))
                #G[l, k] = v / np.sqrt(M * N)

    return G

# IDFT
def idft(G, channel=3):
    # prepare out image
    H, W, _ = G.shape
    out = np.zeros((H, W, channel), dtype=np.float32)

    # prepare processed index corresponding to original image positions
    # 画像処理の場合、xが行番号（縦軸）、yが列番号（横軸）を表す。
    x = np.arange(H).repeat(W).reshape(H, -1)
    y = np.tile(np.arange(W), (H, 1))

    # idft
    for c in range(channel):
        for l in range(H):
            for k in range(W):
                out[l, k, c] = np.abs(np.sum(G[..., c] * np.exp(2j * np.pi * (x * k / W + y * l / H)))) / np.sqrt(W * H)

    # clipping
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    return out

def bpf(G, low_ratio=0.1, high_ratio=0.5, channel=3):
    H, W, _ = G.shape

    # 低周波成分を中心に集める
    _G = np.zeros_like(G)
    _G[:H//2, :W//2] = G[H//2:, W//2:] # 右下
    _G[:H//2, W//2:] = G[H//2:, :W//2] # 左下
    _G[H//2:, :W//2] = G[:H//2, W//2:] # 右上
    _G[H//2:, W//2:] = G[:H//2, :W//2] # 左上

    # 原点(画像中央)からの距離を計算
    # 画像処理の場合、xが行番号（縦軸）、yが列番号（横軸）を表す。
    x = np.arange(H).repeat(W).reshape(H, -1)
    y = np.tile(np.arange(W), (H, 1))

    _x = x - H // 2
    _y = y - W // 2
    r = np.sqrt(_x ** 2 + _y ** 2)

    # フィルタを適用
    mask = np.ones((H, W), dtype=np.float32)
    mask[(r < (W // 2 * low_ratio)) | (r > (W // 2 * high_ratio))] = 0 # low_ratioとhigh_ratioの中間の周波数を残す。
    mask = np.repeat(mask, channel).reshape(H, W, channel)
    _G *= mask

    # 元の場所に戻す
    G[:H//2, :W//2] = _G[H//2:, W//2:]
    G[:H//2, W//2:] = _G[H//2:, :W//2]
    G[H//2:, :W//2] = _G[:H//2, W//2:]
    G[H//2:, W//2:] = _G[:H//2, :W//2]

    return G

path = "/work/Gasyori100knock-1/Tutorial/assets/imori.jpg"
img = cv2.imread(path)
G = dft(img)
G = bpf(G)
out = idft(G)

plt.imshow(out)
plt.show()
