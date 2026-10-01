import os, urllib.request
os.makedirs("data", exist_ok=True)
base = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
for f in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
          "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
    urllib.request.urlretrieve(base + f, "data/" + f)
    print("downloaded", f)