import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_all(path, func, *args): 
    for filename in os.listdir(path):
        image = mpimg.imread(os.path.join(path, filename))
        plot_comparision(func, image, *args)

def plot_comparision(func, image, *args):
    new_image = func(image, *args)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Before', fontsize=18)
    ax2.imshow(new_image)
    ax2.set_title('After', fontsize=18)
    plt.show()

   
