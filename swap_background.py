import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

if __name__ == "__main__":
    filepath = "./Cropped Images To Test SwamImageBackground/black on white.png"
    out_filepath = "./Cropped Images To Test SwamImageBackground/white on gray.png"
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imread.html
    # img = mpimg.imread(filepath)

    img = numpy.array(Image.open(filepath).convert('L'))

    thresh = 180
    img = numpy.where(img >= thresh, 1, 0)

    # convert black to foreground object color, white to background color
    background_color = numpy.array([142, 153, 162])
    foreground_color = numpy.array([256, 256, 256])
    bacpkground_img = numpy.expand_dims(img, 2) * background_color
    foreground_img= numpy.expand_dims(1 - img, 2) * foreground_color
    out_img = foreground_img + bacpkground_img
    plt.imshow(out_img)
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.show()

    # check distribution
    #plt.hist(img.ravel(), bins=256, range=(0.0, 256.0), fc='k', ec='k')
    plt.imsave(out_filepath, arr=img)