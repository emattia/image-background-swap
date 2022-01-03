import numpy
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from PIL import Image
import cv2

if __name__ == "__main__":
    filepath = "./Cropped Images To Test SwamImageBackground/black on white.png"
    out_filepath = "./Cropped Images To Test SwamImageBackground/white on gray.png"
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imread.html
    # img = mpimg.imread(filepath)

    img = numpy.array(Image.open(filepath).convert('L'))

    thresh = 180
    img_mask = numpy.where(img >= thresh, 1, 0)

    # convert black to foreground object color, white to background color
    background_color = numpy.array([256, 256, 256]) # numpy.array([0, 0, 0])
    foreground_color = numpy.array([0, 0, 0]) # numpy.array([256, 256, 256])
    bacpkground_img = numpy.expand_dims(img_mask, 2) * background_color
    foreground_img= numpy.expand_dims(1 - img_mask, 2) * foreground_color
    
    out_img = foreground_img + bacpkground_img
    plt.imshow(out_img)
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.show()

    remove_background = True
    if remove_background:
        out_img = out_img / 256.
        plt.imsave(out_filepath, arr=out_img)
        rgb_image = cv2.imread(out_filepath)
        rgba_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2RGBA)
        rgba_image[:,:,3] = ((1-img_mask)*255).astype('uint8') 
        _, mask = cv2.threshold(rgba_image[:, :, 3], 0, 255, cv2.THRESH_BINARY)
        ### USE THIS TO SAVE - NOT MATPLOTLIB ### 
        out_filepath = "./Cropped Images To Test SwamImageBackground/white mask.png"
        cv2.imwrite(out_filepath, rgba_image)
        plt.imshow(rgba_image)
        plt.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.title("CHANGE cv2.imwrite filepath to save image without background, not this matplotlib widget.")
        plt.show()

    # check distribution
    #plt.hist(img.ravel(), bins=256, range=(0.0, 256.0), fc='k', ec='k')
