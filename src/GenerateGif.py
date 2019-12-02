import os
import imageio
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
CWD_DIR = os.path.dirname(__file__)


def CreateGif(image_list, gif_name, duration=0.2):
    '''
    :param image_list: 这个列表用于存放生成动图的图片
    :param gif_name: 字符串，所生成gif文件名，带.gif后缀
    :param duration: 图像间隔时间
    :return:
    '''
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    logger.info("Save Gif picture to {}/{}".format(CWD_DIR, gif_name))


if __name__ == "__main__":
    img_dir = 'imgs'
    filenames = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
    outname = "Path.gif"
    CreateGif(filenames, outname, duration=0.2)
