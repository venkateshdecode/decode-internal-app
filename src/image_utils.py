from torch import stack as torch_stack 
from torchvision.utils import make_grid as torchvision_make_grid
from torchvision import transforms
from PIL import Image, ImageChops, ImageStat
import numpy as np 
import cv2 
import math 
from io import BytesIO
import requests
import PIL 
import base64


def image2tensor(img):
    return torch.as_tensor(np.array(img, dtype=np.float32)).transpose(2, 0)[None]

def cosine_similarity(v1, v2):
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if (not mag1) or (not mag2):
        return 0
    return np.dot(v1, v2) / (mag1 * mag2)

def read_image(image_path):
    with open(image_path, "rb") as f:
        img = Image.open(f)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.load()
    return img

def get_image_size_after_resize_preserving_aspect_ratio(h, w, target_size):
    aspect_ratio_h_to_w = float(h) / w
    w = int(target_size / math.sqrt(aspect_ratio_h_to_w))
    h = int(target_size * math.sqrt(aspect_ratio_h_to_w))
    h, w = (1 if s <= 0 else s for s in (h, w))  # filter out crazy one pixel images
    return h, w

def download_image(url: str) -> PIL.Image:
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


def get_concat_images_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_images_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


    # Transforms

import torch
def normalize_embeddings(embs):
    if isinstance(embs, list):
        embs = np.array(embs)
    embs_t = torch.from_numpy(embs)
    embs_t /= embs_t.norm(dim=-1, keepdim=True)
    embs_t = embs_t.mean(0)
    embs_t /= embs_t.norm()
    return embs_t.numpy()

# Function to remove black frames (or other extremes) for the collage
def is_black_or_white_frame(f): 
    extrema = Image.open(f).convert("L").getextrema()
    if not(extrema[0] < 5 and extrema[1] <5) and not(extrema[0] > 230 and extrema[1] >230):
        return False 
    else:
        return True

## check if a frame is valid, i.e. not black or white etc.                
def is_valid_frame(frame_path, DEFAULT_IMSIZE = 780):
    im = Image.open(frame_path)
    try:
        im = image_resize(np.array(im), DEFAULT_IMSIZE)
        im = Image.fromarray(im)

        extrema = im.convert("L").getextrema() 
        if not(extrema[0] < 5 and extrema[1] <5) and not(extrema[0] >200 and extrema[1] >200) and not(abs(extrema[0] - extrema[1]) < 5) : #avoid black/white/uniform end frames           
            im.save(frame_path)
            return True 
        else:
            return False
    except:
        
        return False     

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def create_collage(images, title: str, outpath: str="", n_images_per_row: int = 20, pad_value=255):

    totensor = transforms.ToTensor()

    if isinstance(images[0], str):
        frames_t = torch_stack([totensor(Image.open(f)) for f in images])
    else: #assume PIL Images
        frames_t = torch_stack([totensor(f) for f in images])

    if n_images_per_row > 0:
        grid_img = torchvision_make_grid(frames_t, nrow=n_images_per_row, pad_value=pad_value)
    else:
        grid_img = torchvision_make_grid(frames_t, pad_value=pad_value)

    collage_img = transforms.ToPILImage()(grid_img)
    if outpath !="":
        collage_img.save(outpath)
    return collage_img

def trim_image_borders(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def is_black(f): 
    extrema = Image.open(f).convert("L").getextrema()
    if not(extrema[0] < 5 and extrema[1] <5) and not(extrema[0] > 230 and extrema[1] >230):
        return False 
    else:
        return True
    
def remove_black_frames(frames): 
    frames = [f for f in frames if not is_black(f)]
    return frames 

def get_thumbnail(path, width, height):
    i = Image.open(path)
    i.thumbnail((width, height), Image.LANCZOS)
    return i

def get_thumbnail_img(img, width, height):
    i = img
    i.thumbnail((width, height), Image.LANCZOS)
    return i


def get_thumbnail_text(path):
    i = Image.open(path)
    i.thumbnail((300, 80), Image.LANCZOS)
    return i

def get_thumbnail_icon(path):
    i = Image.open(path)
    i.thumbnail((150, 150), Image.LANCZOS)
    return i

def image_base64(im, width, height, thumbnail=True):
    if thumbnail:
        if isinstance(im, str):
            im = get_thumbnail(im, width, height)
        else: 
            im = get_thumbnail_img(im, width, height)
    else: 
        if isinstance(im, str): 
            im = Image.open(im).resize((width, height))

    with BytesIO() as buffer:
        im.save(buffer, 'png')
        return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im, thumbnail=True, width=150, height=150):
    return f'<img src="image_embedding_data:image/png;base64,{image_base64(im, thumbnail, width=width, height=height)}">'

def decode_logo_formatter(im, thumbnail=True):
    return f'<img id="dec_logo", align="right", onclick="window.open("https://decodemarketing.com", "_blank")", src="image_embedding_data:image/png;base64,{image_base64(im, thumbnail)}">'

def client_logo_formatter(im):
    return f'<img id="client_logo", align="left", src="image_embedding_data:image/png;base64,{image_base64(im)}">'

def image_formatter_barchart(im):
    return f'"image_embedding_data:image/png;base64,{image_base64(im)}"'

def image_formatter_pdf(im):
    return f'"image_embedding_data:image/png;base64,{image_base64(im)}"'

######################################################################
########## Utils for drawing stuff on image ###################
####################################################################
import colorsys


def generate_colors(n, exclude_colors=[]):
    """
    Define n distinct bounding box colors

    Args:
    n: number of colors
    Returns:
    colors: (n, 3) np.array with RGB integer values in [0-255] range
    """
    hsv_tuples = [(x / n, 1., 1.) for x in range(n)]
    colors = 255 * np.array([colorsys.hsv_to_rgb(*x) for x in hsv_tuples])
    colors = [list(c.astype(int)) for c in colors]

    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.

    return [(int(c[0]), int(c[1]), int(c[2])) for c in colors if not c in exclude_colors]

def draw_box(image, box, color, thickness=4, draw_margin=2):

    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0] - draw_margin, b[1] -draw_margin), (b[2] +draw_margin, b[3]+draw_margin), color, thickness, cv2.LINE_AA)


def draw_caption(image, box, caption, font_size=2, font_color = (255,255,255), font_thickness = 1):

    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 20), cv2.FONT_HERSHEY_PLAIN, font_size, (0, 0, 0), font_thickness)
    cv2.putText(image, caption, (b[0], b[1] - 20), cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), font_thickness)

    
def draw_boxes(image, boxes, color, thickness=1):
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)


def draw_detections(image, boxes,  labels,colors):

    for i in range(len(boxes)):
        c = colors[i]
        draw_box(image, boxes[i, :], color=c)

        caption =""
        if len(labels) >0:
            caption = labels[i]
            
        draw_caption(image, boxes[i, :], caption)

def write_text_on_frame(text: str, frame: np.array, left=5, top=5, font=cv2.FONT_HERSHEY_SIMPLEX, font_size=1.0,
                        font_thickness=2,
                        font_color=(0, 0, 0), background_color=(255, 255, 255)):
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_size,
                                                thickness=font_thickness)[0]

    border_size = 10
    text_box_coords = (
        (left - border_size, top - border_size), (left + text_width + border_size, top + text_height + border_size))
    cv2.rectangle(frame, text_box_coords[0], text_box_coords[1], background_color, cv2.FILLED)
    cv2.putText(frame, text, (left, top + text_height), font, font_size, font_color, font_thickness,
                cv2.LINE_AA)

    return frame

from PIL import ImageDraw 
from PIL import Image as PIL_Image 
def draw_boxes_PIL(image: PIL_Image, bboxes, color="green",width=5):
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size 
    for bbox in bboxes:
        left = bbox[0]
        top = bbox[1]
        right = bbox[2]
        bottom = bbox[3]
        draw.rectangle(((left, top), (right , bottom)), outline=color, width=width)
    del draw
    return image         
            
########## UTILS for Results DOWNLOAD ############
import os
import base64
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="image_embedding_data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def to_excel_multi(dfs, sheetnames):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for df, sheetname in zip(dfs, sheetnames):
            df.to_excel(writer, sheet_name=sheetname, index=False)
        writer.save()
    processed_data = output.getvalue()
    return processed_data

def to_excel_one(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def df2excel(dfs, sheetnames, filename, xltype="single"):
    """Generates a link allowing the image_embedding_data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    if xltype=="single":
        val = to_excel_one(dfs)
    elif xltype == "multi":
        val = to_excel_multi(dfs, sheetnames)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="image_embedding_data:application/octet-stream;base64,{b64.decode()}" download="{filename}.xlsx">Download excel file</a>' # decode b'abc' => abc

    return doc

import cv2
def pad_image(img, shape, mode = 'constant_mean'):
    """
    Resize and pad image to given size.

    Args:
    img: (H, W, C) input numpy array
    shape: (H', W') destination size
    mode: filling mode for new padded pixels. Default = 'constant_mean' returns
        grayscale padding with pixel intensity equal to mean of the array. Other
        options include np.pad() options, such as 'edge', 'mean' (by row/column)...
    Returns:
    new_im: (H', W', C) padded numpy array
    """
    if mode == 'constant_mean':
        mode_args = {'mode': 'constant', 'constant_values': np.mean(img)}
    elif mode == "white":
        mode_args = {'mode': 'constant', 'constant_values': 255}
    else:
        mode_args = {'mode': mode}

    ih, iw = img.shape[:2]
    h, w = shape[:2]

    # first rescale image so that largest dimension matches target
    scale = min(w/iw, h/ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img = cv2.resize(img, (nw, nh))

    # center-pad rest of image: compute padding and split in two
    xpad, ypad = shape[1]-nw, shape[0]-nh
    xpad = (xpad//2, xpad//2+xpad%2)
    ypad = (ypad//2, ypad//2+ypad%2)

    new_im = np.pad(img, pad_width=(ypad, xpad, (0,0)), **mode_args)

    return new_im


def img_data_to_png_data(img_data):
    with io.BytesIO() as f:
        f.write(img_data)
        img = PIL.Image.open(f)

        with io.BytesIO() as f:
            img.save(f, "PNG")
            f.seek(0)
            return f.read()


def apply_exif_orientation(image):
    try:
        exif = image._getexif()
    except AttributeError:
        exif = None

    if exif is None:
        return image

    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in exif.items()
        if k in PIL.ExifTags.TAGS
    }

    orientation = exif.get("Orientation", None)

    if orientation == 1:
        # do nothing
        return image
    elif orientation == 2:
        # left-to-right mirror
        return PIL.ImageOps.mirror(image)
    elif orientation == 3:
        # rotate 180
        return image.transpose(PIL.Image.ROTATE_180)
    elif orientation == 4:
        # top-to-bottom mirror
        return PIL.ImageOps.flip(image)
    elif orientation == 5:
        # top-to-left mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_270))
    elif orientation == 6:
        # rotate 270
        return image.transpose(PIL.Image.ROTATE_270)
    elif orientation == 7:
        # top-to-right mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_90))
    elif orientation == 8:
        # rotate 90
        return image.transpose(PIL.Image.ROTATE_90)
    else:
        return image

import io
def img_data_to_pil(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_pil = PIL.Image.open(f)
    return img_pil


def img_data_to_arr(img_data):
    img_pil = img_data_to_pil(img_data)
    img_arr = np.array(img_pil)
    return img_arr

def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr