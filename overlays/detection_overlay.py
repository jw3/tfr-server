import io
from PIL import Image, ImageDraw, ImageFont
import os,sys
sys.path.insert(1,"../")
import label_map_util
default_color = 'blue'
highlight_color = 'red'
from matplotlib import pyplot as plt
import numpy as np
import cv2
class DetectionOverlay:
  
  def __init__(self, args):
    self.args = args
    # self.labels_to_highlight = args.labels_to_highlight.split(";")
    self.label_file = args.label_file
    self.font = ImageFont.truetype("./fonts/OpenSans-Regular.ttf", 12)

  def apply_overlay(self, image_bytes, feature):
    """Apply annotation overlay over input image.
    
    Args:
      image_bytes: JPEG image.
      feature: TF Record Feature

    Returns:
      image_bytes_with_overlay: JPEG image with annotation overlay.
    """

    bboxes = self.get_bbox_tuples(feature)
    image_bytes_with_overlay = self.draw_bboxes(image_bytes, bboxes)
    
    return image_bytes_with_overlay
  
  def apply_overlay_img(self, img, feature):
    """Apply annotation overlay over input image.
    
    Args:
      img: JPEG image.
      feature: TF Record Feature

    Returns:
      image_bytes_with_overlay: JPEG image with annotation overlay.
    """

    bboxes = self.get_bbox_tuples(feature)
    # image_with_overlay = self.draw_bboxes(img, bboxes)
    image_with_overlay = self.draw_bboxes_img(img, bboxes)
    
    return image_with_overlay

  def get_bbox_tuples(self, feature):
    """ From a TF Record Feature, get a list of tuples representing bounding boxes
    
    Args:
      feature: TF Record Feature
    Returns:
      bboxes (list of tuples): [ (label, xmin, xmax, ymin, ymax), (label, xmin, xmax, ymin, ymax) , .. ]
    """
    bboxes = []
    if self.args.bbox_name_key in feature:
      for ibbox, label in enumerate (feature[self.args.bbox_name_key].bytes_list.value):
        bboxes.append( (label.decode("utf-8"),
                        feature[self.args.bbox_xmin_key].float_list.value[ibbox],
                        feature[self.args.bbox_xmax_key].float_list.value[ibbox],
                        feature[self.args.bbox_ymin_key].float_list.value[ibbox],
                        feature[self.args.bbox_ymax_key].float_list.value[ibbox]
      ) )
    else:
      print("Bounding box key '%s' not present." % (self.args.bbox_name_key))
    return bboxes

  def bbox_color(self, label):

    PATH_TO_LABELS = os.path.join(self.label_file)
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    labels = ''
    for i in range(1,len(category_index)):
      labels += "'" + category_index[i]["name"] + "',"

    # if label in self.labels_to_highlight:
    if label in labels:
      return highlight_color
    else:
      return default_color

  def bboxes_to_pixels(self, bbox, im_width, im_height):
    """
    Convert bounding box coordinates to pixels.
    (It is common that bboxes are parametrized as percentage of image size
    instead of pixels.)

    Args:
      bboxes (tuple): (label, xmin, xmax, ymin, ymax)
      im_width (int): image width in pixels
      im_height (int): image height in pixels
    
    Returns:
      bboxes (tuple): (label, xmin, xmax, ymin, ymax)
    """
    if self.args.coordinates_in_pixels:
      return bbox
    else:
      label, xmin, xmax, ymin, ymax = bbox
      return [label, xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height]

  def draw_bboxes(self, image_bytes, bboxes):
    """Draw bounding boxes onto image.
    
    Args:
      image_bytes: JPEG image.
      bboxes (list of tuples): [ (label, xmin, xmax, ymin, ymax), (label, xmin, xmax, ymin, ymax) , .. ]
    
    Returns:
      image_bytes: JPEG image including bounding boxes.
    """
    img = Image.open(io.BytesIO(image_bytes))

    draw = ImageDraw.Draw(img)

    width, height = img.size

    for bbox in bboxes:
      label, xmin, xmax, ymin, ymax = self.bboxes_to_pixels(bbox, width, height)
      draw.rectangle([xmin, ymin, xmax, ymax], outline=self.bbox_color(label))

      w, h = self.font.getsize(label)
      draw.rectangle((xmin, ymin, xmin + w + 4, ymin + h), fill="white")

      draw.text((xmin+4, ymin), label, fill=self.bbox_color(label), font=self.font)

    with io.BytesIO() as output:
      img.save(output, format="JPEG")
      output_image = output.getvalue()
    return output_image

  def draw_bboxes_img(self, img, bboxes):
    """Draw bounding boxes onto image.
    
    Args:
      img_bytes: JPEG image.
      bboxes (list of tuples): [ (label, xmin, xmax, ymin, ymax), (label, xmin, xmax, ymin, ymax) , .. ]
    
    Returns:
      img: JPEG image including bounding boxes.
    """

    width, height = img.shape[1], img.shape[0]

    for bbox in bboxes:
      
      label, xmin, xmax, ymin, ymax = self.bboxes_to_pixels(bbox, width, height)
      
      xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
      
      font_scale = .8
      thickness = 2
      font_family = cv2.FONT_HERSHEY_SIMPLEX
      font_size = cv2.getTextSize(label, font_family, font_scale, thickness)
      text_point = (xmin, ymin+20)
      temp_p = (text_point[0], text_point[1] - font_size[0][1])
      cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 0), thickness)
      cv2.rectangle(img, temp_p, (text_point[0] + font_size[0][0], text_point[1] + font_size[0][1] - 5), (0, 0, 0),cv2.FILLED)
      cv2.putText(img, label, text_point, font_family, font_scale, (255, 255, 255),lineType=cv2.LINE_AA,thickness=thickness) 

      cv2.imshow("TFRECORD DATA", img)
      cv2.waitKey(0)

    return img