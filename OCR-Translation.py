#!/usr/bin/env python
'''Crop an image to just the portions containing text, apply OCR to cropped image and translate the text

usage: OCR-Translation.py [-h] -i IMAGE [-l LANGUAGE] [-s {0}] [-r]

required arguments:

  -i IMAGE, --image IMAGE
                        path to input image to be OCR'd

optional arguments:

  -h, --help            show this help message and exit

  -l LANGUAGE, --language LANGUAGE
                        destination language used in translation, please use 2
                        letters format : English -> en

  -s {0}, --size {0}    size of the box containing text in percent of the
                        image, must be a float value between 0 and 1

  -r, --no_rotation     disable special treatment for tilted texts

For details on the methodology, see
http://www.danvk.org/2015/01/07/finding-blocks-of-text-in-an-image-using-python-opencv-and-numpy.html
Script created by Dan Vanderkam (https://github.com/danvk)
Adapted to Python 3 by Lui Pillmann (https://github.com/luipillmann)
Adapted to work with photographs and added OCR + Translation by Fleury Anthony and Schnaebele Marc, HE-Arc (https://github.com/Staminah/OCR-Translation.git)
'''

import glob
import os
import argparse
import random
import sys
import math
import json
from collections import defaultdict
import pytesseract
from googletrans import Translator
import imutils

import cv2
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage.filters import rank_filter


def dilate(ary, N, iterations):
    """Dilate using an NxN '+' sign shape. ary is np.uint8."""

    kernel = np.zeros((N,N), dtype=np.uint8)
    kernel[(N-1)//2,:] = 1  # Bug solved with // (integer division)

    dilated_image = cv2.dilate(ary / 255, kernel, iterations=iterations)

    kernel = np.zeros((N,N), dtype=np.uint8)
    kernel[:,(N-1)//2] = 1  # Bug solved with // (integer division)
    dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations)

    return dilated_image


def props_for_contours(contours, ary):
    """Calculate bounding box & the number of set pixels for each contour."""
    c_info = []
    # For each contour
    for c in contours:
        # Get the best fitting rect (horizontal) around contours points
        x,y,w,h = cv2.boundingRect(c)
        # Create new full black image with same dimensions as given image
        c_im = np.zeros(ary.shape)
        # Draw the rectangle filled with white
        cv2.drawContours(c_im, [c], 0, 255, -1)
        # Add rectangle infos inside list
        c_info.append({
            'x1': x,
            'y1': y,
            'x2': x + w - 1,
            'y2': y + h - 1,
            'sum': np.sum(ary * (c_im > 0))/255 # Calculates the number of white pixels in the initial image that correspond to the mask zone (white)
        })
    return c_info


def union_crops(crop1, crop2):
    """Union two (x1, y1, x2, y2) rects."""
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return min(x11, x12), min(y11, y12), max(x21, x22), max(y21, y22)


def intersect_crops(crop1, crop2):
    """Intersection between two (x1, y1, x2, y2) rects"""
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return max(x11, x12), max(y11, y12), min(x21, x22), min(y21, y22)


def crop_area(crop):
    """ Computes and return a minimum area of zero """
    x1, y1, x2, y2 = crop
    return max(0, x2 - x1) * max(0, y2 - y1)


def find_border_components(contours, ary, percent_size):
    """ Return a list of contours that have a minimum area compared to full image """
    borders = []
    # Area of initial image
    area = ary.shape[0] * ary.shape[1]
    # For each contour in image
    for i, c in enumerate(contours):
        # Get best fitting rect (horizontal) to enclose current contour
        x,y,w,h = cv2.boundingRect(c)
        # If area of current rect is bigger than a certain limit
        if w * h > percent_size * area:
            print("area ", w*h)
            # Add this rect to list
            borders.append((i, x, y, x + w - 1, y + h - 1))
    return borders


def angle_from_right(deg):
    return min(deg % 90, 90 - (deg % 90))


def remove_border(contour, ary):
    """Remove everything outside a border contour."""
    # Create a new full black image with same dimensions as given image, this image will be used as mask
    c_im = np.zeros(ary.shape)

    r = cv2.minAreaRect(contour)
    degs = r[2]
    print("degree ", degs)
    box = cv2.boxPoints(r)
    box = np.int0(box)

    # Draw a pure white rectangle inside the given contours (considered as sheet)
    cv2.drawContours(c_im, [contour], 0, 255, -1)
    # Draw a black line around the rectangle to be sure to get everything inside
    cv2.drawContours(c_im, [contour], 0, 0, 4)

    # NOTE : Debug
    cv2.imwrite("imgcrop/05_remove_border_mask.png", c_im)

    # Return a new image in which each pixel is the minimum between the mask and given image (keeps everything inside the white part of the mask)
    return np.minimum(c_im, ary), degs


def find_components(edges, max_components=16):
    """Dilate the image until there are just a few connected components.
    Returns contours for these components."""
    # Perform increasingly aggressive dilation until there are just a few
    # connected components.
    dilated_image = 0
    count = max_components + 1 # To loop at least one time and get the real count
    n = 1

    # Iterates until there are less than a certain number of contours detected
    while count > max_components:
        n += 1
        # Dilates image
        dilated_image = dilate(edges, N = 3, iterations = n)
        # Convert image to uint8
        dilated_image = np.uint8(dilated_image)
        # Updates contours list in image
        _, contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Updates count
        count = len(contours)

    # Return the final contours
    return contours

def find_optimal_components_subset(contours, edges):
    """Find a crop which strikes a good balance of coverage/compactness.
    Returns an (x1, y1, x2, y2) tuple.
    """
    # Get list containing info about each contours
    c_info = props_for_contours(contours, edges)
    # Sort contour list in descending order on the number of white pixels contained in their area.
    # -> Contour with bigger area first in the list
    c_info.sort(key=lambda x: -x['sum'])
    # Total number of white pixels in initial image (canny)
    total = np.sum(edges) / 255
    # Area
    area = edges.shape[0] * edges.shape[1]

    # Get info of the first contours and delete it from list
    c = c_info[0]
    del c_info[0]
    this_crop = c['x1'], c['y1'], c['x2'], c['y2']
    crop = this_crop
    covered_sum = c['sum']

    # Iterates to try combination (union) of contours to see if they fit together
    while covered_sum < total:
        changed = False
        # Percentage of area containing text in current shape (union-contour) over to total number of pixel in image
        recall = 1.0 * covered_sum / total
        # Percentage of area dismissed in current shape (union-contour) over to initial image
        prec = 1 - 1.0 * crop_area(crop) / area
        # Fitness function, used to evaluate viability of union between two diffrent contours
        f1 = 2 * (prec * recall / (prec + recall))

        # For each contours still in info list
        for i, c in enumerate(c_info):
            # Get current contours infos
            this_crop = c['x1'], c['y1'], c['x2'], c['y2']
            # Union the saved contours with current contours
            new_crop = union_crops(crop, this_crop)
            # Calculates values with new contour unioned
            new_sum = covered_sum + c['sum']
            new_recall = 1.0 * new_sum / total
            new_prec = 1 - 1.0 * crop_area(new_crop) / area
            new_f1 = 2 * new_prec * new_recall / (new_prec + new_recall)

            # Add this crop if it improves f1 score,
            # _or_ it adds 25% of the remaining pixels for <15% crop expansion.
            # ^^^ very ad-hoc! make this smoother
            remaining_frac = c['sum'] / (total - covered_sum)
            new_area_frac = 1.0 * crop_area(new_crop) / crop_area(crop) - 1
            if new_f1 > f1 or (remaining_frac > 0.25 and new_area_frac < 0.15):
                # print('%d %s -> %s / %s (%s), %s -> %s / %s (%s), %s -> %s' % (
                #         i, covered_sum, new_sum, total, remaining_frac,
                #         crop_area(crop), crop_area(new_crop), area, new_area_frac,
                #         f1, new_f1))

                # Save the unioned contour
                crop = new_crop
                covered_sum = new_sum
                # Delete the unioned contours from info list (it can only be union once)
                del c_info[i]
                # To no break while because there's more combinations to try
                changed = True
                break

        # If no combination was better after a try with each other contours left, it is time to escape the while loop
        if not changed:
            break

    # Return the coordinates of the area containing the text to crop
    return crop


def pad_crop(crop, contours, edges, border_contour, pad_px=15):
    """Slightly expand the crop to get full contours.
    This will expand to include any contours it currently intersects, but will
    not expand past a border.
    """
    bx1, by1, bx2, by2 = 0, 0, edges.shape[0], edges.shape[1]

    # If a sheet (main border) has been found
    if border_contour is not None and len(border_contour) > 0:
        # Get infos about border contour
        c = props_for_contours([border_contour], edges)[0]
        # Calc Sheet position +  small padding
        bx1, by1, bx2, by2 = c['x1'] + 5, c['y1'] + 5, c['x2'] - 5, c['y2'] - 5

    def crop_in_border(crop):
        x1, y1, x2, y2 = crop
        x1 = max(x1 - pad_px, bx1)
        y1 = max(y1 - pad_px, by1)
        x2 = min(x2 + pad_px, bx2)
        y2 = min(y2 + pad_px, by2)
        return crop

    # Chose the more inside sheet coordinates for each of the 4 crop rect points
    crop = crop_in_border(crop)
    # Get infos about contours
    c_info = props_for_contours(contours, edges)
    changed = False
    # For each contour
    for c in c_info:
        this_crop = c['x1'], c['y1'], c['x2'], c['y2']
        # Area of current contour
        this_area = crop_area(this_crop)
        # Area of intersection between this contour and given cropping rect
        int_area = crop_area(intersect_crops(crop, this_crop))
        # Creates new cropping rect by unioning the current contour and last saved cropping rect
        new_crop = crop_in_border(union_crops(crop, this_crop))

        # If not full area contour is contained inside the save cropping rect
        # Then the new area calculated bewteen the current contour and the last saved cropping rect become the new the saved cropping rect
        if (0 < int_area < this_area) and crop != new_crop:
            print('%s -> %s' % (str(crop), str(new_crop)))
            changed = True
            crop = new_crop

    # If there has been any change to the saved cropping rect, it needs to verified again.
    if changed:
        return pad_crop(crop, contours, edges, border_contour, pad_px)
    else:
        return crop


def downscale_image(im, max_dim=2048):
    """Shrink im until its longest dimension is <= max_dim.
    Returns new_image, scale (where scale <= 1).
    """
    # Image dimensions
    a, b = im.size
    # If width and height are already smaller than max dim
    if max(a, b) <= max_dim:
        return 1.0, im
    # Get ratio to scale the bigger dimensions to max dim
    scale = 1.0 * max_dim / max(a, b)
    print('size : ', a , ' / ' , b)
    # Resize image to new scale
    new_im = im.resize((int(a * scale), int(b * scale)), Image.ANTIALIAS)

    return scale, new_im


def process_image(path, out_path, percent_size, apply_rotation):
    """ Main treatment applied to image """
    # Open specified image
    orig_im = Image.open(path)
    # Scale image size to a maximum
    scale, im = downscale_image(orig_im)

    # Canny edge detection on our image
    edges = cv2.Canny(np.asarray(im), 100, 200)

    # NOTE : Debug
    cv2.imwrite("imgcrop/01_canny_only.png", edges)

    # Dilation with simple kernel to get better contouring
    kernel = np.ones((5,5),np.uint8)
    edges = cv2.dilate(edges, kernel, iterations = 3)

    # NOTE : Debug
    cv2.imwrite("imgcrop/02_canny_dilated.png", edges)

    # Contour listing on Canny image
    _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # NOTE : Debug
    egdes_contours = np.asarray(im)
    cv2.drawContours(egdes_contours, contours, -1, (0,0,255), 3)
    cv2.imwrite("imgcrop/03_edges_contours.png", egdes_contours)

    # Try to detect contours that are a sheet border
    borders = find_border_components(contours, edges, percent_size)
    # Sort in ascending order by area on selected contours
    borders.sort(key=lambda i_x1_y1_x2_y2: (i_x1_y1_x2_y2[3] - i_x1_y1_x2_y2[1]) * (i_x1_y1_x2_y2[4] - i_x1_y1_x2_y2[2]))

    border_contour = None
    degs = None
    # If border sheet are detected
    if len(borders):
        # NOTE : Debug
        print("border ", (1+borders[0][3] - borders[0][1]) * (1+borders[0][4] - borders[0][2]))

        # Get the contour with smaller area
        border_contour = contours[borders[0][0]]

        # NOTE : Debug
        border_remove = np.asarray(im)
        cv2.drawContours(border_remove, [border_contour], -1, (0,0,255), 3)
        cv2.imwrite("imgcrop/04_border_detection.png", border_remove)

        # Remove everything outside this contour on our image
        edges, degs = remove_border(border_contour, edges)


    # Put every pixel in image that are not completely black to pure white
    edges = 255 * (edges > 0).astype(np.uint8)

    # NOTE : Debug
    cv2.imwrite("imgcrop/06_edges_border_removed.png", edges)

    # Remove ~1px borders using a rank filter.
    # maxed_rows = rank_filter(edges, -4, size=(1, 20))
    # maxed_cols = rank_filter(edges, -4, size=(20, 1))
    # debordered = np.minimum(np.minimum(edges, maxed_rows), maxed_cols)
    # edges = debordered

    # # NOTE : Debug
    # cv2.imwrite("imgcrop/edges_rank.png", edges)

    # Dilates image to decrease the number of contours
    contours = find_components(edges)

    # NOTE : Debug
    cpy_orig_im = np.asarray(im)
    cv2.drawContours(cpy_orig_im, contours, -1, (0,0,255), 3)
    cv2.imwrite("imgcrop/07_contours_border_removed.png", cpy_orig_im)

    # Image must have enough contours to contains a text
    if len(contours) == 0:
        print('%s -> (no text!)' % path)
        return

    # Try to find the best place to crop image with given contours containing texts
    crop = find_optimal_components_subset(contours, edges)
    # Adjust crop coordinates to not break texts
    crop = pad_crop(crop, contours, edges, border_contour)
    # Rescale crop coordinates to fit full size image from the begining
    crop = [int(x / scale) for x in crop]  # upscale to the original image size.

    # Crop full size image
    text_im = orig_im.crop(crop)
    # Save cropped text
    text_im.save(out_path)
    print('%s -> %s' % (path, out_path))

    # If the sheet detected is rotated we rotate the cropped image too.
    # An image with horizontal text will give better results with OCR, that's why we rotate it correctly.
    if degs and apply_rotation:
        img = cv2.imread(out_path)

        degs_rotation = angle_from_right(degs)
        # When the sheet is tilted by less than 45°, the sheet is considred in Portrait mode. Otherwise Landscape mode
        if degs < -45:
            degs_rotation = -degs_rotation

        rotated = imutils.rotate_bound(img, degs_rotation)

        # Update the cropped image
        cv2.imwrite(out_path, rotated)


def main():
    """Main programm"""
    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, type=str, help="path to input image to be OCR'd")
    ap.add_argument("-l", "--language", type=str, default="en", help="destination language used in translation, please use 2 letters format : English -> en")
    ap.add_argument("-s", "--size", type=float, default=0.25, choices=range(0, 1),  help="size of the box containing text in percent of the image, must be a float value between 0 and 1")
    ap.add_argument("-r", "--no_rotation", action='store_false', default=True,  help="disable special treatment for tilted texts")

    args = vars(ap.parse_args())

    # Process images
    path = args["image"]
    out_path = 'imgcrop/08_cropped_image.png'
    #out_path = path.replace('.jpg', '.crop.png')
    #out_path = path.replace('.png', '.crop.png')  # .png as input

    try:
        process_image(path, out_path, args["size"], args["no_rotation"])
    except Exception as e:
        print('%s %s' % (path, e))

    # File paths
    out_path_file_ocr_crop = 'imgcrop/ocr_with_crop.txt'
    out_path_file_ocr = 'imgcrop/ocr_without_crop.txt'
    out_path_file_trad = 'imgcrop/trad.txt'

    print('OCR Initial image in progress')

    # Original image OCR
    text = pytesseract.image_to_string(Image.open(path))
    file = open(out_path_file_ocr, 'w')
    file.write(text)
    file.close()

    print('OCR Cropped image in progress')

    # Cropped image OCR
    text = pytesseract.image_to_string(Image.open(out_path))
    file = open(out_path_file_ocr_crop, 'w')
    file.write(text)
    file.close()

    print('Translation in progress')

    #Translation
    translator = Translator()
    translation = translator.translate(text, dest=args["language"])
    file = open(out_path_file_trad, 'w')
    file.write(translation.text)
    file.close()


if __name__ == '__main__':
    main()
