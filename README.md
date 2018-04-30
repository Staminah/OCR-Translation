# OCR-Translation

Python project using OpenCV to find the text in an image and crop it. Then passing the cropped image to Tesseract to OCR it with a better result than the same non-cropped image. the project is based partially on an [existing project](https://github.com/luipillmann)

## Installation

To lunch it, create a virtual environment.

`python -m venv virtual_env_name`

Then activate it.

`virtual_env_name\Scripts\activate`

Install the requirements.

`pip install -r requiremetns.txt`

To install Tesseract [download](https://github.com/tesseract-ocr/tesseract/wiki/Downloads) it. Maybe you'll need to install also [C++ 14](http://landinghub.visualstudio.com/visual-cpp-build-tools).

After that add _tesseract.exe_ int the PATH and create a new Path Variable named _TESSDATA_PREFIX_ with the following value _C:\Program Files (x86)\Tesseract-OCR_.

## Usage

You can now execute our project with the following command.

`python crop.py -i image_name`

Below you'll find the explanation of the options.

usage: `crop.py [-h] -i IMAGE [-l LANGUAGE] [-s {0}]`

required arguments:

`-i IMAGE, --image IMAGE` path to input image to be OCR'd

optional arguments:

  `-h, --help ` show this help message and exit

  `-l LANGUAGE, --language LANGUAGE` destination language used in translation, please use 2 letters format : English -> en

  `-s {0}, --size {0}` size of the box containing text in percent of the image, must be a float value between 0 and 1

<br/>

For details on the methodology, see
http://www.danvk.org/2015/01/07/finding-blocks-of-text-in-an-image-using-python-opencv-and-numpy.html

Script created by Dan Vanderkam (https://github.com/danvk)

Adapted to Python 3 by Lui Pillmann (https://github.com/luipillmann)

Adapted to work with photographs and added OCR + Translation by Fleury Anthony and Schnaebele Marc, HE-Arc (https://github.com/Staminah/OCR-Translation.git)
