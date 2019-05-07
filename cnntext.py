"""
dumpimages.py
    Downloads all the images on the supplied URL, and saves them to the
    specified output file ("/test/" by default)

Usage:
    python dumpimages.py http://example.com/ [output]
"""
from bs4 import BeautifulSoup as bs
from urllib.request import (
    urlopen, urlparse, urlunparse, urlretrieve, urljoin)
import os
import sys

def main(url, out_folder="/test/"):
    """Downloads all the images at 'url' to /test/"""
    soup = bs(urlopen(url),features='html.parser')
    parsed = list(urlparse(url))

    for image in soup.findAll("img"):
        print("Image: %(src)s" % image)
        image_url = urljoin(url, image['src'])
        filename = image["src"].split("/")[-1]
        outpath = os.path.join(out_folder, filename)
        urlretrieve(image_url, outpath)

def _usage():
    print("usage: python dumpimages.py http://example.com [outpath]")

if __name__ == "__main__":
    url = sys.argv[-1]
    out_folder = "/test/"
    if not url.lower().startswith("http"):
        out_folder = sys.argv[-1]
        url = sys.argv[-2]
        if not url.lower().startswith("http"):
            _usage()
            sys.exit(-1)
    main(url, out_folder)
    
    
import cv2
import sys
import pytesseract

if __name__ == '__main__':

  
  # Read image path from command line
 imPath = "/home/maria/Downloads/image4.jpeg"
    
  # Uncomment the line below to provide path to tesseract manually
  # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

  # Define config parameters.
  # '-l eng'  for using the English language
  # '--oem 1' for using LSTM OCR Engine
config = ('-l eng --oem 1 --psm 3')

  # Read image from disk
im = cv2.imread(imPath, cv2.IMREAD_COLOR)

  # Run tesseract OCR on image
text = pytesseract.image_to_string(im, config=config)

  # Print recognized text
print(text)

#Save file
f = open("textmodel.txt" , "w+")
f.write(text)
f.close()


