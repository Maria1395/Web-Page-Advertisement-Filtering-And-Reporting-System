import urllib.request
import ctypes
from html.parser import HTMLParser

# Grab image url
response = urllib.request.urlopen('http://apod.nasa.gov/apod/astropix.html')
html = response.read()

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        # Only parse the 'anchor' tag.
        if tag == "a":
           # Check the list of defined attributes.
           for name, value in attrs:
               # If href is defined, print it.
               if name == "href":
                   if value[len(value)-3:len(value)]=="jpg":
                       #print value
                       self.output=value

parser = MyHTMLParser()
parser.feed(html.decode('UTF-8'))
imgurl='http://apod.nasa.gov/apod/'+parser.output
print(imgurl)

# Save the file
img = urllib.request.urlopen(imgurl)
localFile = open('desktop.jpg', 'wb')
localFile.write(img.read())
localFile.close()

# set to desktop(windows method)
'''SPI_SETDESKWALLPAPER = 20 
ctypes.windll.user32.SystemParametersInfoA(SPI_SETDESKWALLPAPER, 0, "desktop.jpg" , 0)'''


