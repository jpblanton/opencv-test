import io
import socket
import struct
from datetime import datetime
from pytz import timezone

import cv2
import numpy as np
from decouple import config


def decide_color(img, g_l, g_u, r_l, r_u):

    grn_mask = cv2.inRange(img, grn_lower, grn_upper)
    grn_out = cv2.bitwise_and(img, img, mask=grn_mask)
    red_mask = cv2.inRange(img, red_lower, red_upper)
    red_out = cv2.bitwise_and(img, img, mask=red_mask)

    grn_sum = grn_out.sum()
    red_sum = red_out.sum()
    # with light on, img_sum > 100 million
    # with off, more like 10 million or less
    # probably won't work in real env
    # but ok for now - can then use greater of red/green to predict color
    img_sum = img.sum()

    if img_sum > 50_000_000:
        # too bright
        # TODO: solve this - actually, decide if necessary when inside tent
        return (None, grn_sum, red_sum, img_sum)
    else:
        if grn_sum > red_sum:
            return ('grn', grn_sum, red_sum, img_sum)
        else:
            return ('red', grn_sum, red_sum, img_sum)


ctz = timezone(config('TIMEZONE'))
# Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
# all interfaces)
server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 9999))
server_socket.listen(0)

grn_lower = np.array(config('GRN_LOWER').split(','), dtype='uint8')
grn_upper = np.array(config('GRN_UPPER').split(','), dtype='uint8')
red_lower = np.array(config('RED_LOWER').split(','), dtype='uint8')
red_upper = np.array(config('RED_UPPER').split(','), dtype='uint8')

# Accept a single connection and make a file-like object out of it
connection = server_socket.accept()[0].makefile('rb')
try:
    while True:
        # Read the length of the image as a 32-bit unsigned int. If the
        # length is zero, quit the loop
        image_len = struct.unpack('<L', connection.read(4))[0]
        if not image_len:
            break
        # Construct a stream to hold the image data and read the image
        # data from the connection
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        # Rewind the stream, open it as an image with cv2 and do some
        # processing on it
        # Construct a numpy array from the stream
        image_stream.seek(0)
        data = np.fromstring(image_stream.getvalue(), dtype=np.uint8)
        # "Decode" the image from the array, preserving colour
        image = cv2.imdecode(data, 1)
        # save image to directory
        # TODO: process and store in db
        ct = datetime.now(tz=ctz).strftime('%Y-%m-%d_%H:%M:%S.%f')
        cv2.imwrite(f'raw_images/img{ct}.jpg', image)
        # color, grn, red, img = decide_color(image, grn_lower, grn_upper, red_lower, red_upper)
        # TODO: store those in database
finally:
    # feel like this line is just for client...
    # connection.write(struct.pack('<L', 0))
    connection.close()
    server_socket.close()
