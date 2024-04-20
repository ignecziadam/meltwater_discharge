# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:58:07 2022

@author: sv21669
"""

import sys
import json

#
input_stream = sys.stdin.read()
info = json.loads(input_stream)

#
x_size = info["size"][0]
y_size = info["size"][1]

upperLeft_x = info["geoTransform"][0]
upperLeft_y = info["geoTransform"][3]

res_x = info["geoTransform"][1]
res_y = info["geoTransform"][5]

#
x = [upperLeft_x,
     upperLeft_x + x_size * res_x]

y = [upperLeft_y,
     upperLeft_y + y_size * res_y]

#
output = f"{min(x)} {min(y)} {max(x)} {max(y)}"

print(output)
