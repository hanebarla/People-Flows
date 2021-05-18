#!/bin/sh

ffmpeg -r 30 -start_number $1 -i Demo/demo-%d.png -vcodec libx264 -pix_fmt yuv420p -r 30 Demo/out-$1.mp4