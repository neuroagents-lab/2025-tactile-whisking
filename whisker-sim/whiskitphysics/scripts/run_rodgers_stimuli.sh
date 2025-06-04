#!/bin/bash

./rodgers_stimuli.sh concave near
./rodgers_stimuli.sh concave medium
./rodgers_stimuli.sh concave far
./rodgers_stimuli.sh convex near
./rodgers_stimuli.sh convex medium
./rodgers_stimuli.sh convex far

# for dir in ../output/*; do
#     if [ -d "$dir" ]; then
#         echo "Processing directory: $dir"
#         ffmpeg -i "$dir/video.mp4" -vf "setpts=0.1*PTS,fps=100,scale=320:-1:flags=lanczos" -loop 0 -y "$dir/video.gif"
#     fi
# done