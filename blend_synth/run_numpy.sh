#!/bin/bash
for num_frames in 1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100
do
	python numpy.py $num_frames
	echo "Done with " $num_frames
done
