import numpy as np
import time
import cv2
import pandas as pd
import argparse
cv = cv2

"""
command line args
"""
parser = argparse.ArgumentParser()
parser.add_argument("--file",default="chords.xlsx",help="chords excel file")
args = parser.parse_args()

"""
read and process file
"""
df = pd.read_excel(args.file, converters={"Name": str, "Tab": str})
# remove 'comment' lines and empty lines
df = df[~df['Name'].str.startswith("#", na=False)]
df = df.dropna(how='all')

if df.isnull().values.any():
    raise ValueError("NaN values encountered:\n\n" + df.to_string())

chords = df.to_numpy()

max_len = max([len(i) for i in chords[:,0]]) + 2

def print_chord(c):
    return "{:{max_len}} {}".format(*c, max_len=max_len)

print("available chords:")
for i in chords:
    print(print_chord(i))


"""
begin video capture
"""
cap = cv2.VideoCapture(0)

print("\nA chord will show, and then a countdown from 3. On 0, play the the chord. " 
      "Then, prepare for the next. Using a larger font size in the terminal (CMD PLUS "
      "on my Mac) is probably useful")
print("\nStarting 10 second pre-preparation time. Make sure the neck of the " 
      "guitar is entirely in the frame, and then make sure you can view the terminal")

time.sleep(2)

frame_ind = 0
frames = []
countdown = 0

start = time.time()
def elapsed():
    return time.time() - start

while(True):
    # Capture frame-by-frame
    ok, frame = cap.read()
    if not ok:
        print("frame failed, retrying...")
        continue

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) != -1:
        end = time.time()
        print("ending capture")
        break

    t = elapsed()
    if countdown > 10:
        print("Starting")
        countdown = 0
    elif t >= countdown:
        print(10 - countdown)
        countdown += 1
    
    
    
    




# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


