import argparse
import math
import os
import sys
import time
import sys

import cv2
cv = cv2
import matplotlib.pyplot as plt

import numpy as np

sys.path.append(".") # if running from parent dir
from src.cv_helpers import *
from src.hands import detect_hands

print("Your openCV version:", cv.__version__)
print("It might not work if it isnt 4.5.x")

def blur(vid):
    blurred = [cv.GaussianBlur(i, (5,5), 2) for i in vid]
    return blurred

def bg_subtract(vid, threshold=50):
    # bgsub = cv.createBackgroundSubtractorKNN(detectShadows=False)
    # masks = [bgsub.apply(i) for i in vid]
    vid = np.array(vid) / 255
    bg = np.mean(vid, 0)
    # print(bg, bg.shape)
    mask = np.any(np.any(np.logical_or(vid > bg+threshold/255, vid < bg-threshold/255), axis=-1), axis=0)
    mask = np.float32(mask)
    kern = cv.getStructuringElement(cv.MORPH_CROSS, (5,5))
    mask = cv.dilate(mask, kern)
    showim(bg, ms=3000)
    showim(mask, ms=3000)
    exit()

def edge_process(vid, edge_threshold=70, show_result=False):
    """
    grayscale and blur, and find edges
    """
    print("Finding edges")
    gray = [cv.cvtColor(i, cv.COLOR_BGR2GRAY) for i in vid]

    """
    edge detection. the second number is the main threshold (lower => more edges, more noise)
    """
    edges = [cv.Canny(i, 10, edge_threshold) for i in gray]
    # edges = [cv.dilate(i, np.ones((3,3),np.uint8),iterations=1) for i in edges]
    # edges = [cv.erode(i, np.ones((5,5),np.uint8),iterations=1) for i in edges]

    if show_result:
        showvid(edges)

    return edges

def draw_lines(lines, frame):
    """
    helper to draw a set of lines on a frame
    """
    for i in range(len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
        pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
        cv.line(frame, pt1, pt2, (255,255,255), 3, cv.LINE_AA)

def find_lines(edges, orig_vid, show_result=False):
    """
    find lines from the edges
    args:
        orig_vid: video that is copied, then the lines drawn on and showed
    returns:
        linesvid: black video with white lines drawn on
        lines
    """
    print("Finding horizontal lines")

    def horizontal_ish(angle):
        """
        filter angles within 45deg of horizontal
        """
        return (np.pi*1/4 < angle < np.pi*3/4)

    def hough(im):
        lines = cv.HoughLines(im, 
                    rho=3, 
                    theta=np.pi/180, 
                    threshold=300)
        lines = [] if lines is None else lines
        # remove mostly vertical lines and take the first 150; they should be returned in order of confidence
        lines = [j for j in lines if horizontal_ish(j[0][1])][:150]
        return lines

    lines = [hough(i) for i in edges]

    def findparallel(lines, theta_threshold=3, rho_threshold=200, maxn=None):
        """
        args:
            theta_threshold: degrees, the threshold for how close in angle lines 
                have to be to be parallel
            rho_threshold: pixels, distance to be within
            maxn: number of top line bundles to find in each frame
        returns:
            for each frame, a set of bundles of lines. each bundle is parallel
        """
        theta_threshold = np.pi/180 * theta_threshold
        vid_lines = []
        for frame_lines in lines:
            all_lines = []
            for i, l1 in enumerate(frame_lines):
                this_lines = [l1]
                for j, l2 in enumerate(frame_lines):
                    if (i == j):
                        continue
                    if (abs(l1[0][1] - l2[0][1]) <= theta_threshold) and \
                        (abs(l1[0][0] - l2[0][0]) <= rho_threshold):
                        this_lines.append(l2)
                all_lines.append(this_lines)
            # sort by number of lines found in each bundle, take top maxn
            sorted(all_lines, key=lambda x: len(x), reverse=True)
            if maxn is not None:
                all_lines = all_lines[:maxn]
            # joined = []
            # for x in all_lines:
            #     joined += x
            vid_lines.append(all_lines)
        return vid_lines

    line_bundles = findparallel(lines, maxn=2)

    if show_result:
        orig_vid = [np.copy(i) for i in orig_vid]
    linesvid = [np.zeros_like(i) for i in orig_vid]

    for i,framelines in enumerate(line_bundles):
        """
        write the lines onto a black video
        """
        for bundle in framelines:
            if show_result:
                draw_lines(bundle, orig_vid[i])
            draw_lines(bundle, linesvid[i])
            # showim(linesvid[i],name=str(i))

    linesvid = [np.uint8(cv.cvtColor(np.float32(i), cv.COLOR_BGR2GRAY)) for i in linesvid]

    if show_result:
        showvid(orig_vid, ms=100)
        showvid(linesvid, ms=100)
    # now with lines drawn on
    return linesvid, lines


def find_contours(linesvid, vid, show_result=False):
    """
    find the contours of the linesvid. It should find a big bounding box around the big
    cluster of the fretboard, so we take the contour with the biggest area
    args:
        linesvid: vid to find contours on (should be black, with white lines)
        vid: vid to copy and draw bounding box on
    returns:
        bounding boxes
    """
    print("Finding bounding boxes")

    if show_result:
        boxes_vid = [np.copy(i) for i in vid]

    boxes = []
    for i, frame in enumerate(linesvid):
        contours, heirarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # get largest contour
        cnt = max(contours, key=cv.contourArea)
        # find its (potentially nonvertical) bounding rect
        rect = cv2.minAreaRect(cnt)
        box = cv.boxPoints(rect).astype(int)
        if show_result:
            im = boxes_vid[i]
            cv2.drawContours(im,[box],0,(0,0,255),2)
        box = order_points(box)
        boxes.append(box)
        if show_result:
            cs = [(255,0,0), (0,255,0), (0,0,255), (255,255,255)]
            for i in range(4):
                cv.circle(im, tuple(box[i]), 5, cs[i], -1, cv.FILLED)

    if show_result:
        showvid(boxes_vid, "bounding boxes", ms=100)
    
    return boxes


def smooth_bounding_boxes(boxes, orig_vid, show_result=True):
    """
    smoothing
    args:
        boxes: bounding box for each frame
        orig_vid: copied vid to sho result on
    returns:
        bounding boxes
    """
    print("Smoothing bounding boxes")

    # span is number of frames on each side to average the bounding boxes of
    span = 3
    # num_outliers number of highest and lowest to remove before averaging
    num_outliers = 0
    smoothed_boxes = []

    if show_result:
        boxes_vid = [np.copy(i) for i in orig_vid]

    assert len(boxes) >= (2 * span + 1)
    for i, box in enumerate(boxes):
        # get a selection of 2*span+1 consecutive boxes, and repeat some when at the edges of the video
        top = i + span + 1
        bottom = i-span
        if bottom < 0:
            bottom = 0
        selection = boxes[bottom:top]
        if i < span:
            selection = np.concatenate((selection, boxes[:(span-i)]), axis=0)
        elif i >= (len(boxes) - span):
            i2 = len(boxes) - i - 1
            selection = np.concatenate((selection, boxes[-(span - i2):]), axis=0)

        # remove highest and lowest n
        # if num_outliers > 0:
        #     newboxes = []
        #     for box in np.array(selection).T:
        #         new = []
        #         for row in box:
        #             for _ in range(num_outliers):
        #                 row = np.delete(row, row.argmin())
        #                 row = np.delete(row, row.argmax())
        #             new.append(row)
        #         newboxes.append(new)
        #     selection = np.array(newboxes).T
            
        # average the rest
        newbox = np.mean(selection, axis=0).astype(int)
        smoothed_boxes.append(newbox)
        if show_result:
            cv2.drawContours(boxes_vid[i],[newbox],0,(0,0,255),2)

    if show_result:
        showvid(boxes_vid, name="smoothed", ms=100)

    return smoothed_boxes


def angles_from_boxes(boxes, show_result=False):
    """
    get approx rotation angel from smoothed bounding boxes
    """
    print("getting rotation angles")

    angles = []
    for box in boxes:

        tl, tr, br, bl = box
        topangle = -np.arctan2(*(tr - tl)) + np.pi
        bottangle = -np.arctan2(*(br - bl)) + np.pi
        angle = (topangle + bottangle) / 2

        angles.append(angle)

        # im = np.zeros((600, 1400, 3))
        # draw_lines([[[300, angle]]], im)
        # showim(im)
    
    return angles


def rotate_frames(boxes, angles, vid, show_result=False):
    """
    rotate the frames so that the fretboard is horizontal, and crop to the same size
    args:
        boxes
        lines
        vid: orig vid to copy and modify
    returns:
        rotated and cropped vid
        (top, bottom) y positions to crop to
    """
    print("Rotating and cropping to fretboard")

    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def rotate_points(pts, p0, a):
        """
        rotate a set of points, like a contour, around p0
        returns:
            lists of xs, ys
        """
        x1, y1 = pts[:,0], pts[:,1]
        x0, y0 = p0
        x2 = ((x1 - x0) * np.cos(a)) - ((y1 - y0) * np.sin(a)) + x0
        y2 = ((x1 - x0) * np.sin(a)) + ((y1 - y0) * np.cos(a)) + y0
        return np.int16(x2), np.int16(y2)

    center = (vid[0].shape[1]//2, vid[0].shape[0]//2)
    rotated_vid = []
    rot_bounds = []
    for i, frame in enumerate(vid):
        radians = angles[i] - (np.pi * 1/2)
        degrees = np.degrees(radians)
        # draw_lines([[[300, radians + (np.pi * 1/2)]]], frame)

        # showim(mask)
        rot_frame = rotate_image(frame, degrees)
        rotated_vid.append(rot_frame)

        # ys,xs,z = np.nonzero(rot_mask)
        # if len(np.squeeze(y)) == 0:
        #     # bad frame, no box found
        #     continue
        xs, ys = rotate_points(boxes[i], center, -radians)
        topy = max(0, np.min(ys))
        # topx = max(0, np.min(xs))
        bottomy = np.max(ys)
        # bottomx = np.max(xs)
        bounds = (topy, bottomy+1)
        rot_bounds.append(bounds)
        # rot_frame = rot_frame[topy:bottomy+1, topx:bottomx+1]
        # NOTE extend the bottom, since it tends to miss the smaller strings more often?

    if show_result:
        showvid(rotated_vid, name="fretboard", ms=200)

    return rotated_vid, rot_bounds


def normalize_shape(rotated_vid, rotated_bounds, handspos, target_ratio, show_result=False):
    """
    make all frames the same shape
    args:
        rotated vid
        rotated bounds: rotated bounding boxes
        handspos: list of [[x1,y1],...] positions of hands, for each frame (or None when missing)
        target ratio: ratio of height/width
    """
    print("Normalizing frames to consistent shape")

    def pad_to_target(frame, target_y, target_x=None):
        top = target_y - frame.shape[0]
        bottom = (top + 1) // 2
        top = top // 2
        if target_x is None:
            left = 0
            right = 0
        else:
            left = target_x - frame.shape[1]
            right = (left + 1) // 2
            left = left // 2
        return cv.copyMakeBorder(frame, top, bottom, left, right, cv.BORDER_CONSTANT, 0)

    rotated_bounds = np.array(rotated_bounds)
    topys = rotated_bounds[...,0]
    bottomys = rotated_bounds[...,1]
    # topxs = rotated_bounds[:,2]
    # bottomxs = rotated_bounds[:,3]

    height = np.int16(np.median(bottomys - topys))
    # this also ensures the dimensions are even numbers, which is required when we are writing the video out
    hspan = height // 2
    # y midpoints, measured from the bottom, as it tends to miss the bottom a little more
    mid_ys = bottomys - hspan

    fretboard_vid = [frame[mid_ys[i]-hspan:mid_ys[i]+hspan] for i,frame in enumerate(rotated_vid)]
    
    height, width, z = fretboard_vid[0].shape

    if (height / width) < 0.98 * target_ratio:
        print("Cropping width")
        # too long and skinny, crop in x dimension
        target_width_span = int((height / target_ratio) // 2)
        min_x = target_width_span
        max_x = width - target_width_span
        curr_handx = None
        hand_xs = []
        for i, pts in enumerate(handspos):
            if pts is not None:
                # for each possible hand position detected, keep the rightmost one that is within the bounds of the fretboard
                good_xs = [x for (x,y) in pts if abs(y - mid_ys[i]) < (hspan * 1.2)]
                if good_xs:
                    # 0.1 factor is to focus toward the nut side of the hand, if possible
                    this_handx = max(good_xs) + int(0.1 * width)
                    # clip to min/max bounds, then make it our new most recent x position
                    curr_handx = max(min_x, min(max_x, this_handx))
            hand_xs.append(curr_handx)
        
        # replace any initial Nones recursively
        def replace_nulls(i):
            if hand_xs[i] is None:
                hand_xs[i] = replace_nulls(i+1)
            return hand_xs[i]
        replace_nulls(0)

        fretboard_vid = [ 
            frame[:, hand_xs[i]-target_width_span:hand_xs[i]+target_width_span] for i,frame in enumerate(fretboard_vid)
        ]
    elif (height / width) > 1.02 * target_ratio:
        # pad in y direction
        print("Padding height")
        target_height = int(width * 0.3)
        fretboard_vid = [pad_to_target(frame, target_height) for frame in fretboard_vid]


    if show_result:
        showim(fretboard_vid[0], ms=2000)
        showvid(fretboard_vid)

    print("normalized shape:", fretboard_vid[0].shape)

    return fretboard_vid


def find_vert_lines(edges, fretboard_vid, prob_lines=False, show_result=False):
    """
    find vertical lines that should correspond to frets
    args:
        edges: fretboard edges
        fretboard_vid
        prob_lines: whether to do probabalistic HoughLinesP (true) or regular HoughLines (false)
    returns:
        line positions: x coords for all lines in each frame
    """
    print("Finding vertical lines")

    def vertical(angle, threshhold=3):
        """
        find vertical lines within <threshhold> degrees
        """
        # convert to radians
        threshhold = np.pi/180 * threshhold
        # find vertical lines (either "up" or "down")
        if (-threshhold < angle < threshhold) or (np.pi-threshhold < angle < np.pi+threshhold):
            return True
        return False
        
    def vertical_p(x1, y1, x2, y2, *, threshhold=3):
        """
        find vertical lines within <threshhold> degrees, from their endpoints 
        """
        # convert to vertical slope (x/y)
        threshhold = abs(1 / math.tan(threshhold))
        # find vertical lines (either up or down)
        slope = abs(x1 - x2) / max(abs(y1 - y2), 0.00001)
        if slope < threshhold:
            return True
        return False

    top_n = 150

    linesvid = [np.copy(i) for i in fretboard_vid]

    if prob_lines:
        lines = [cv.HoughLinesP(i, 1, np.pi/180, 10, None, 50, 10) for i in edges]
        lines = [i if i is not None else [] for i in lines]
        # find vertical lines, take first top_n (as the lines are returned in order of confidence)
        lines = [[j for j in framelines if vertical_p(*j[0])][:top_n] for framelines in lines]

        if show_result:
            for idx, framelines in enumerate(lines):
                for i in range(len(framelines)):
                    l = framelines[i][0]
                    cv.line(linesvid[idx], (l[0], l[1]), (l[2], l[3]), (255,255,255), 3, cv.LINE_AA)

    else:
        lines = [cv.HoughLines(i, 1, np.pi/180, 25, None, 0, 0) for i in edges]
        lines = [i if i is not None else [] for i in lines]
        # find vertical lines, take first top_n (as the lines are returned in order of confidence)
        # lines = [[j for j in framelines if vertical(j[0][1])][:top_n] for framelines in lines]

        if show_result:
            for idx, frame_lines in enumerate(lines):
                for i in range(len(frame_lines)):
                    rho = frame_lines[i][0][0]
                    theta = frame_lines[i][0][1]
                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
                    pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
                    cv.line(linesvid[idx], pt1, pt2, (255,255,255), 3, cv.LINE_AA)

    if show_result:
        showvid(linesvid, "vert_lines")

    if prob_lines:
        # get horiz position, by averaging x coordinates of endpoints
        line_pos = [np.unique(np.array([(j[0][0]+j[0][2])/2 for j in framelines], dtype=int)) for framelines in lines]
    else:
        # get position of lines horizontally (since they must be approximately vertical, the 
        #  rho should be approximately the horizontal distance). np.unique also sorts
        line_pos = [np.unique(np.array([j[0][0] for j in framelines], dtype=int)) for framelines in lines]

    return line_pos

def match_frets(line_pos, fretboard_vid, show_result=False):
    """
    match vertical lines to fret spacing
    """
    print("Finding frets")

    width = fretboard_vid[0].shape[1]

    # factor to be within to be considered a fret match
    fret_close_factor = 0.08
    # min dist to be considered a match, in pixels
    min_fret_close = 3/1000 * width

    def is_close(real, target, fret_len):
        """
        determine if a line should be considered a match for this fret
        """
        within = np.maximum(min_fret_close, fret_len*fret_close_factor)
        return np.abs(real - target) <= within

    # in pixels. fret distances less than or equal to this num pixels will be ignored
    min_fret = 0.018 * width
    # frets greater than this are ignored
    max_fret = width / 10
    print("min fret", min_fret, "max_fret", max_fret)

    # number of consecutive misses after which to stop searching
    max_miss = 4

    # ratio of fret length to length of next fret, left (higher) and right 
    #  (lower) on the neck
    L_RATIO = 0.943874312682
    R_RATIO = 1.05946309436

    def match_frame_frets(line_pos):
        """
        for each pair of lines positions, use the distance between them as a possible fret
        length, and then extend out to either side and see how many other lines would match
        that fret pattern
        """
        matches = []
        # bestij = None
        for i in range(len(line_pos) - 1):
            imatches = []
            # bestj = None
            for j in range(i+1, len(line_pos)):
                jmatches = []
                r = line_pos[j]
                l = line_pos[i]
                d = r - l
                if d <= min_fret:
                    continue
                if d >= max_fret:
                    break # end inner loop, go to next outer loop interation

                # left = higher on the neck
                if i > 0:
                    dl = d
                    l_arr = line_pos[:i]
                    def ok_idx(n):
                        return min(n, len(l_arr)-1)
                    misscounter = 0
                    while misscounter < max_miss:
                        dl *= L_RATIO
                        l -= dl
                        if l < 0:
                            break
                        idx = ok_idx(np.searchsorted(l_arr, l, side="left"))
                        val = l_arr[idx]
                        if is_close(val, l, dl):
                            misscounter = 0
                            jmatches.append(val)
                        else:
                            misscounter += 1

                # matches so far are in reverse sorted order            
                jmatches.reverse()
                # add reference fret
                jmatches.append(line_pos[i])
                jmatches.append(line_pos[j])

                # right = lower on the neck
                if j+1 < len(line_pos)-1:
                    dr = d
                    r_arr = line_pos[j+1:]
                    def ok_idx(n):
                        return min(n, len(r_arr)-1)
                    misscounter = 0
                    while misscounter < max_miss and r < line_pos[-1]:
                        dr *= R_RATIO
                        r += dr
                        idx = ok_idx(np.searchsorted(r_arr, r, side="left"))
                        val = r_arr[idx]
                        if is_close(val, r, dr):
                            misscounter = 0
                            jmatches.append(val)
                        else:
                            misscounter += 1
                
                if len(jmatches) > len(imatches):
                    imatches = jmatches
                    # bestj = line_pos[j]
            
            if len(imatches) >= len(matches):
                matches = imatches
                # bestij = (line_pos[i], bestj)
        
        return matches


    def match_frame_frets_v2(line_pos):
        """
        quicker (about 2x faster) vectorized version, but more prone to match 
        incorrect frets out past the fretboard
        """
        matches = []
        # bestij = None
        for i in range(len(line_pos) - 1):
            imatches = []
            # bestj = None
            for j in range(i+1, len(line_pos)):
                r = line_pos[j]
                l = line_pos[i]
                d = r - l

                targets = [l, r]
                dl = d
                if dl <= min_fret or dl >= max_fret:
                    continue
                while l > 0 and dl > min_fret:
                    dl *= L_RATIO
                    l -= dl
                    targets.insert(0, l)
                dr = d
                while r <= line_pos[-1] and dr < max_fret:
                    dr *= R_RATIO
                    r += dr
                    targets.append(r)

                targets = np.array(targets)
                fret_lens = np.diff(targets)
                fret_lens = np.append(fret_lens, fret_lens[-1])

                # find the indexes where 'targets' would be inserted in the array.
                # the closest value to each target must be either the value there or
                # the value at the index right before
                indsR = np.searchsorted(line_pos, targets)
                indsL = indsR - 1
                indsR = np.minimum(indsR, len(line_pos)-1)
                indsL = np.maximum(indsL, 0)

                is_fret = np.logical_or(
                            is_close(line_pos[indsL], targets, fret_lens),
                            is_close(line_pos[indsR], targets, fret_lens)
                        )
                
                jmatches = targets[is_fret]
                if len(jmatches) > len(imatches):
                    imatches = jmatches

            if len(imatches) >= len(matches):
                matches = imatches

        return np.int16(matches)


    t = time.time()
    matches = [match_frame_frets_v2(i) for i in line_pos]
    print("fret match speed per frame:", (time.time() - t) / len(line_pos))

    print("fret matches in each frame:", [len(i) for i in matches])


    fretvid = [np.copy(i) for i in fretboard_vid]

    for i, frame in enumerate(fretvid):
        for x in matches[i]:
            cv.line(frame, (x,0), (x,2000), (0,0,255), 3)

    if show_result:
        showvid(fretvid, ms=200)

    return matches



class MyArgs:

    def __init__(self, kwargs_dict):
        for k,v in kwargs_dict.items():
            setattr(self, k, v)



def timer(name="", _cache=[], clear=False):
    if clear:
        return _cache.clear()
    if not _cache:
        _cache.append(time.time())
    else:
        t = time.time()
        print(" ", round(t - _cache[0], 3), "sec", name)
        _cache[0] = t


def main(**kwargs):
    """
    args (either to call, or supplied in command line):
        file: filename to read in
        outfile: filename to save rotated and cropped fretboard vid
        full: whether to process the full video
        nofrets: whether to not do fret processing
        show: whether to show videos during processing
    """

    if kwargs:
        args = MyArgs(kwargs)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--file",default="acoustic_light_short.mov")
        parser.add_argument("--outfile",default="fretboard_rotated",help="place to store rotated fretboard vid")
        parser.add_argument("--full",action="store_true",default=False)
        parser.add_argument("--nofrets",action="store_true",default=False)
        parser.add_argument("--show",action="store_true",default=False)
        args = parser.parse_args()

    maxframes = None if args.full else 30
    vid = readvid(args.file, maxframes=maxframes)

    print(len(vid), "frames,", vid[0].shape)
    start = time.time()

    if False and args.show:
        showvid(vid)

    # process in batches of 500, cuz allocating a ton of frames all at once is too expensive
    timer()
    full_rotated = []
    full_bounds = []
    full_hands = []
    for i in range(0, len(vid), 500):
        print("\nFrames", i, "through", i+500)
        batch = vid[i:i+500]
        blurred = blur(batch)
        # bg = bg_subtract(blurred)
        edges = edge_process(blurred, show_result=True and args.show)
        timer()
        linesvid, lines = find_lines(edges, batch, show_result=True and args.show)
        timer()
        boxes = find_contours(linesvid, batch, show_result=True and args.show)
        timer()
        boxes = smooth_bounding_boxes(boxes, batch, show_result=True and args.show)
        timer()
        angles = angles_from_boxes(boxes, show_result=True)
        timer()
        rotated_vid, fretboard_bounds = rotate_frames(boxes, angles, batch, show_result=True and args.show)
        full_rotated += rotated_vid
        full_bounds += fretboard_bounds
        timer()
        handbounds = detect_hands(rotated_vid, show_result=True and args.show)
        full_hands += handbounds
        timer()

    # shape has to be the same for the whole video, so can't do it in batches
    # target ratio is height/width ratio
    TARGET_RATIO = 0.2
    fretboard_vid = normalize_shape(full_rotated, full_bounds, full_hands, 
                        target_ratio=TARGET_RATIO, show_result=True and args.show)
    timer()


    os.makedirs("data", exist_ok=True)
    writevid(fretboard_vid, args.outfile)
    timer()

    if not args.nofrets:
        for i in range(0, len(fretboard_vid), 500):
            fretboard_batch = fretboard_vid[i:i+500]
            blurred = blur(fretboard_batch)
            fretboard_edges = edge_process(blurred, edge_threshold=50,
                                show_result=True and args.show)
            timer()
            line_positions = find_vert_lines(fretboard_edges, fretboard_batch, 
                                prob_lines=True, show_result=False and args.show)
            timer()
            matches = match_frets(line_positions, fretboard_batch, 
                        show_result=True and args.show)
            timer()

    print(round((time.time() - start) / 60, 2), "minutes elapsed")
    timer(clear=True)


if __name__ == "__main__":
    main()
