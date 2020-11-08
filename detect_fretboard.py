import argparse
import math
import os
import sys
import time

import cv2
cv = cv2
import matplotlib.pyplot as plt

import numpy as np

from cv_helpers import *



def edge_process(vid, edge_threshold=70, show_result=False):
    """
    grayscale and blur, and find edges
    """
    gray = [cv.cvtColor(i, cv.COLOR_BGR2GRAY) for i in vid]
    blurred = [cv.GaussianBlur(i, (5,5), 2) for i in gray]

    # thresh = [cv.adaptiveThreshold(i, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 23, 0) for i in blurred]
    # thresh_vid = [
    #     cv2.threshold(i, 28, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] for i in blurred
    # ]
    # showvid(thresh_vid)

    """
    edge detection. the second number is the main threshold (lower => more edges, more noise)
    """
    edges = [cv.Canny(i, 10, edge_threshold) for i in blurred]
    # edges = [cv.dilate(i, np.ones((3,3),np.uint8),iterations=1) for i in edges]
    # edges = [cv.erode(i, np.ones((5,5),np.uint8),iterations=1) for i in edges]

    if show_result:
        showvid(edges)

    return edges

def find_lines(edges, orig_vid, show_result=False):
    """
    find lines from the edges
    args:
        orig_vid: video that is copied, then the lines drawn on and showed
    returns:
        linesvid: black video with white lines drawn on
        lines
    """

    def horizontal_ish(angle):
        """
        filter angles within 45deg of horizontal
        """
        if (np.pi*1/4 < angle < np.pi*3/4) or (np.pi*1/4 < angle < np.pi*3/4):
            return True
        return False

    lines = [cv.HoughLines(i, 1, np.pi/180, 100, None, 0, 0) for i in edges]
    lines = [i if i is not None else [] for i in lines]

    # remove mostly vertical lines and take the first 500; they should be returned in order of confidence
    lines = [[j for j in framelines if horizontal_ish(j[0][1])][:500] for framelines in lines]

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

    def draw_lines(lines, frame):
        """
        draw a set of lines on a frame
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

    orig_vid = [np.copy(i) for i in orig_vid]
    linesvid = [np.zeros_like(i) for i in orig_vid]

    for i,framelines in enumerate(line_bundles):
        """
        write the lines onto a black video
        """
        for bundle in framelines:
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
    boxes_vid = [np.copy(i) for i in vid]

    boxes = []
    for i, frame in enumerate(linesvid):
        _, contours, heirarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        im = boxes_vid[i]
        # get largest contour
        cnt = max(contours, key=cv.contourArea)
        # find its (potentially nonvertical) bounding rect
        rect = cv2.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(im,[box],0,(0,0,255),2)
        boxes.append(box)

    if show_result:
        showvid(boxes_vid, "bounding boxes")
    
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
    # span is number of frames on each side to average the bounding boxes of
    span = 3
    # num_outliers number of highest and lowest to remove before averaging
    num_outliers = 0
    smoothed_boxes = []
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
        if num_outliers > 0:
            newboxes = []
            for box in np.array(selection).T:
                new = []
                for row in box:
                    for _ in range(num_outliers):
                        row = np.delete(row, row.argmin())
                        row = np.delete(row, row.argmax())
                    new.append(row)
                newboxes.append(new)
            selection = np.array(newboxes).T
            
        # average the rest
        newbox = np.mean(selection, axis=0)
        newbox = np.int0(newbox)
        smoothed_boxes.append(newbox)
        cv2.drawContours(boxes_vid[i],[newbox],0,(0,0,255),2)

    if show_result:
        showvid(boxes_vid, name="smoothed")

    return smoothed_boxes

def rot_and_crop(boxes, lines, vid, show_result=False):
    """
    rotate the frames so that the fretboard is horizontal, and crop to the same size
    args:
        boxes
        lines
        vid: orig vid to copy and modify
    returns:
        rotated and cropped vid
    """
    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    fretboard_vid = []
    for i, frame in enumerate(vid):
        radians = np.mean([l[0][1] for l in lines[i]])
        degrees = np.degrees(radians) - 90
        mask = np.zeros(frame.shape)
        cv2.drawContours(mask, [boxes[i]], 0, (255,255,255), -1)

        rot_frame = rotate_image(frame, degrees)
        rot_mask = rotate_image(mask, degrees)

        y,x,z = np.nonzero(rot_mask)
        if len(np.squeeze(y)) == 0:
            # bad frame, no box found
            continue
        (topy, topx) = (np.min(y), np.min(x))
        (bottomy, bottomx) = (np.max(y), np.max(x))
        rot_frame = rot_frame[topy:bottomy+1, topx:bottomx+1]
        # TODO extend the bottom, since it tends to miss the smaller strings more often?
        fretboard_vid.append(rot_frame)


    def pad_to_target(frame, target_x, target_y):
        top = target_y - frame.shape[0]
        left = target_x - frame.shape[1]
        bottom = (top + 1) // 2
        right = (left + 1) // 2
        top = top // 2
        left = left // 2
        return cv.copyMakeBorder(frame, top, bottom, left, right, cv.BORDER_CONSTANT, 0)

    max_y = max([i.shape[0] for i in fretboard_vid])
    max_x = max([i.shape[1] for i in fretboard_vid])
    fretboard_vid = [pad_to_target(i, max_x, max_y) for i in fretboard_vid]

    if show_result:
        showvid(fretboard_vid, name="fretboard", ms=200)

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

        for idx, framelines in enumerate(lines):
            for i in range(len(framelines)):
                l = framelines[i][0]
                cv.line(linesvid[idx], (l[0], l[1]), (l[2], l[3]), (255,255,255), 3, cv.LINE_AA)

    else:
        lines = [cv.HoughLines(i, 1, np.pi/180, 25, None, 0, 0) for i in edges]
        lines = [i if i is not None else [] for i in lines]
        # find vertical lines, take first top_n (as the lines are returned in order of confidence)
        lines = [[j for j in framelines if vertical(j[0][1])][:top_n] for framelines in lines]

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





def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--file",default="acoustic_light_short.mov")
    parser.add_argument("--outfile",default="fretboard_rotated",help="place to store rotated fretboard vid")
    parser.add_argument("--full",action="store_true",default=False)
    args = parser.parse_args()

    vid = readvid(args.file)
    if not args.full:
        vid = vid[:30]

    showvid(vid)

    edges = edge_process(vid, show_result=True)
    linesvid, lines = find_lines(edges, vid, show_result=True)
    boxes = find_contours(linesvid, vid, show_result=True)
    boxes = smooth_bounding_boxes(boxes, vid, show_result=True)
    fretboard_vid = rot_and_crop(boxes, lines, vid, show_result=True)

    os.makedirs("data", exist_ok=True)
    writevid(fretboard_vid, args.outfile)

    fretboard_edges = edge_process(fretboard_vid, edge_threshold=50)
    line_positions = find_vert_lines(fretboard_edges, fretboard_vid, prob_lines=False)
    matches = match_frets(line_positions, fretboard_vid)



if __name__ == "__main__":
    main()
