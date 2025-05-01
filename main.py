import math

import cv2
import numpy as np


def q1():
    # part a
    # Load the input image
    img = cv2.imread('data\circle3.jpg')
    centroid_img = cv2.imread('data\circle3.jpg')

    # Convert the image to grayscale
    gray = cv2.cvtColor(centroid_img, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours and compute their centroids
    for cnt in contours:
        # Compute the bounding box of the contour
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2

        # Draw the centroid in red color
        cv2.circle(centroid_img, (cx, cy), 5, (0, 0, 255), -1)

    # Display the result
    cv2.imshow('Centroids', centroid_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # part b
    # Create a mask to store all red pixels
    red_mask = np.zeros_like(gray)

    # Loop over the contours and compute their centroids
    for cnt in contours:
        # Compute the area and perimeter of the contour
        area = cv2.contourArea(cnt)

        # Avoid division by zero by checking if the area of the contour is non-zero
        if area != 0:
            # Compute the centroid coordinates
            cx = 0
            cy = 0
            for pt in cnt:
                cx += pt[0][0]
                cy += pt[0][1]
            cx = int(cx / len(cnt))
            cy = int(cy / len(cnt))

            # If the pixel at the centroid is red, store it in the red_mask
            if centroid_img[cy, cx][2] > 0 and centroid_img[cy, cx][1] == 0 and centroid_img[cy, cx][0] == 0:
                red_mask[cy, cx] = 255

    # Perform morphological dilation using the red mask as marker points
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(red_mask, kernel, iterations=12)

    # Merge the dilated mask and the original image
    result = cv2.bitwise_or(img, cv2.cvtColor(dilated_mask, cv2.COLOR_GRAY2BGR))

    # Show the results
    # cv2.imshow('Original Image', img)
    # cv2.imshow('Dilation mask', dilated_mask)
    cv2.imshow('Holes Filled', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # part c
    # Find the contour with the biggest hole
    biggest_hole_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the contour
    mask = cv2.drawContours(np.zeros_like(gray), [biggest_hole_contour], 0, 255, -1)

    # Get the area of the black connected component inside the contour
    black_area = cv2.countNonZero(mask) - cv2.countNonZero(gray & mask)

    # cv2.imshow('Holes Filled', gray & mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    diameter = ((black_area / math.pi) ** 1/2) * 2

    # Display the result
    print('diameter of largest contour:', diameter)

    # part d
    # Load the input image
    img_d = cv2.imread('data\circles.jpg', 0)

    # Threshold the image
    ret, thresh_d = cv2.threshold(img_d, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh_d, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Count the number of contours with no black pixels inside them
    count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = gray[y:y + h, x:x + w]
        if cv2.countNonZero(roi) > 8500:
            count += 1

    # Print the count of contours with no black pixels inside them
    print('Objects with holes', count)
    print('Objects without holes', 4 - count)


def q2():
    # Load the thermal image
    img = cv2.imread('data\\thermal.jpg', 0)

    # Apply thresholding to obtain binary image
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    # Perform morphological operations to remove noise and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours and compute their bounding boxes
    for cnt in contours:
        # Compute area of contour
        area = cv2.contourArea(cnt)

        # Set maximum area
        max_area = 950

        # Draw bounding box around the person if contour area is less than max_area
        if area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('People Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def q3():
    # part a
    # Read in the image and convert it to grayscale
    img = cv2.imread('data\\text.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Apply a morphological transformation to the image to close gaps in the text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find the contours of the connected components in the image
    contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter the contours based on their size to keep only those that are likely to represent text lines
    min_line_length = 405
    max_line_length = 1500
    filtered_contours = [cnt for cnt in contours if min_line_length < cv2.arcLength(cnt, True) < max_line_length]

    # Count the number of remaining contours, which corresponds to the number of text lines in the image
    num_lines = len(filtered_contours)
    print("Number of text lines in the image: ", num_lines)

    # part b
    # Find contours in the text image
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out the contours that are unlikely to represent words
    word_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h
        area = cv2.contourArea(cnt)
        if aspect_ratio > 0.2 and area > 120 and h > 15:
            word_contours.append(cnt)

    # Count the number of remaining contours, which corresponds to the number of words in the image
    num_words = len(word_contours)
    print("Number of words in the image: ", num_words)


def q4():
    # Load image
    img = cv2.imread('data\\soduku.jpg')

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold image
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_size = 10

    # Loop over contours
    for i in range(len(contours)):
        cnt = contours[i]
        hierarchy_level = hierarchy[0][i][3]

        # Approximate contour to polygon
        approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), False)

        # Check if polygon has four sides (i.e. is a square)
        if len(approx) == 4 and all(cv2.norm(approx[i] - approx[(i + 1) % 4]) >= min_size for i in range(4)):
            if hierarchy_level > 0:
                # Check if square is empty (has no black pixels)
                x, y, w, h = cv2.boundingRect(approx)
                square = gray[y:y + h, x:x + w]
                if cv2.countNonZero(square) == square.size:
                    # Draw red filled square on image
                    cv2.drawContours(img, [approx], 0, (0, 0, 255), cv2.FILLED)

    # Display image
    cv2.imshow('Output', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def q5():
    # part a
    # Load the image in grayscale
    img = cv2.imread('data\\alphabets-urdu.jpg', 0)

    # Apply binary thresholding to the image
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Set minimum height threshold
    min_height = 20

    # Loop through each contour and check if its width exceeds its height
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > h > min_height:
            # Draw a rectangle around the contour
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Characters where width > height', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # part b
    # Set threshold for minimum and maximum aspect ratio
    min_aspect_ratio = 0.98
    max_aspect_ratio = 1.03

    # Loop through each contour and check if its aspect ratio is within the threshold range
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio and h > min_height:
            # Draw a rectangle around the contour
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Alphabets where width == height', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # part c
    # Set minimum and maximum area thresholds for nukta detection
    min_nukta_area = 10
    max_nukta_area = 100

    # Set minimum and maximum black to white pixel ratios for nukta detection
    min_bw_ratio = 1
    max_bw_ratio = 0.8

    # Loop through each contour and check if it has a nukta
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_nukta_area <= area <= max_nukta_area:
            x, y, w, h = cv2.boundingRect(contour)
            nukta_roi = thresh[y:y + h, x:x + w]
            bw_ratio = cv2.countNonZero(nukta_roi) / float(nukta_roi.size)
            if bw_ratio <= max_bw_ratio:
                # Draw a rectangle around the nukta
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Nuktas', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # part e
    # Set minimum and maximum area thresholds for hole detection
    min_hole_area = 25
    max_hole_area = 200

    # Loop through each contour and check if it has holes
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_hole_area <= area <= max_hole_area:
            # Check if the contour has holes
            _, holes = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if holes is not None and len(holes) > 0:
                # Draw a rectangle around the contour
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def q6():
    img = cv2.imread('data\\bubble.jpg', 0)  # 0 flag loads image in grayscale

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply Otsu's thresholding
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow('Bubble Sheet', thresh)

    # Define the number of rows and columns
    rows = 5
    cols = 5

    # Calculate the dimensions of each sub-image
    height, width = thresh.shape[:2]
    row_height = height // rows
    col_width = width // cols

    # Loop through each row and find the column with the highest number of black pixels
    for row in range(rows):
        # Initialize variables to keep track of the maximum number of black pixels and the column index
        max_black_pixels = 0
        max_col = 0

        for col in range(cols):
            # Define the coordinates of the sub-image
            x1, y1 = col * col_width, row * row_height
            x2, y2 = x1 + col_width, y1 + row_height

            # Crop the sub-image and count the number of black pixels
            sub_img = thresh[y1:y2, x1:x2]
            black_pixels = np.sum(sub_img == 0)

            # Update the maximum number of black pixels and the column index if necessary
            if black_pixels > max_black_pixels:
                max_black_pixels = black_pixels
                max_col = col

        if max_col == 1:
            r = row + 1
            print(str(r) + ":A")
        if max_col == 2:
            r = row + 1
            print(str(r) + ":B")
        if max_col == 3:
            r = row + 1
            print(str(r) + ":C")
        if max_col == 4:
            r = row + 1
            print(str(r) + ":D")


# q1()
# q2()
# q3()
# q4()
# q5()
q6()