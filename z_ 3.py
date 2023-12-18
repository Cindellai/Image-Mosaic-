import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ensure images are in the working directory or provide the full path.
image_path1 = '/Users/GirlsWhoCode/Documents/Imgs/uno.png'
image_path2 = '/Users/GirlsWhoCode/Documents/Imgs/dos.png'
image_path3 = '/Users/GirlsWhoCode/Documents/Imgs/IMG_7764.jpg'
image_path4 = '/Users/GirlsWhoCode/Documents/Imgs/IMG_7765.jpg'

image1 = cv2.imread(image_path1, cv2.IMREAD_COLOR)
image2 = cv2.imread(image_path2, cv2.IMREAD_COLOR)
image3 = cv2.imread(image_path3, cv2.IMREAD_COLOR)
image4 = cv2.imread(image_path4, cv2.IMREAD_COLOR)

# Convert images to grayscale for feature detection
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
gray4 = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)

#array of images 
images = [image1, image2]

def find_keypoints_and_descriptors(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

keypoints1, descriptors1 = find_keypoints_and_descriptors(gray1)
image_with_keypoints = cv2.drawKeypoints(image1, keypoints1, None)

keypoints2, descriptors2 = find_keypoints_and_descriptors(gray2)
image_with_keypoints2 = cv2.drawKeypoints(image2, keypoints2, None)

def match_keypoints(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    good_matches = []
    for m,n in knn_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)  # Note: not wrapping m in a list
    
    return good_matches

#Remove this from area
keypoints1, descriptors1 = find_keypoints_and_descriptors(gray1)
keypoints2, descriptors2 = find_keypoints_and_descriptors(gray2)
keypoints3, descriptors3 = find_keypoints_and_descriptors(gray3)
keypoints4, descriptors4 = find_keypoints_and_descriptors(gray4)

matches12 = match_keypoints(descriptors1, descriptors2)
matches23 = match_keypoints(descriptors2, descriptors3)
matches34 = match_keypoints(descriptors3, descriptors4)

def get_homography(keypoints1, keypoints2, good_matches):
    # No need to filter matches again, as they are already filtered by match_keypoints
    if len(good_matches) >= 4:
        # Extract location of good matches
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Compute homography using RANSAC
        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Check the number of inliers
        inliers = np.sum(status)
        print(f"{inliers} / {len(status)} inliers")

        # Only accept homography if enough inliers
        if inliers / len(status) > 0.5:  # You can adjust this threshold
            return H, status
        else:
            print("Not enough inliers to consider a good homography.")
            return None, None
    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), 4))
        return None, None

# Example homography matrix from matches between image1 and image2
matches12 = match_keypoints(descriptors1, descriptors2)
H12, status = get_homography(keypoints1, keypoints2, matches12)
H23 = get_homography(keypoints2, keypoints3, matches23)
H34 = get_homography(keypoints3, keypoints4, matches34)

if H12 is not None:
    print("Homography Matrix for Image 1 and 2:\n", H12)
else:
    print("Invalid Homography Matrix.")
    

def stitch_images(image1, image2, H):
    """
    Stitch two images using the provided homography matrix.
    """
    # Get dimensions of input images
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    # Points from image1's corners
    corners_image1 = np.array([
        [0, 0],
        [0, height1],
        [width1, height1],
        [width1, 0]
    ], dtype=np.float32).reshape(-1, 1, 2)

    # Transform corners of image1 to the panorama view
    corners_image1_transformed = cv2.perspectiveTransform(corners_image1, H)

    # Corners of image2
    corners_image2 = np.array([
        [0, 0],
        [0, height2],
        [width2, height2],
        [width2, 0]
    ], dtype=np.float32).reshape(-1, 1, 2)

    # Get the bounds of the area that will contain both images
    all_corners = np.concatenate((corners_image1_transformed, corners_image2), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())

    # Translation matrix to move the image back into the visible canvas area
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([
        [1, 0, translation_dist[0]],
        [0, 1, translation_dist[1]],
        [0, 0, 1]
    ])

    # Warp image1 using the translation matrix
    panorama_width = x_max - x_min
    panorama_height = y_max - y_min
    panorama = cv2.warpPerspective(image1, H_translation.dot(H), (panorama_width, panorama_height))

    # Translate image2 position based on minimum coordinates
    translation = (translation_dist[0], translation_dist[1])
    panorama[translation[1]:translation[1]+height2, translation[0]:translation[0]+width2] = image2

    return panorama

def create_panorama(images):
    """
    Stitch multiple images into a panorama.
    """
    # Start with the first image as the base for the panorama
    panorama = images[0]

    # Iterate through all images and stitch them to the panorama
    for i in range(1, len(images)):
        image = images[i]

        # Convert images to grayscale for feature detection
        gray_image1 = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find keypoints and descriptors
        keypoints1, descriptors1 = find_keypoints_and_descriptors(gray_image1)
        keypoints2, descriptors2 = find_keypoints_and_descriptors(gray_image2)

        # Match keypoints
        matches = match_keypoints(descriptors1, descriptors2)

        # Calculate homography
        H, _ = get_homography(keypoints1, keypoints2, matches)

        # Stitch the images using the homography matrix
        if H is not None:
            panorama = stitch_images(panorama, image, H)
        else:
            print("Homography could not be calculated for images", i-1, "and", i)
            return None

    return panorama

panorama = create_panorama(images)

# Check if the panorama creation was successful
if panorama is not None:
    # Display the resulting panorama in a new window
    cv2.imshow('Panoramic Image', panorama)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()  # Ensure all windows are closed when done
else:
    print("Failed to create the panorama.")
