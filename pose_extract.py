import numpy as np
import matplotlib.pyplot as plt
import cv2

# Nose = 1
# Right_Eye = 3
# Left_Eye = 2
# Right_Cheek = 5
# Left_Cheek = 4
# Right_Hand = 11
# Left_Hand = 10
# Right_Elbow = 9
# Left_Elbow = 8
# Right_Shuolder = 7
# Left_Shoulder = 6
# Right_Hip = 13
# Left_Hip = 12
# Right_Knee = 15
# Left_Knee = 14
# Right_Foot = 17
# Left_Foot = 16

connections = [
    (1, 2), (1, 3), (3, 5), (2, 4), (5, 7), (4, 6), (7, 9), (6, 8), (9, 11),
    (8, 10), (7, 6), (7, 13), (6, 12), (13, 15), (15, 17), (12, 14), (14, 16),
    (12, 13)
]

def Extrac_Pose(all_keypoints , frame):

    black_frame = np.zeros_like(frame)

    for person_keypoints in all_keypoints:
        
        kp = person_keypoints.xy.cpu().numpy()
        kp = np.reshape(kp, (-1, 2))  # Reshaping to 2D array

        for point in kp:
            if not np.array_equal(point, [0, 0]):  # Skip (0,0) points
                cv2.circle(black_frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Green dots

        for start, end in connections:
            start_point = (int(kp[start-1][0]), int(kp[start-1][1]))
            end_point = (int(kp[end-1][0]), int(kp[end-1][1]))
            
            # Draw line only if neither of the points is (0,0)
            if not np.array_equal(start_point, [0, 0]) and not np.array_equal(end_point, [0, 0]):
                cv2.line(black_frame, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])), (0, 255, 0), 2)  # Green lines

        return black_frame