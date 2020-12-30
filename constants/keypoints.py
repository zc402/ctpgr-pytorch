
"""
Keypoints in AIChallenger dataset:
1: Right shoulder
2: Right elbow
3: Right wrist

4: Left shoulder
5: Left elbow
6: Left wrist

7: Right hip
8: Right knee
9: Right ankle

10: Left hip
11: Left knee
12: Left ankle

13: head top
14: neck

"""

# Keypoint connection of bones in AIChallenger dataset:
aic_bones = [
    [1, 2],  # 右大臂
    [2, 3],  # 右小臂

    [4, 5],  # 左大臂
    [5, 6],  # 左小臂

    [14, 1],  # 右肩
    [14, 4],  # 左肩

    [1, 7],  # 右侧躯干
    [4, 10],  # 左侧躯干

    [7, 8],  # 右大腿
    [8, 9],  # 右小腿

    [10, 11],  # 左大腿
    [11, 12],  # 左小腿

    [13, 14]]  # 头

aic_bone_pairs = (
    ([14, 1], [1, 2]),  # 右肩与右大臂
    ([1, 2], [2, 3]),  # 右大臂小臂
    ([14, 4], [4, 5]),  # 左
    ([4, 5], [5, 6]),  # 左

    ([7, 8], [8, 9]),  # 右大小腿
    ([10, 11], [11, 12]),  # 左大小腿
)