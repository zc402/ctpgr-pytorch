import numpy as np
import torch

from st_gcn.edges import edges


class AdjacencyMatrix:

    def __init__(self):
        num_nodes = 14
        # 每个关键点所处的高度{关键点index，高度}。关键点的index从1开始。
        self.heights = {9: 0, 12: 0, 8: 1, 11: 1, 7: 2, 10: 2, 3: 2, 6: 2, 2: 3, 5: 3, 1: 4, 4: 4, 14: 5, 13: 6}

        adjacency = np.zeros((num_nodes, num_nodes))  # 邻接矩阵
        edges_0based = np.array(edges, np.int) - 1
        for i, j in edges_0based:
            adjacency[j, i] = 1
            adjacency[i, j] = 1

        self.adjacency = adjacency

        # A相当于将邻接矩阵拆成3组，分别对应3个标签。3组加在一起后等于邻接矩阵。
        A = np.zeros((3, num_nodes, num_nodes))  # 3: number of labels (lower, equal, higher)
        for root in range(num_nodes):
            for j in range(num_nodes):
                if adjacency[root, j] == 1:
                    # ij相邻
                    hr = self.__height0b(root)  # 高度
                    hj = self.__height0b(j)
                    if hj - hr > 0:  # 邻接点在root之上
                        A[2, root, j] = 1
                    elif hj - hr < 0:
                        A[0, root, j] = 1
                    else:  # 高度一致。例如左右胯部。
                        A[1, root, j] = 1
        # 每个关键点与自身高度一致
        A[1] = A[1] + np.eye(num_nodes)
        assert 0 <= A.any() <= 1
        assert 0 <= A.sum(axis=0).any() <= 1
        self.A = torch.tensor(A, dtype=torch.float32, requires_grad=False)

    def get_adjacency(self):
        """返回邻接矩阵"""
        return self.adjacency

    def get_height_config_adjacency(self):
        """返回以关键点高度进行配置的邻接矩阵，比邻接矩阵多一个label维度"""
        return self.A

    def __height0b(self, i):
        # 0-based keypoint height
        ind_1based = i + 1
        return self.heights[ind_1based]
