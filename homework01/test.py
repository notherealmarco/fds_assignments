import torch as th

if __name__ == '__main__':
    i1 = [3, 3, 5, 2, 7, 0, 0, 2, 3]
    i2 = [3, 5, 3, 1, 0, 5, 2, 6, 8]
    kernel = [-1, 1, -1, 1]
    k_tensor = th.tensor(kernel).view(1, 1, 2, 2)

    i1_tensor = th.tensor(i1).view(1, 1, 3, 3)
    i2_tensor = th.tensor(i2).view(1, 1, 3, 3)

    o1 = th.conv2d(i1_tensor, k_tensor, padding=1)

    o2 = th.conv2d(i2_tensor, k_tensor, padding=1)
    print(o2+o1)

    o_result = o1 + o2

    sum_i1_i2 = i1_tensor + i2_tensor

    out = th.conv2d(sum_i1_i2, k_tensor, padding=1)

    # print(out == o_result)
