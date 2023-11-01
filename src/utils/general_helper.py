def hash_2D_coordinate(x: int, y: int):
    tmp = y + (x + 1) / 2
    return x + tmp * tmp
