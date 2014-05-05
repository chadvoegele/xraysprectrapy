import math

def l2_norm(image1, image2):
    # image frequencies should be the same length as the distances if they were
    # not modified after construction
    if len(image1.distances) != len(image2.distances):
        raise ValueError("images must be the same length.")

    for i in range(0, len(image1.distances)):
        if image1.distances[i] != image2.distances[i]:
            raise ValueError("images' distances must match")

    return math.sqrt(sum(((x-y)*(x-y) for (x,y)
                in zip(image1.frequencies, image2.frequencies))))

