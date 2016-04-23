def MAE(predicted, expected):
    difference = 0
    for p, e in zip(predicted, expected):
        difference += abs(p - e)
    return difference / len(predicted)
