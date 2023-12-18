"""переходы проигравших (DE_OLD)"""
DE_OLD_loser: list = [
    [4, 5, 7, 7, 9, 11, 13, 13, 15, 17, 19, 19, 21, 23, 25, 25, 27, 29, 31, 31, 33, 35, 37, 37, 39, 41, 43, 43, 45, 47, 49],
    [6, 6, 8, 8, 10, 12, 14, 14, 16, 18, 20, 20, 22, 24, 26, 26, 28, 30, 32, 32, 34, 36, 38, 38, 40, 42, 44, 44, 46, 48, 50],
    [8, 13, 9, 11, 13, 15, 15, 15, 17, 19, 21, 21, 23, 25, 27, 27, 29, 31, 33, 33, 35, 37, 39, 39, 41, 43, 45, 45, 47, 49, 51],
    [0, 10, 17, 21, 14, 16, 16, 16, 18, 20, 22, 22, 24, 26, 28, 28, 30, 32, 34, 34, 36, 38, 40, 40, 42, 44, 46, 46, 48, 50, 52],
    [0, 12, 18, 13, 25, 17, 19, 19, 21, 23, 23, 23, 25, 27, 29, 29, 31, 33, 35, 35, 37, 39, 41, 41, 43, 45, 47, 47, 49, 51, 53],
    [0, 0, 14, 22, 17, 29, 20, 20, 22, 24, 24, 24, 26, 28, 30, 30, 32, 34, 36, 36, 38, 40, 42, 42, 44, 46, 48, 48, 50, 52, 54],
    [0, 0, 16, 23, 26, 21, 33, 37, 23, 25, 27, 29, 31, 33, 31, 31, 33, 35, 37, 37, 39, 41, 43, 43, 45, 47, 49, 49, 51, 53, 55],
    [0, 0, 0, 18, 27, 30, 34, 38, 41, 26, 28, 30, 32, 34, 32, 32, 34, 36, 38, 38, 40, 42, 44, 44, 46, 48, 50, 50, 52, 54, 56],
    [0, 0, 0, 20, 28, 31, 25, 25, 42, 45, 29, 31, 33, 35, 37, 37, 39, 41, 39, 39, 41, 43, 45, 45, 47, 49, 51, 51, 53, 55, 57],
    [0, 0, 0, 0, 22, 32, 35, 39, 28, 46, 49, 53, 34, 36, 38, 38, 40, 42, 40, 40, 42, 44, 46, 46, 48, 50, 52, 52, 54, 56, 58],
    [0, 0, 0, 0, 24, 33, 36, 40, 43, 31, 50, 54, 57, 37, 39, 39, 41, 43, 45, 47, 49, 51, 47, 47, 49, 51, 53, 53, 55, 57, 59],
    [0, 0, 0, 0, 0, 26, 37, 29, 44, 47, 51, 55, 58, 61, 40, 40, 42, 44, 46, 48, 50, 52, 48, 48, 50, 52, 54, 54, 56, 58, 60],
    [0, 0, 0, 0, 0, 28, 38, 41, 33, 48, 35, 37, 59, 62, 65, 69, 43, 45, 47, 49, 51, 53, 55, 55, 57, 59, 55, 55, 57, 59, 61],
    [0, 0, 0, 0, 0, 0, 30, 42, 45, 49, 52, 38, 40, 63, 66, 70, 73, 46, 48, 50, 52, 54, 56, 56, 58, 60, 56, 56, 58, 60, 62],
    [0, 0, 0, 0, 0, 0, 32, 43, 46, 37, 53, 56, 41, 43, 67, 71, 74, 77, 49, 51, 53, 55, 57, 57, 59, 61, 63, 65, 67, 69, 63],
    [0, 0, 0, 0, 0, 0, 0, 34, 47, 50, 54, 57, 60, 44, 68, 72, 75, 78, 81, 85, 54, 56, 58, 58, 60, 62, 64, 66, 68, 70, 64],
    [0, 0, 0, 0, 0, 0, 0, 36, 48, 51, 41, 58, 61, 64, 47, 47, 76, 79, 82, 86, 89, 57, 59, 59, 61, 63, 65, 67, 69, 71, 73],
    [0, 0, 0, 0, 0, 0, 0, 0, 38, 52, 55, 47, 62, 65, 48, 48, 50, 80, 83, 87, 90, 93, 60, 60, 62, 64, 66, 68, 70, 72, 74],
    [0, 0, 0, 0, 0, 0, 0, 0, 40, 53, 56, 59, 49, 66, 69, 73, 51, 53, 84, 88, 91, 94, 97, 101, 63, 65, 67, 69, 71, 73, 75],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 57, 60, 63, 67, 70, 74, 77, 54, 85, 89, 92, 95, 98, 102, 105, 66, 68, 70, 72, 74, 76],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 44, 58, 61, 64, 53, 71, 75, 78, 81, 57, 59, 93, 96, 99, 103, 106, 109, 69, 71, 73, 75, 77],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 62, 65, 68, 72, 76, 79, 82, 58, 60, 62, 97, 100, 104, 107, 110, 113, 117, 74, 76, 78],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 63, 66, 69, 59, 55, 80, 83, 86, 61, 63, 65, 101, 105, 108, 111, 114, 118, 121, 77, 79],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 67, 70, 73, 77, 58, 84, 87, 90, 64, 66, 102, 106, 109, 112, 115, 119, 122, 125, 80],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 52, 68, 71, 74, 78, 81, 85, 88, 91, 94, 67, 69, 71, 110, 113, 116, 120, 123, 126, 129],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 54, 72, 75, 79, 82, 62, 89, 92, 95, 98, 70, 72, 74, 114, 117, 121, 124, 127, 130],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 56, 73, 76, 61, 83, 86, 90, 93, 96, 99, 71, 73, 75, 77, 118, 122, 125, 128, 131],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 77, 80, 65, 87, 66, 94, 97, 100, 103, 107, 76, 78, 119, 123, 126, 129, 132],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 78, 81, 84, 88, 91, 69, 98, 101, 104, 108, 111, 79, 81, 83, 127, 130, 133],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 62, 82, 85, 69, 92, 95, 72, 102, 105, 109, 112, 115, 82, 84, 86, 131, 134],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 83, 86, 89, 93, 96, 99, 103, 106, 110, 113, 116, 83, 85, 87, 89, 135],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 66, 87, 90, 73, 97, 100, 76, 107, 111, 114, 117, 120, 86, 88, 90, 136],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 68, 88, 91, 94, 98, 101, 104, 108, 112, 115, 118, 121, 124, 89, 91, 93],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 92, 95, 77, 102, 105, 80, 82, 116, 119, 122, 125, 128, 92, 94],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 93, 96, 99, 81, 106, 109, 83, 85, 120, 123, 126, 129, 132, 95],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 74, 97, 100, 103, 107, 110, 113, 86, 121, 124, 127, 130, 133, 96],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 76, 98, 101, 104, 85, 111, 114, 117, 89, 125, 128, 131, 134, 137],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 78, 102, 105, 108, 112, 115, 118, 90, 126, 129, 132, 135, 138],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 103, 106, 109, 89, 116, 119, 122, 93, 130, 133, 136, 139],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 82, 107, 110, 113, 93, 120, 123, 94, 96, 134, 137, 140],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84, 108, 111, 114, 117, 121, 124, 127, 97, 99, 138, 141],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 86, 112, 115, 118, 97, 125, 128, 131, 100, 139, 142],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 88, 113, 116, 119, 122, 126, 129, 132, 135, 103, 143],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 117, 120, 123, 101, 130, 133, 136, 104, 144],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 92, 118, 121, 124, 127, 105, 134, 137, 140, 107],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 94, 122, 125, 128, 131, 135, 138, 141, 108],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 96, 123, 126, 129, 132, 109, 139, 142, 145],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 98, 127, 130, 133, 136, 140, 143, 146],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 128, 131, 134, 137, 113, 144, 147],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 102, 132, 135, 138, 141, 145, 148],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 104, 133, 136, 139, 142, 117, 149],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 106, 137, 140, 143, 146, 150],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 108, 138, 141, 144, 147, 121],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 110, 142, 145, 148, 151],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 112, 143, 146, 149, 152],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 114, 147, 150, 153],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 116, 148, 151, 154],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 118, 152, 155],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 120, 153, 156],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 122, 157],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 124, 158],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 126],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128],
]