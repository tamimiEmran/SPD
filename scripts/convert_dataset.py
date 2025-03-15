"""
convert_dataset module for V2X-Seq project.

This module provides functionality for convert dataset.
"""


# compare the binary content of two files
def compare_files(path1, path2):
    """
    path1: for a .pcd file
    path2: for a .pcd file
    """

    with open(path1, "rb") as f1, open(path2, "rb") as f2:
        return f1.read() == f2.read()
    

if __name__ == "__main__":
    import os
    
    path1 = r"M:\Documents\Mwasalat\dataset\Full Dataset (train & val)-20250313T155844Z\Full Dataset (train & val)\V2X-Seq-SPD\V2X-Seq-SPD\V2X-Seq-SPD-vehicle-side-velodyne\V2X-Seq-SPD-vehicle-side-velodyne\000000.pcd"

    path2 = r"M:\Documents\Mwasalat\dataset\Full Dataset (train & val)-20250313T155844Z\Full Dataset (train & val)\V2X-Seq-SPD\V2X-Seq-SPD\V2X-Seq-SPD-vehicle-side-velodyne\000000.pcd"

    #change the 000000.pcd to 000001.pcd to see the difference

    def changeNumber(path):
        "increase the number of the file by 1"
        #the end of the path is the number of the file and the extension
        path = path.split(".")
        number = int(path[0][-6:]) + 1
        path[0] = path[0][:-6] + str(number).zfill(6)
        return ".".join(path)
    for i in range(11000):
        path1 = changeNumber(path1)
        path2 = changeNumber(path2)
        print(compare_files(path1, path2))
        