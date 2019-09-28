import os
import csv

f = open("dataset_csv.csv", "w+")
for i in range(1, 13494):
    label = ""
    if i < 460:
        label = "A"
    elif i < 894:
        label = "B"
    elif i < 1363:
        label = "V"
    elif i < 1784:
        label = "G"
    elif i < 1976:
        label = "D"
    elif i < 2344:
        label = "GJ"
    elif i < 2912:
        label = "E"
    elif i < 3350:
        label = "ZH"
    elif i < 3780:
        label = "Z"
    elif i < 4219:
        label = "DZ"
    elif i < 4725:
        label = "I"
    elif i < 5182:
        label = "J"
    elif i < 5629:
        label = "K"
    elif i < 6046:
        label = "L"
    elif i < 6280:
        label = "LJ"
    elif i < 6744:
        label = "M"
    elif i < 7187:
        label = "N"
    elif i < 7372:
        label = "NJ"
    elif i < 7819:
        label = "O"
    elif i < 8290:
        label = "P"
    elif i < 9107:
        label = "R"
    elif i < 9565:
        label = "S"
    elif i < 10017:
        label = "T"
    elif i < 10465:
        label = "KJ"
    elif i < 11014:
        label = "U"
    elif i < 11475:
        label = "F"
    elif i < 11937:
        label = "X"
    elif i < 12377:
        label = "C"
    elif i < 12840:
        label = "CH"
    elif i < 13063:
        label = "DZH"
    else:
        label = "SH"

    f.write(str(i) + ".png," + label + "\n")

f.close()