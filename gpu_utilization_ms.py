# -*- coding: utf-8 -*-  
from __future__ import print_function

import numpy as np  
import argparse
import matplotlib.pyplot as plt 
import socket
import linecache
from openpyxl import Workbook

def get_time_in_ms(str):
    s = str[:-1]
    if s[-1] == 'n':
        t = float(s[:-1])/1000000
    elif s[-1] == 'u':
        t = float(s[:-1])/1000
    elif s[-1] == 'm':
        t = float(s[:-1])
    else:
        t = float(s)*1000
    return t

def main():
    parser = argparse.ArgumentParser(description="progrom description")
    parser.add_argument('-f', '--file', type=str, default="test.txt")
    parser.add_argument('-x', '--xlsxfile', type=str, default="gpu_utilization.xlsx")
    args = parser.parse_args()
    tracefile = args.file
    xlsxfile = args.xlsxfile

    line_cur = 5
    start = 0
    end = 0
    start_time = 0
    end_time = 0
    precision = 1 # unit: ms
    
    with open(tracefile, 'r') as f:
        lines = f.readlines()
        line = lines[4]
        line = line.split()
        start_time = get_time_in_ms(line[0])
        line = lines[-1]
        line = line.split()
        end_time = get_time_in_ms(line[0]) + get_time_in_ms(line[1])

    slots = int((end_time - start_time) * 100 + 1)
    used = [0 for i in range(slots)]

    while 1 :
        line = linecache.getline(tracefile, line_cur) # skip the first two lines
        if not line:
            break
        line = line.split()
        start = get_time_in_ms(line[0])
        end = start + get_time_in_ms(line[1])

        for i in range(int(start*100), int(end*100)):
            if i >= slots:
                break
            used[i] = 1

        line_cur = line_cur + 1


    wb = Workbook()
    sheet = wb.active
    sheet["A1"].value = "time (ms)"
    sheet["B1"].value = "utilization (%)"
    for i in range(int(slots/(precision*100))):
        sheet["A"+str(i+2)].value = i*precision
        percent = 0
        for j in range(i*(precision*100), (i+1)*(precision*100)):
            if used[j] == 1:
                percent = percent + 1
        sheet["B"+str(i+2)].value = percent*1.0/(precision*100)*100
    wb.save(xlsxfile)

if __name__ == '__main__':
    main()