import matplotlib
import numpy as np
import pandas as pd
import os

os.getcwd()
os.chdir('/Users/sik/Desktop/dataset_test')

#(택1)

day = (x*365) + (x/4) - (x/100) + (x/400)

if day == 1 :
    print("월요일입니다")
elif day == 2 :
    print("화요일입니다")
elif day == 3:
    print("수요일입니다")
elif day == 4 :
    print("목요일입니다")
elif day == 5:
    print("금요일입니다")
elif day == 6:
    print("토요일입니다")
elif day == 7:
    print("일요일입니다")



#(택3)
# 구구단 출력 프로그램
for x in range(2, 10):
    for y in range(1, 10):
        print(x, "*", y, "=", x*y)
