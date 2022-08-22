import numpy as np
import datetime
from calendar import monthrange
import sys

# Count the number of Mondays and Thursdays in a given year

def count_mondays_thursdays(year):
    num = 0
    for month in range(1,13):
        for day in range(1, monthrange(year, month)[1] + 1):
            num += 1*(datetime.datetime(year, month, day).weekday() in [0, 3])
    return num

if __name__ == "__main__":
    print(count_mondays_thursdays(int(sys.argv[1])))

