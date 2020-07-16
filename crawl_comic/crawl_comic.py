#! /usr/bin/env python3
import argparse, sys, os, errno
import logging
from tqdm import tqdm
import time
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')

def progress_bar(length,frequency):
    for i in tqdm(range(length)):
        time.sleep(1/frequency)

#✧(≖ ◡ ≖✿)
def dayspan(args):
    import datetime
    today = datetime.date.today()
    frequency=args.frequency
    if args.dayspan:
        logger.info('calculate day span since love')
        dayspan = daysBetweenDates(2013, 2, 14, today.year, today.month, today.day)
        progress_bar(dayspan,frequency)
        print ('Daidai & Duoduo have been together for: '+str(dayspan)+' Days ^_^~'+'\n'+
        'Congratulations to this affectionate couple!')

        logger.info('calculate day span since first photograph')
        dayspan = daysBetweenDates(2012, 9, 26, today.year, today.month, today.day)
        progress_bar(dayspan,frequency)
        print ('It has been '+str(dayspan)+' Days' +' since Daidai Took the First Picture of Duoduo ⁄(⁄ ⁄•⁄ω⁄•⁄ ⁄)⁄'+'\n'+
        'What a Romantic Man Daidai is.')

        logger.info('calculate day span since first travel')
        dayspan = daysBetweenDates(2014, 6, 10, today.year, today.month, today.day)
        progress_bar(dayspan,frequency)
        print ('It has been '+str(dayspan)+' Days' +' since Daidai Took the First Trip with Duoduo O(∩_∩)O~~'+'\n'+
        'The Journey across China is Beautiful and Remembering.')
	    
        dayspan = daysBetweenDates(2018, 8, 6, today.year, today.month, today.day)
        progress_bar(dayspan,frequency)
	
#	logger.info('calculate day span since meng meng came to the world as a pig')
 #       dayspan = daysBetweenDates(1996, 8, 6, today.year, today.month, today.day)
  #      progress_bar(dayspan,frequency)
   #     print ('It has been '+str(dayspan)+' Days' +' since Duoduo came to the world O(∩_∩)O~~'+'\n')

def isBigMonth(month):
    bigMonth = [1,3,5,7,8,10,12]
    if (month in bigMonth):
        return True
    else:
        return False     
def isLeapYear(year):
    if((year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)):
        return True
    else:
        return False
def validateDate(year1, month1, day1, year2, month2, day2):
    if(year1 < year2):
        return True
    elif(year1 == year2):
        if(month1 < month2):
            return True
        elif(month1 == month2):
            if(day1 < day2):
                return True
    else:
        return False
def daysBetweenDates(year1, month1, day1, year2, month2, day2):
    days = 0
    yearCount = year1
    monthCount = month1
    dayCount = day1
    while(validateDate(yearCount, monthCount, dayCount, year2, month2, day2)):
        days += 1
        if(monthCount == 12 and dayCount == 31):
            monthCount = 1
            dayCount = 0
            yearCount += 1
        if(isLeapYear(yearCount) and monthCount == 2):         
            if(dayCount == 29):
                dayCount = 1
                monthCount += 1
            else:
                dayCount += 1
        elif(isLeapYear(yearCount) != True and monthCount == 2):
            if(dayCount == 28):
                dayCount = 1
                monthCount += 1
            else:
                dayCount += 1
        elif(monthCount != 2):        
            if(isBigMonth(monthCount)):                      
                if(dayCount == 31):                             
                    dayCount = 1
                    monthCount += 1
                else:                                            
                    dayCount += 1
            elif(isBigMonth(monthCount) == False):             
                if(dayCount == 30):
                    dayCount = 1
                    monthCount += 1
                else:
                    dayCount += 1
    return days        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Love for Duoduo Last Forever Module')

    parser.add_argument('--dayspan', '-d', type=str,default='1',
        help='if print the day span since we fell in love')
    parser.add_argument('--frequency', '-f', type=float, default=5000,
        help='display frequency')
    args = parser.parse_args()

    logger = logging.getLogger('Love for Duoduo Last Forever Module')

    dayspan(args)



