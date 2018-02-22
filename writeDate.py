import datetime
with open('dateInfo.txt', 'a') as outFile:
    outFile.write('\n' + str(datetime.datetime.now()))