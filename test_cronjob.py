from crontab import CronTab

cron = CronTab(user='Daniel')
job = cron.new(command='python writeDate.py')
job.minute.every(1)

cron.write()