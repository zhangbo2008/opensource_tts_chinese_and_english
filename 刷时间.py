import time
now = time.localtime()
nowt = time.strftime("%Y-%m-%d-%H-%M-%S", now)  #这一步就是对时间进行格式化
print(nowt)