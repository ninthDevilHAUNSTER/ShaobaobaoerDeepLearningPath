import requests
token = "eSfGlK91rBmlBNKhGqlU7IY98tvYfd29"
url = "http://111.231.140.29:10000/"

s = requests.Session()
s.post(url,data={'token':token})
import re
while 1 :
	content = s.get(url+'question').text
	# print(content)
	cal = re.findall('<span>(.*)=\?</span>',content)
	try:
		val = eval(cal[0])
	except:
		print(content)
	content = s.post(url+'solution',data={'answer':val})
	print(content.text)