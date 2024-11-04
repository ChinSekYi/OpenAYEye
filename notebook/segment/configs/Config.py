class Config(object):
	def __init__(self, *args, **kwargs):
		self.__dict__.update(kwargs)
		# for (k, v) in self.__dict__.items():
			# print("{}: {}".format(k,v))
	
	def get_Configs(self):
		# for (k, v) in self.__dict__.items():
		# 	print("{}: {}".format(k,v))
		return self.__dict__.items()