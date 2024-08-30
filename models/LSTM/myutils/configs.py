class Config(object):
	def __init__(self, *args, **kwargs):
		self.__dict__.update(kwargs)
		for (k, v) in self.__dict__.items():
			print("{}: {}".format(k,v))