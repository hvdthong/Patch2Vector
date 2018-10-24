def load_file(path):
	with open(path) as f:
		content = f.readlines()
	return [x.strip() for x in content]

if __name__ == '__main__':
	print ('hello')
	print ('testing vim')
