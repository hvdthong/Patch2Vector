import re

if __name__ == '__main__':
    item = 'Added intro image .'
    matchObj = re.search("\^ignore update ' .* $", item)

