__author__ = 'arenduchintala'
import sys

if __name__ == '__main__':
    # script here
    sample = int(sys.argv[1])
    writer = open('data/news.small.' + str(sample), 'w')
    for idx, line in enumerate(open('data/news.2012.en.shuffled.lower', 'r')):
        if idx % sample == 0:
            writer.write(line)
writer.flush()
writer.close()
