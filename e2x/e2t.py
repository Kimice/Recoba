#!/usr/bin/env python
#coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import xlrd
import codecs
from xml.dom import minidom


def open_excel(_file):
    try:
        data = xlrd.open_workbook(_file)
        return data
    except Exception, e:
        print str(e)


def read_excel(_file):
    data = open_excel(_file)
    table = data.sheets()[0]
    nrows, ncols = table.nrows, table.ncols
    _list = []
    for rownum in xrange(nrows):
        row = table.row_values(rownum)
        if row:
            element = []
            for i in xrange(ncols):
                element.append(row[i])
            _list.append(element)
    return _list


def write_text(_list):
    f = open('card.txt', 'w')
    for l in _list:
        for x in l:
            f.write(str(x).encode('utf8').strip('\n')+'|***|')
        f.write('\n')
    f.close()


def main():
    tables = read_excel('card.xls')
    write_text(tables)

if __name__ == "__main__":
    main()
