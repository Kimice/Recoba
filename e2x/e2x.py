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


def write_xml(_list):
    doc = minidom.Document()
    params = doc.createElement('params')
    doc.appendChild(params)

    codes = _list[0]
    childcodes = _list[1]
    for element in _list[2:]:
        param = doc.createElement('param')
        params.appendChild(param)
        for i, e in enumerate(element):
            if codes[i]:
                code = doc.createElement(str(codes[i]))
                param.appendChild(code)
                if codes[i+1]:
                    code.appendChild(doc.createTextNode(str(e)))
                else:
                    count = get_count(codes)
                    for j in range(count[i+1]+1):
                        childcode = doc.createElement(str(childcodes[i+j]))
                        code.appendChild(childcode)
                        childcode.appendChild(doc.createTextNode(str(element[i+j])))

    f = codecs.open('file.xml', encoding='utf8', mode='w')
    doc.writexml(f, " ", " ", "\n")
    f.close()
    # print doc.toprettyxml()


def get_count(values):
    count = [0]*len(values)
    i = 0
    while i < len(values):
        num = 0
        if not values[i]:
            while i+num < len(values) and not values[i+num]:
                num += 1
        count[i] = num
        if num:
            i += num
            continue
        i += 1
    return count


def main():
    tables = read_excel('file.xls')
    write_xml(tables)

if __name__ == "__main__":
    main()