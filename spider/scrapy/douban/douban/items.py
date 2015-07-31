from scrapy.item import Item, Field


class DoubanItem(Item):
    groupName = Field()
    groupURL = Field()
    totalNumber = Field()
    RelativeGroups = Field()
    ActiveUesrs = Field()