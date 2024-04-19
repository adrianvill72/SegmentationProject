# This is an input class. Do not edit.
class LinkedList:
    def __init__(self, value):
        self.value = value
        self.next = None


def mergeLinkedLists(headOne, headTwo):
    # Write your code here.
    if(headOne.value<headTwo.value):
        return headOne.value,mergeLinkedLists(headOne.next,headTwo)
    else:
        return headTwo.value,mergeLinkedLists(headOne,headTwo.next)
    pass

headOne =LinkedList