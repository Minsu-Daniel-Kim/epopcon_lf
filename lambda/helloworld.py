
from __future__ import print_function
import json

print('Loading function...')

def lambda_handler(event, context):

    print('value1 = ' + event.get('key1'))
    print('value2 = ' + event.get('key2'))
    print('value3 = ' + event.get('key3'))

    return event.get('key1')


if __name__ == '__main__':

    class Event:
        def get(self, key):
            e = {
                    'key1': 'value1',
                    'key2': 'value2',
                    'key3': 'value3'
                 }
            return e[key]
    context = 'context'
    event = Event()
    print(lambda_handler(event, context))