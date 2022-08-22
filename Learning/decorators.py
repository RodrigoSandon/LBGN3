# first class functions:
# treating it just like a variable
# can pass them into functions, can assign it as variables, can return them

# closures:
# takes advantage of functions
# able to return inner functions that has acces to local to the scope

"""def outer_function(msg):

    def inner_function():
        print(msg)
    
    return inner_function()

hi_func = outer_function("hi")
bye_func = outer_function("bye")

hi_func()
bye_func()"""

# Decorator
# a func that takes another unc as an argument that adds some kind o functionality
# without altering the source code that you passed in
# you're supposed to run wrapper func

# need to ba able to pass in as many arguments into wrapper func
# put (*args, **kwargs) in wrapper to have any arg or any key word args

from typing import Any


def deco_func(og_func):

    def wrapper_func(*args, **kwargs):
        # can add functionality without changing the func you're passing in
        print(f"executed this before {og_func.__name__}")
        return og_func(**kwargs)
    
    return wrapper_func

@deco_func
def hi():
    print("hi")

@deco_func
def bye():
    print("bye")

@deco_func
def bye_info(*args, **kwargs):
    print(f"bye {' '.join([value for key, value in kwargs.items()])}")

hi()
bye()
bye_info(name="rigo",age="23")

# Classes as decorators
# Essentially instead of having a method within a method, you have methods within a class
# to decorate your methods

class DecoClass(object):

    def __init__(self, og_func):
        self.og_func = og_func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        print(f"executed this before {self.og_func.__name__}")
        return self.og_func(**kwargs)

@DecoClass
def bye_info(*args, **kwargs):
    print(f"bye {' '.join([value for key, value in kwargs.items()])}")

bye_info(name="rigo",age="23")