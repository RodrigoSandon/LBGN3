"""def outer_function(msg):
    message = msg

    def inner_function():
        print(message)
    
    return inner_function()

def inner_function():
        print(message)

hi_func = outer_function("hi")
bye_func = outer_function("bye")

hi_func()
bye_func()"""

"""def deco_func(og_func):
    
    return og_func()

@deco_func
def hi():
    print("hi")

@deco_func
def bye():
    print("bye")

deco_display_hi = deco_func(hi)
deco_display_bye = deco_func(bye)

deco_display_hi()
deco_display_bye()"""

# This gives an:
# TypeError: 'NoneType' object is not callable bc we giving the result an () at the end

# but what if we don't assign the func variables?
# im guessing we don't get that same error again, but then we lose on taking adv of
# first-class funcs - which is the whole point of decos

def deco_func(og_func):
    
    return og_func()

@deco_func
def hi():
    print("hi")

@deco_func
def bye():
    print("bye")

deco_func(hi)
deco_func(bye)

# TypeError: 'NoneType' object is not callable bc we giving the result an () at the end

def deco_func(og_func):

    def wrapper_func():
        # can add functionality without changing the func you're passing in
        print(f"executed this before {og_func.__name__}")
        return og_func
    
    return wrapper_func()

@deco_func
def hi():
    print("hi")

@deco_func
def bye():
    print("bye")

deco_display_hi = deco_func(hi)
deco_display_hi()

deco_display_bye = deco_func(bye)
deco_display_bye()