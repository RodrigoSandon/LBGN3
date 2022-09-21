"""def delete_recursively(root, name_endswith_list):

    for i in name_endswith_list:

        files = glob.glob(os.path.join(root, "**", "*%s") % (i), recursive=True)

        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))


def delete_recursively_test(root, name_endswith):

    files = glob.glob(os.path.join(root, "**", "*%s") % (name_endswith), recursive=True)

    for f in files:
        if name_endswith in f:
            print(f)
            try:
                os.remove(f)
                pass
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))

def delete_old_mp4s(root):
    #delete_recursively_test(root, "_.MP4")
    pass"""

x = r"\\"
y = "\\"
#z = "\\\" <- string literal undetermined
print(x)
print(y)