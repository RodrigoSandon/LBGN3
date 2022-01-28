

def string_to_list(str, separator=","):
    return str.split(separator)

def delete_specified_cols (dataframe, lst_to_remove):

def main():
    cols_to_remove = input(
        "Enter the columns you would like to remove (separate by coma): "
    )
    lst = string_to_list(cols_to_remove)
    print(lst)



if __name__ == "__main__":
    main()
