def change_colors(name_of_color):
    css_code = ".aines-colors-class {{ color : {}; }}".format(name_of_color)
    print(css_code)

if __name__ == '__main__':
    change_colors('purple')

    # answer = input("Enter yes or no: ")
    # if answer == "yes":
    #     # Do this.
    #     print('hello papa')
    # elif answer == "no":
    #     # Do that.
    #     print("hello Aine")
    # else:
    #     print("Please enter yes or no.")

