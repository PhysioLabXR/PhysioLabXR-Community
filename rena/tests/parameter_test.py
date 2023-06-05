

def random_function(a, b, c, d):
    print(a)
    print(b)
    print(c)
    print(d)

param_dict = {
    'a' : 1,
    'b' : 3,
    'd': 5,
    'c': 4,

}

if __name__ == '__main__':
    random_function(**param_dict)

