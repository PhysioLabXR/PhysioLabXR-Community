# from enum import Enum
#
# def create_enum_from_list(lst):
#     enum_values = {value: idx for idx, value in enumerate(lst)}
#     enum_class = Enum('CustomEnum', enum_values)
#     return enum_class
#
# # Example list
# my_list = ['apple', 'banana', 'orange', 'grape']
#
# # Creating the enum
# MyEnum = create_enum_from_list(my_list)
#
# # Using the enum
# print(MyEnum.apple)   # Output: CustomEnum.apple
# print(MyEnum.banana)  # Output: CustomEnum.banana
# print(MyEnum.orange)  # Output: CustomEnum.orange
# print(MyEnum.grape)   # Output: CustomEnum.grape
#
# # Accessing the index (value) of each member
# print(MyEnum.apple.value)   # Output: 0
# print(MyEnum.banana.value)  # Output: 1
# print(MyEnum.orange.value)  # Output: 2
# print(MyEnum.grape.value)   # Output: 3