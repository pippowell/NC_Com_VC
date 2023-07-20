test_list = [1, 4, 5, 7, 3, 0]
target_list = [10, 33, 20, 4, 5, 6, 34, 70, 11389, 45, 45, 657]

target_list.sort(reverse=True)
test_list.sort(reverse=True)

print(test_list)
print(target_list)

for value in test_list:
    print(value)
    del target_list[value]
    print(target_list)

