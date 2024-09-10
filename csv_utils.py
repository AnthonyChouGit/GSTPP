import csv

def read_keyval(path, keytype, valtype):
    rst_dict = dict()
    with open(path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            key = keytype(row[0])
            val = valtype(row[1])
            rst_dict[key] = val
    return rst_dict

def write_keyval(path, in_dict: dict):
    with open(path, 'w') as file:
        writer = csv.writer(file)
        for key in in_dict.keys():
            val = in_dict[key]
            writer.writerow([key, val])

if __name__ == '__main__':
    write_keyval('test.csv', ({10: 2.324, 20: -23.434}))
    rst = read_keyval('test.csv', int, float)
    print()