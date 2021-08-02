import re

path = r'C:\Users\a00575982\Desktop\PPT笔记\20210801、1329备份'
readFile = path + r'\read1.md'
writeFile = path + r'\Thinking&Action.md'

'''
XX<img src="assets/image-20210724170239426.png" alt="image-20210724170239426" style="zoom:80%;" />XX
XX![image-20210724170235289](assets/image-20210724170235289.png)XX

-> XX<img src="https://github.com/archted/markdown-img/blob/main/img/image-20210725144557400-1627797399216.png?raw=true"
 alt="image-20210725144557400" style="zoom:80%;" />XX

'''

# list1 = []
# with open(readFile, "r", encoding='utf-8') as f1:
#     list1 = f1.readlines()
#     print(list1)

list_write = []
with open(writeFile, "r", encoding='utf-8') as f:
    list_write = f.readlines()


for idx, line in enumerate(list_write):
    # 将正则表达式编译成Pattern对象
    pattern = re.compile("(.*?)<img src=\"assets/(.*?)\" (.*?)/>(.*)")
    # 使用Pattern匹配文本，获得匹配结果，无法匹配时将返回None
    match = pattern.match(line)
    if match:
        while match:
            # 使用Match获得分组信息
            # 去匹配列表中的字符串
            string = match.group(1)+"<img src=\"https://github.com/archted/markdown-img/blob/main/img/" + match.group(2)+"?raw=true\" "+match.group(3)+"/>"+match.group(4)+"\r\n"
            list_write[idx] = string
            match = pattern.match(list_write[idx])
            print(string)
    pattern = re.compile("(.*?)!\[(.*?)\]\(assets/(.*?)\)(.*)")
    match = pattern.match(line)
    if match:
        while match:
            string = match.group(1)+"<img src=\"https://github.com/archted/markdown-img/blob/main/img/"+match.group(3)+"?raw=true\""+" alt=\""+match.group(2)+"\""+" style=\"zoom:90%;\" />"+match.group(4)+"\r\n"
            list_write[idx] = string
            match = pattern.match(list_write[idx])
            print(string)
new_file_name = path + r'\tmp.md'
with open(new_file_name, "w", encoding='utf-8') as f2:
    for line in list_write:
        f2.write(line)
print("done")
