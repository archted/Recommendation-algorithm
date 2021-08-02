import re
path = r'G:\newnewne后厂理工学院资深计算广告(推荐方向）\资料代码\L9\tmp'
readName = path+r'\所有图片  写程序复制.md'
list1=[]
with open(readName,"r",encoding='utf-8') as f1:
    list1 = f1.readlines(    )
    print(list1)
writeFile = path+r'\9笔记.md'

list_write=[]
with open(writeFile, "r",encoding='utf-8') as f:
    list_write = f.readlines()
for idx,line in enumerate(list_write):
    # string = r'![image-20210731160856456](assets/image-20210731160856456.png)'
    # 将正则表达式编译成Pattern对象
    pattern = re.compile(".*?image-(\d*)")
    # 使用Pattern匹配文本，获得匹配结果，无法匹配时将返回None
    match = pattern.match(line)
    if match:
        # 使用Match获得分组信息
        print(match.group(1))
        # 去匹配列表中的字符串
        imageId = match.group(1)
        string  = ".*"+imageId+".*"
        for read_line in list1:
            match1 = re.compile(string).match(read_line)
            if match1:
                list_write[idx] = read_line
                print(read_line)
new_file_name = path+r'\tmp.md'
with open(new_file_name,"w",encoding='utf-8') as f2:
    for line in list_write:
        f2.write(line)
print("done")