import numpy as np
nums={}

with open("我被迫挖了邪神的墙脚.txt", "r") as f:
    lines=f.readlines()

for line in lines:
    for word in line:
        if word in nums:
            nums[word]+=1
        else:
            nums[word]=1

print(len(nums))

# for num in nums:
#     print(num, end="")
write_name="字频.txt"
write_f=open(write_name, "w")
write_content=""
for name, num in nums.items():
    write_content+=name+"\t"+str(num)+"\n"
write_f.write(write_content)
write_f.close()