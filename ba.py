from PIL import Image
from rembg import remove

# 读取图片
input_path = r"I:/ec74c47f84d93dec8ba1bc1b53c86d19.jpeg"
output_path = r"I:/tianx.png"
input_image = Image.open(input_path)

# 移除背景
output_image = remove(input_image)

# 将背景替换为纯白色
output_image = output_image.convert("RGBA")
data = output_image.getdata()
new_data = []
for item in data:
    if item[0] == 0 and item[1] == 0 and item[2] == 0:
        new_data.append((255, 255, 255, 0))
    else:
        new_data.append(item)
output_image.putdata(new_data)

# 保存图片
output_image.save(output_path)
