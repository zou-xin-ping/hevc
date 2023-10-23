import os
from PIL import Image
import re
import operator
global width
global height
global x_blocks,x
global y_blocks,y


def split_image(image_path, output_path):
    for filename in os.listdir(image_path):
        print(filename)
        file_path = os.path.join(image_path, filename)
        image_name = os.path.splitext(filename)[0]
        image = Image.open(file_path)
        width, height = image.size
        x_blocks = width // 32
        y_blocks = height // 32
        
        for i in range(x_blocks):
            for j in range(y_blocks):
                left = i * 32
                upper = j * 32
                right = left + 32 if left + 32 < width else width
                lower = upper + 32 if upper + 32 < height else height

                box = (left, upper, right, lower)
                tile = image.crop(box)

                tile_path = os.path.join(output_path, f"{image_name}_{width}_{height}_{i}_{j}.png")
                tile.save(tile_path)

import os
from PIL import Image

# def combine_images(input_path, output_path):
#     images = []
#     x_blocks = []
#     y_blocks = []
#     for filename in sorted(os.listdir(input_path)):
#         print(filename)
#         match = re.search(r'(\d+)_(\d+)_(\d+)_(\d+).png', filename)
#         if match:

#             width = int(match.group(1))
#             height = int(match.group(2))
#             x_blocks.append(int(match.group(3))) 
#             y_blocks.append(int(match.group(4)))  
#             x = int(match.group(3))
#             y = int(match.group(4))

#         if filename.endswith(".png"):

#             image_path = os.path.join(input_path, filename)
#             # with Image.open(image_path) as image:
#             #     images.append(image)
#                 #print(type(images))
#             image = Image.open(image_path) 
#             #images.append(image)
#             result = Image.new("RGB", (width, height))
#             left = x * 32
#             upper = y * 32
#             right = left + 32 if left + 32 < width else width
#             lower = upper + 32 if upper + 32 < height else height
#             left_uper = (left,upper)
#             result.paste(image, box)
#             #image.show()
#     print(type(images[1]))

#     # width, height = images[0].size
#     # print(width)
#     # x_blocks = width // 32
#     # y_blocks = height // 32
#     # print(x_blocks)


#     x = max(x_blocks)
#     y = max(y_blocks)
#     print(x)
#     print(y)
#     result = Image.new("RGB", (width, height))
#     result.save('44.png') #output_path
#     for i in range(x+1): #15
#         for j in range(y+1): #7
#             left = i * 32
#             upper = j * 32
#             right = left + 32 if left + 32 < width else width
#             lower = upper + 32 if upper + 32 < height else height

            
#             box = (left, upper, right, lower)
#             left_uper = (left,upper)
#             if(j==0):
#                 print(i*(y+1))
#                 index =i*(y+1)
#             else:
#                 print(i*(y+1)+j)
#                 index = i*(y+1)+j
#             print(box)
#             tile = images[index] #.crop(box)
#             #tile.show()
#             #tile.save('1.png')
#             #print(tile)
#             result.paste(tile, box)
#     print(type(result))

#     result.save(output_path) #output_path

def combine_image(image_name, patch_image_path, combin_image_path,width=512,height=256):
    #遍历patch_image_path
    result = Image.new("RGB", (width, height))
    for filename in os.listdir(patch_image_path):
        if filename.endswith(".png"):
            file_path = os.path.join(patch_image_path, filename)
            print(filename)
            print(type(image_name))
            print(image_name)
            if filename.find(image_name): #operator.contains(str(filename),str(image_name)):#filename.__contains__(str(image_name)):
                print('yes')
                print(image_name)
                match = re.search(r'(\d+)_(\d+)_(\d+)_(\d+).png', filename)
                if match:

                    width = int(match.group(1))
                    height = int(match.group(2))
                    # x_blocks.append(int(match.group(3))) 
                    # y_blocks.append(int(match.group(4)))  
                    x = int(match.group(3))
                    y = int(match.group(4))
                    image = Image.open(file_path) 
                    #images.append(image)
                    
                    left = x * 32
                    upper = y * 32
                    right = left + 32 if left + 32 < width else width
                    lower = upper + 32 if upper + 32 < height else height
                    box = (left, upper, right, lower)
                    #left_uper = (left,upper)
                    result.paste(image, box)
    result.save(r'H:\image_reloc\dataset\test_AB\combine\we.png')#(r'{}\{}.png'.format(combin_image_path,image_name))
def get_image_name(image_path):
    for filename in os.listdir(image_path):
        print(filename)
        if filename.endswith(".png"):
            file_path = os.path.join(image_path, filename)
            image = Image.open(file_path)
            width, height = image.size
            image_name = os.path.splitext(filename)[0]+'\n'
        print(image_name)
        #获取这个图片的分辨率，把分辨率传参

        combine_image(image_name,r'H:\image_reloc\dataset\test_AB\out2',r'H:\image_reloc\dataset\test_AB\combine',width,height)

#split_image(r'H:\image_reloc\dataset\test_AB\test\20120611093440939.png',r'H:\image_reloc\dataset\test_AB\block')
#combine_images(r'H:\image_reloc\dataset\test_AB\block',r'H:\image_reloc\dataset\test_AB\out\44.png')
get_image_name(r'H:\image_reloc\dataset\test_AB\test')
#split_image(r'H:\image_reloc\dataset\test_AB\test',r'H:\image_reloc\dataset\test_AB\out2')