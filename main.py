print("SJTU-AuTop Openart starting")
print("github: https://github.com/SJTU-AuTop")
import sensor, image, time, math, os, tf 
from machine import UART,Pin
from pyb import LED

uart = UART(1, baudrate=115200)

# 分辨率使用建议QVGA较为合适，也可在某些时候进行分辨率切换为QQVGA以加快模型运行
# QQVGA: 160x120  QVGA: 320x240
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA) 
sensor.set_brightness(800) 
sensor.skip_frames(time = 1000)
sensor.set_auto_gain(True)      
sensor.set_auto_whitebal(True)  
sensor.set_auto_exposure(True)

clock = time.clock()
# Define Leds
red_light = LED(1)
green_light = LED(2)
blue_light = LED(3)
white_light = LED(4)

uart_str = b'\x09'

#Load Two_class net 动物水果三分类模型
animal_fruit_labels = ["neg","animals","fruit"]
#tf_animal_fruit_net = tf.load("fruit_animal_64x64_quant.tflite", load_to_fb=True)
tf_animal_fruit_net = tf.load("fruit_animal_quant.tflite", load_to_fb=True)

#Load Num net 找框+数字识别分类
#num_labels = ["neg","0", "1", "2", "3","4","5", "6", "7", "8", "9"]
#tf_num_net = tf.load("digit_24x24_quant.tflite", load_to_fb=True)

#Load Full_num net 数字识别全局分类器
full_num_labels = ["0", "1", "2", "3","4","5", "6", "7", "8", "9","neg"]
#full_num_net = tf.load("number128x64_quant.tflite", load_to_fb=True)
full_num_net = tf.load("number160x64_quant.tflite", load_to_fb=True)

net_threshold = 0.01
rect_threshold = 50000
show = True

## NNCU net
def nncu_detect(img1,net):
    scores = nncu.classify(net , img1)[0].output()
    max_score = max(scores)
    if(max_score>net_threshold):
        max_label = scores.index(max_score)
        return (max_score,max_label)
    return (0,None)
## Tflite net
def tf_detect(img1,net):
    scores = net.classify(img1)[0].output()
    print(scores)
    max_score = max(scores)
    if(max_score>net_threshold):
        max_label = scores.index(max_score)
        return (max_score,max_label)
    return (0, None)

## Save the img to sd
def img_save(img1,name):
    image_pat = "/save_img/"+str(name)+".jpg"
    img1.save(image_pat,quality=90)
    print(image_pat)

while(True):
    t0 = time.ticks()
    img = sensor.snapshot()
    ##check the uart and read
    uart_num = uart.any()
    if(uart_num):
        while(uart_num>0):
            uart_num-=1;
            uart_str = uart.read(1)
            print(uart_str)

    
    blue_light.on()

    ## Number detect Demo
    if(uart_str==b'\x02'):
        #对指定ROI进行全局分类
        result = full_num_net.classify(img,roi = ((320-160)//2,25,160,64))[0].output()
        max_score = max(result)
        max_label = result.index(max_score)

        if(show == True):
            img.draw_string((320-160)//2 + 80, 5, str(max_label)+str(": ")+ str(max_score),color = (255,0,0), scale = 2,mono_space=False)
            img.draw_rectangle(((320-160)//2,25,160,64), color = (255, 0, 0))

        #串口通信
        if(max_label<10):
            if((max_label)%2==0):
               uart_array = bytearray([0XFF,int(2),int(0),int(0),int(0)]) 
               uart.write(uart_array)
               green_light.toggle()
            else:
               uart_array = bytearray([0XFF,int(2),int(1),int(0),int(0)])  
               uart.write(uart_array)
               red_light.toggle()

    ## Apriltags detect Demo
    elif(uart_str==b'\x01'):
        for tag in img.find_apriltags(families=image.TAG25H9):
             if(tag.id()<10):
                 if(show==True):
                     img.draw_rectangle(tag.rect(), color = (255, 0, 0))
                     img.draw_cross(tag.cx(), tag.cy(), color = (0, 255, 0))
                 print("Tag ID %d" % tag.id())
                 #可根据旋转角度筛去一些负样本，更好的方法还是串接分类器二次确定
                 #print("rotation %d" %(180*tag.rotation()/math.pi))
                 if(tag.id() %2==0):
                    uart_array = bytearray([0XFF,int(1),int(0),int(0),int(0)])
                    uart.write(uart_array)
                    green_light.toggle()
                 else:
                    uart_array = bytearray([0XFF,int(1),int(1),int(0),int(0)])
                    uart.write(uart_array)
                    red_light.toggle()


    ## Fruit_Animal detect Demo
    elif(uart_str==b'\x04'):
        # ROI区域
        if(show==True):
            img.draw_rectangle((0,25,320,120), color = (0, 255, 0))

        for r in img.find_rects(threshold = rect_threshold,roi = (0,25,320,120)) :
            #差比和太大 - 非正方形   方框太小  不可能
            if(r.rect()[3]/r.rect()[2]>1.3 or r.rect()[3]/r.rect()[2]<0.6):
                break
            if(abs(r.rect()[3]) < 40 or abs(r.rect()[2]) < 40):
                break

            #模型输入与反馈
            img1 = img.copy(r.rect())
            (result_score,result_obj) = tf_detect(img1,tf_animal_fruit_net)
            
            #是否画图（找到的外接矩形及内点）
            if(show == True):
                for p in r.corners():
                    img.draw_circle(p[0], p[1], 5, color = (0, 255, 0))
                img.draw_rectangle(r.rect(), color = (255, 0, 0))
            
            #非负样本串口通信
            if(result_obj!=None and result_obj>0):
                result_label = animal_fruit_labels[result_obj]
                if(show==True):
                    img.draw_string(r.rect()[0] + 20, r.rect()[1]-20, result_label+str(result_score),color = (255,0,0), scale = 2,mono_space=False)
                    img.draw_cross(int(r.rect()[0] + r.rect()[2]/2), int(r.rect()[1] + r.rect()[3]/2), color = (0, 255, 0))
                print("Fruit_animal: %s" %(result_label))
                #返回校验、识别模式、识别结果、中点坐标
                if(result_obj==1):
                   green_light.toggle()
                   uart_array = bytearray([0XFF,int(4),int(0),int((r.rect()[0] + r.rect()[2]/2)/2),int((r.rect()[1] + r.rect()[3]/2)/2)])
                   uart.write(uart_array)
                else:
                   red_light.toggle()
                   uart_array = bytearray([0XFF,int(4),int(1),int((r.rect()[0] + r.rect()[2]/2)/2),int((r.rect()[1] + r.rect()[3]/2)/2)])
                   uart.write(uart_array)
    else:
        blue_light.off()
        green_light.off()
        red_light.off()
    
    print(time.ticks() - t0)
    #画出中点，辅助快速打靶标定
    img.draw_cross(int(160), int(120), color = (0, 0, 255))

