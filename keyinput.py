from getkey import getkey, keys
from kulka import Kulka
import time
speed = 0
rotate = 0
def user_input():
    key = getkey()
    global speed, rotate
    if key == keys.UP:
        speed += 8
    elif key == keys.DOWN:
        speed =0
    elif key == keys.N1:
        speed = 0
    elif key == keys.LEFT:
        rotate -= 10
    elif key == keys.RIGHT:
        rotate += 10
    elif key == keys.N2:
        rotate = 0

    if speed < 0:
        speed = 0
    elif speed >= 250:
        speed = 250

    if rotate < 0:
        rotate += 360
    rotate = rotate % 360



    return speed, rotate

if __name__ == '__main__':
  with Kulka('68:86:E7:09:FE:F9') as kulka:
    print('blue tooth connected')

    while (True):

     # kulka.set_inactivity_timeout(3600)
      speed, rotate = user_input()
      print("speed=",speed,"rotate=",rotate)
      kulka.roll(speed, rotate)
