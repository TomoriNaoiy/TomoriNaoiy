import pygame
import sys
import time
import random

drenhp=50
wohp=20
from pygame.locals import * 
diren=pygame.image.load("diren.png")
paoche=pygame.image.load("paoche.png")
dzidan=pygame.image.load("dzidan.png")
dzidanyidong=10
time=0
drenzidan=[]
def direnshow():
    global time
    
    screen.blit(diren,(500,230))
def dzidans():
    global dzidanyidong,time
    time+=10
    if time==500:       
        drenzidan.append([500,243])
        time=0
    for pppos in drenzidan:
        if pppos[0]<=0:
            drenzidan.remove(pppos)      
    return time
def showdzidan():
    global wohp
    for ppos in drenzidan:
        screen.blit(dzidan,(ppos[0],ppos[1]))
        if ppos[0]>=renwu_pos[0] and ppos[0]<=renwu_pos[0] and ppos[1]>=renwu_pos[1] and ppos[1]<=renwu_pos[1]+250:
            drenzidan.remove(ppos)
            wohp-=3
    for i in range(len(drenzidan)):
        drenzidan[i][0]-=10

renwu=[]
for i in range(2):
    aa=pygame.image.load(str(i+1)+".png")
    renwu1=pygame.transform.smoothscale(aa,(50,60))
    renwu.append(renwu1)
tupian3=pygame.image.load("3.png")
tupian3=pygame.transform.smoothscale(tupian3,(50,30))
renwu.append(tupian3)

canjump=True
jump=50
zidantp=pygame.image.load("zidan.png")
zidan=[]
renwu111=[]
renwu_pos=[150,230]
a=0
def show():
    global a
    screen.blit(renwu[a],(renwu_pos[0],renwu_pos[1]))
    
def renwumove():
    global a
    global jump
    global canjump
    if event.key==K_LEFT and renwu_pos[0]>0:
        if a==0:   
            a=1
        else:
            a=0        
        renwu_pos[0]-=20
        
    if event.key==K_RIGHT and renwu_pos[0]<800-50:
        renwu_pos[0]+=20
        if a==0:   
            a=1
        else:
            a=0       
    
                 
def zidans():
    if event.type==KEYDOWN:
        if event.key==K_s:
            zidan.append([renwu_pos[0]+50,renwu_pos[1]+10])          
    for pos in zidan:
        if pos[0]>=800:
            zidan.remove(pos)
def showzidan():
    global drenhp
    for pos in zidan:
        screen.blit(zidantp,(pos[0],pos[1]))
        if pos[0]>=500 and pos[0]<=500 and pos[1]>=230 and pos[1]<=250:
            zidan.remove(pos)
            drenhp-=3
    for i in range(len(zidan)):
        zidan[i][0]+=20
    
        
        
                












pygame.init()  
screen = pygame.display.set_mode((800, 600))

pygame.display.set_caption("go")

times = pygame.time.Clock()
bj=pygame.image.load("bj.png")
bj=pygame.transform.smoothscale(bj,(800,600))

    
    
    
    
    
    
    

while True:
    event = pygame.event.poll()
    screen.blit(bj,(0,0))
    renwu_pos[1]-=jump
    if renwu_pos[1]<=170:
        jump=-jump
    elif renwu_pos[1]>=230:
        jump=0
        renwu_pos[1]==230
        canjump=True
    direnshow()
    if wohp<=0 or drenhp<=0:
        break
            
    
    
    
    dzidans()
    showdzidan()
    
  
    
    
    
    
    
           
 
    if event.type==KEYDOWN:
        if event.key==K_UP:
            if canjump:
                jump=50
                a=2
                renwu_pos[1]=230
                canjump=False
    
  
                
            
                
    
    
    if event.type==KEYDOWN:
        renwumove()    
    zidans()
    showzidan()
    show()
    if renwu_pos[1]<=210:
        renwu_pos[1]+=30
    if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
            
    
   
   
   
    
    
    keys = pygame.key.get_pressed()
    if keys[K_ESCAPE]:
        pygame.quit()
        sys.exit() 
            
    pygame.display.flip()
    times.tick(30)

    
        
       
                 
