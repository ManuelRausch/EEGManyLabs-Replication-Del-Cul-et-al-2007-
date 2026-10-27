################################################################################
########## EEGManyLabs: Replication of Del Cul et al. 2007  ####################
################################################################################
#
# TEMPLATE FOR THE EEG-EXPERIMENT
# some changes will be necessary if additional labs late-join. 

# Manuel Rausch, based on code by Kobe Desender and Rene Skukkies

# The script was created using Psychopy standalone version 'PsychoPy 2024.2.4'. 
# Using the PsychoPy standalone is recommended for the EEGmanyLabs Del Cul replication project, because it is also recommended by the PsychoPy developpers. 
# However, if installing the standalone version is not an option for your lab, the following Python modules are required:
# - psychopy==2024.2.4
# - pyglet==1.4.11
# - numpy==1.24.4
# - matplotlib==3.4.3

# In addition, it is necessry to install the font COURIER NEW.

# IMPORTANT: Before running the experiment, please set monitor brightness in way so pure white corresponds to 100 cd/m^2. 

# For issues with this code contact manuel.rausch@aau.at

### TO BE ADAPTED BY EACH PARTICIPATING LAB ### 

portType = "parallel"  # Options: "parallel", "lsl", "none". Use "none" for a purely behavioral experiment
parallelPortAddress = 0x0378  # Change according to local hardware

MonitorResolutionInPxls = [1920, 1080]  # enter here the resolution of the monitor you run the experiment on in Pixels
ViewingDistanceInCm = 80 # please position your subjects aprpoximately 80 cm from the monitor. 
ScreenWidthInCm = 54 # enter the width of the monitor, in cm. When measuring your computer monitor, please start from the inner edge of the frame, where the visible screen begins
ScreenHeightInCm = 30 # Enter the HEIGHT of the monitor, in cm. When measuring your computer monitor, please start from the inner edge of the frame, where the visible screen begins
LabID = '' # 'Belgrade', 'Münster', 'Stuttgard', 'Verona', 'Kraków', 'Eichstätt'

myFontFile = 'cour.ttf' # only necessary to set if the font Courier is not installed on the local machine. Please provide the full path to the file.
myScreen = 0 # 0 = Experiment is shoen on primary monitor, 1 = second monitor, etc.
rgb_at_100_cd_m2 = [0.37, 0.37, 0.37] 
    #RGB-values that set maximum luminance at 100 cd/m^2. We use PsychoPy scaling: 1 is max, and -1 ks minimal
    # Needs to be masured. Gue
rgb_at_1_cd_m2 = [-.90, -.90, -.90] 
    #RGB-values that set maximum luminance at 1 cd/m^2. (# CRT monitors were not completely black. 
    # To simulate the behaviour of a CRT monitor, we are setting the background only to nearly black . 




### Experiment

# import packages (needs to be installed, are included in Standalone PsychoPy per default) 
from psychopy import visual as vis
from psychopy import event, core, gui, data
import numpy as np
import pyglet
from matplotlib import font_manager 

# import modules (should be part of a standard Python distribution) 
import os, random
import platform
from math import atan2, degrees

# Set working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

##Create a Data folder if it doesn't exist yet
if os.path.isdir('Data') == False:
    os.mkdir('Data')

# EEG trigger configuration 

TRIGGER_TARGET = 10
TRIGGER_MASK = 20
TRIGGER_RESPONSE = 30
TRIGGER_VISIBILITY = 40

# Initialize selected interface once
if portType == "parallel":

    from psychopy import parallel
    
    triggerPort = parallel.ParallelPort(address=parallelPortAddress)
    triggerPort.setData(0)

    def sendTrigger(code):
        triggerPort.setData(code)

    def resetTrigger():
        triggerPort.setData(0)

elif portType == "lsl":

    from pylsl import StreamInfo, StreamOutlet

    streamInfo = StreamInfo(name="DelCulMarkers", type="Markers", channel_count = 1, 
        nominal_srate=0, channel_format="int32", source_id="EEGManyLabs_DelCul")
    triggerOutlet = StreamOutlet(streamInfo)

    def sendTrigger(code):
        triggerOutlet.push_sample([code])

    def resetTrigger():
        pass  # LSL markers are discrete events

elif portType == "none": 
    def sendTrigger(code):
        pass
    def resetTrigger():
        pass

else:
    raise ValueError(f"Unknown portType: {portType}")




# Define participant info dictionary
info = {
    "ID": "", 
    "Age": "", 
    "Gender": ["", "Man", "Woman", "Non-binary", "Prefer to self-describe", "Prefer not to say"], 
    "Handedness": ["", "left", "right", "both", "Prefer not to say"]
}

# Create dialog box
myDlg = gui.DlgFromDict(dictionary = info, title = "Masked number discrimination experiment with EEG")

# If user selected "Prefer to self-describe", ask for input
if info["Gender"] == "Prefer to self-describe":
    extraDlg = gui.Dlg(title="Self-Describe Gender")
    extraDlg.addField("Enter your gender identity:")
    extraDlg.show()
    if extraDlg.OK:
        info["Gender"] = extraDlg.data[0]  # Store the user's response

# Filename
sub = info['ID']; age = info['Age']; gender = info['Gender']; handedness = info['Handedness'];
timestamp = data.getDateStr() # use psychopy function here
file_name = "Data/ReplDelCul_%s_%s_%s" % (LabID, sub, timestamp)

#Clock
clock = core.Clock()

## Define number of trials/blocks/etc
nTraining = 36 # unknown how many trials actually there were for training, and how the training looked like. 
nTrials = (4*8*4*4 + 80) #multide of 4 (target) * 8 (soa) * 4 (position) * 4 (repetitions) + 80 (mask only) = 592 trials 

nTrialsPerBlock = 74
nBlocks = 9

##timing of stimuli, in frames
ITI = 60 # Intertrial interval in flips, 60 flips of 1/60 = 1s. Not mentioned in the paper but we strongly suspect that some form of ITI had to have been there. We set it to 1 s. 
time_warning = 6 ## Number of flips in which dots converge to the fixation cross. Servesa as warning that the trial now starts. #
                  # 6 flips were chosen so the warning takes 100 ms, the same as in Charles et al. (2023)
time_target = 1 #  # 1 #target time in flips, 1 flip = 16.7 ms
# are there any peripheral moving dots? If yes, how long are the moving dots presented?
time_soa = [1,2,3,3,4,4,5,6] #SOA duration in flips, 1,2,3,4,5*16ms = 16,33,50,66,83,100ms. 
    # Note2: 50ms and 66ms appear twice as often
    # 1 SOA is not identical to target absent. 
time_soa_training = [6, 5, 4, 3, 2, 1] #  not reported; using the same as used in the main experiment. 
time_mask = 150 #  #mask duration in flips, 15 flips of 1/60 = 250ms
post_target_delay = 48 # mask-response delay in flips, 48 flips = 800ms
post_response_delay = 0
scale_sensitivity = .1 #delay between moving the rating scale cursor and checking the keys again (i.e. allowing a smooth rating scale)

myFont = "Courier New" # "Courier New" 
    # Lucie Charles provided an experiment with Courier New
    # It would have been very plausible that this font was used by Del Cul as well. You can see from the Fig. that Del Cul's script was serif.
    



# check if Courier New is available 
font_path = font_manager.findfont(font_manager.FontProperties(family=myFont), fallback_to_default=False)
if myFont not in font_path:
    print(f"The font '{myFont}' needs to be installed! Trying to use font from file.")
    try:
        pyglet.font.add_file(myFontFile)
    except FileNotFoundError:
        print("Please download the file 'cour.ttf' from the shared Google folder if you haven't yet.\n"
              "Then, please provide the correct, complete path and filename for myFontFile.")
        core.quit()


### visual features

win  = vis.Window(size = MonitorResolutionInPxls, screen = myScreen, 
                  color = rgb_at_1_cd_m2, allowGUI = False , 
                  units = 'norm', fullscr=True)
win.mouseVisible = False
# check refresh rate
refresh_rate = vis.getMsPerFrame(win, nFrames = 60, showVisual=False, msg='', msDelay=0.0) #averge, median, SD
#Crash if different from 60hz
if refresh_rate[0] <  16 or refresh_rate[0] >  17: # check refresh rate is ~ 1/60
    print("Refresh rate has to be 60hz!")
    core.quit()

scr_width = win.size[0]
scr_height = win.size[1]



# stting-up real-time keyboard state tracking in PsychoPy using Pyglet
key = pyglet.window.key               # Contains all possible key codes
keyboard = key.KeyStateHandler()      # Dictionary containing state of keys
win.winHandle.push_handlers(keyboard) # Connect dictionary with window

#Function to convert visual degrees to pixels, to make it platform independent (Note that a single visual degree typically corresponds to 30 - 60 pixels)

def deg2pix(size_in_deg, h = ScreenHeightInCm, d = ViewingDistanceInCm, r = scr_height):
    #size_in_deg=the stimulus size in degrees
    #h=monitor height in cm
    #d=distance between monitor and participant in cm
    #r=vertical resolution of the monitor

    # Degrees per pixel:
    deg_per_px = degrees(atan2(.5*h, d)) / (.5*r)
    # Stimulus size in degrees
    size_in_px = size_in_deg / deg_per_px
    return(size_in_px)

# Elements degrees of visual angle
radius_dot_degrees = .075 # .075 # on-the-fly choice to choice to determine dot size, not reported in the paper
upper_mask_distance_degrees = .7 # .9 # paper: "surrounding but not touching". on-the-fly-decision
lower_mask_distance_degrees = .7 # .9  # paper: "surrounding but not touching". on-the-fly-decision
left_mask_distance_degrees = .6 # .7 #random choice to determine vertical target-mask difference, not mentioned in the paper ("not overlapping")
right_mask_distance_degrees = .6 # .7 #rand

stim_height = 1.04 # The size is not reported in the paper. 
                   # estimated from the experimental programm shared by Lucie Charles 
                   # Assuming a viewing distance of 57 cm, the stimulus is 1.28°. 
                   # Assuming a viewing distance of 70 cm implies a stimulus size of 1.04 deg
                   # Assuming a viewing distance of 100 cm implies a stimulus size of 0.73 deg
                   # Based on the pilot, I think that 1.04 is the most likeliy choice
                   # Random estimate by Kobe Desender: .9 random choice to determine size of stimuli.  
                    
fix_height = 0.6                  
pos_dot_degrees = 1.4 #dots are 1.4° above/below or left/right from center
pos_dot_movement = pos_dot_degrees / (time_warning-1)   # Time_warning-1 is the number of steps we need from pos_dot_degrees to 0. 
# The speed of the converging dots is not reported in the paper. I designed here 
rating_scale_deg = [26,2] #size of rating scale, 26° and 2°
cursor_deg = [1.2, 2] #size of cursor, 1.2° and 2°
x_positions = np.linspace(-deg2pix(rating_scale_deg[0])/2,
                            deg2pix(rating_scale_deg[0])/2,21) #21 positions for the rating scale, in pixels

#position of dots and masks
pos_dot_pix = deg2pix(pos_dot_degrees)
pos_dot = [(pos_dot_pix,pos_dot_pix),(-pos_dot_pix,pos_dot_pix),
           (pos_dot_pix,-pos_dot_pix),(-pos_dot_pix,-pos_dot_pix)]

#grey target dots (note, position is relative to the center)
radius_dot_pix = deg2pix(radius_dot_degrees)
fix0   = vis.Circle(win, units='pix', radius = radius_dot_pix, fillColor=rgb_at_100_cd_m2)
fix1   = vis.Circle(win, units='pix', radius = radius_dot_pix, fillColor=rgb_at_100_cd_m2)
fix2   = vis.Circle(win,units='pix',radius=radius_dot_pix,fillColor=rgb_at_100_cd_m2)
fix3   = vis.Circle(win,units='pix',radius=radius_dot_pix,fillColor=rgb_at_100_cd_m2)

fix   = vis.TextStim(win, units = 'pix', text = "+", color = rgb_at_100_cd_m2, height = deg2pix(fix_height), font = myFont)

# Text labels
resp_labels1 = vis.TextStim(win, text = 'Was the number larger or smaller than five?', 
                            pos = (0,.1), height = .05, font = myFont, 
                            color = rgb_at_100_cd_m2) #height was .05 here
resp_labels2 = vis.TextStim(win, text = 'Smaller                     Larger', 
                            pos = (0,0), height = .05, font = myFont, color = rgb_at_100_cd_m2)
Welcome_txt = 'Welcome to our experiment!\n\n'
Welcome_txt += 'Please keep your eyes fixated on the cross throughout the entire experiment.\n\n'
Welcome_txt += 'On each trial, you will briefly see a number, which will then be replaced by surrounding letters.\n'
Welcome_txt += 'The number may appear in one of four locations relative to the fixation cross: bottom-left, bottom-right, top-left, or top-right.\n\n'
Welcome_txt += 'Your task is to indicate whether the number was smaller or larger than 5.\n'
Welcome_txt += 'If the number is SMALLER than 5, press "D".\n'
Welcome_txt += 'If the number is LARGER than 5, press "F".\n\n'
Welcome_txt += 'After responding, please indicate how visible the number was.\n'
Welcome_txt += 'To move the cursor on the scale, use the left and right arrow keys.\n\n'
Welcome_txt += 'Press SPACE to start the experiment. Good luck!'

Welcome_txt_height = .05 

##define keys
choice_keys = ['d','f','escape'] #left, right, escape

#slider
rect = vis.Rect(win, units = "pix", width = deg2pix(rating_scale_deg[0]),
                 height=deg2pix(rating_scale_deg[1]), color = [x * 0.5 for x in rgb_at_100_cd_m2], 
                 fillColor= [x * 0.5 for x in rgb_at_100_cd_m2])
cursor = vis.Rect(win, units = "pix", 
                  width = deg2pix(cursor_deg[0]),
                  height = deg2pix(cursor_deg[1]),
                  color = rgb_at_100_cd_m2, 
                  fillColor=rgb_at_100_cd_m2)
rating_question = vis.TextStim(win, units = 'pix', text = 'How visible was the number?', 
                               pos = (0 ,100), height = 20, color = rgb_at_100_cd_m2, font = myFont)
rating_text1 = vis.TextStim(win, units = 'pix',text = 'Not seen', 
                            pos = (-deg2pix(rating_scale_deg[0])/2, 
                            -deg2pix(rating_scale_deg[1])), height=20, color = rgb_at_100_cd_m2, font = myFont)
rating_text2 = vis.TextStim(win, units = 'pix',text = 'Maximal Visibility', 
                            pos = (deg2pix(rating_scale_deg[0])/2, 
                            -deg2pix(rating_scale_deg[1])), height=20, color = rgb_at_100_cd_m2, font = myFont)
rating_text3 = vis.TextStim(win, units = 'pix', text = 'Press space to confirm', 
                            pos = (0,-deg2pix(3)), height=20, color =rgb_at_100_cd_m2, font = myFont)

###MAKE A DATA FILE
#TrialHandler

info = {"ID": sub, "age": age, "gender": gender, "handedness": handedness, 
        "LabID": LabID, "os_name": platform.system(), "date": timestamp,  "portType": portType}
thisExp = data.ExperimentHandler(dataFileName = file_name, extraInfo=info)

# Balance frequencies across trials

target = np.repeat(np.array([1,4,6,9]),(nTrials-80)/4)  #generate targets
cresp = np.repeat(np.array(['left','left','right','right']),(nTrials-80)/4)  # correact answers!
soa = np.repeat(np.array(time_soa*4), (nTrials-80)/4/len(time_soa))  
loc = np.array([0,1,2,3]*int((nTrials-80)/4))
#Add non-mask condition
target = np.concatenate((target,np.repeat('',80))) #mask only = no target
cresp = np.concatenate((cresp,np.repeat('',80))) #mask only = no cresp
soa = np.concatenate((soa,np.repeat(1, 80))) #mask only=1 refresh rate=16ms
loc = np.concatenate((loc,np.repeat([0,1,2,3],80))) #mask only appears everywhere

#generate counter, and shuffle trials
idx = np.arange(nTrials) # creates a range of indices
random.shuffle(idx) #shuffle the indices
cresp = cresp[idx] # now reoder trials based on index
target = target[idx]
loc = loc[idx]
soa = soa[idx]

# add training

target_training = np.random.choice([1, 4, 6, 9], size=int(nTraining))
cresp_training = np.where(target_training < 5, 'left', 'right')
target = np.concatenate((target_training, target))
cresp = np.concatenate((cresp_training, cresp))
soa = np.concatenate((np.sort(np.resize(time_soa_training, nTraining))[::-1], soa))
loc = np.concatenate((np.random.choice([0, 1, 2, 3], size=nTraining), loc))

# Consent form 

Consent_txt = f"""
**Purpose of the study:**
The experiment investigates visibility in masked digit discrimination task. 
Specifically, the purpose is to measure stimulus difficulty in preparation for
a future EEG experiment. 

** Participation requirements**
At least 18 years of age, No psychological, psychiatric or neuropsychological diseases, 
espacially no epilepsy. 

** What data will be collected? **
Digit disrimination judgments under varying levels of difficulty, 
visibility judgments, reaction times
Demographic information (age, gender, handedness) is optional. 

** What happens to the data? **
The data will be analyzed in anonymous form. The anonymous data may be published in 
a scienfitic journal or an open data repository. 

** Voluntarines **
Your participation in this study is voluntary. You have the right to withdraw from the study
any time without negative consequences. 

If you agree to particpate in this experiment under these conditions, 
plese press SPACE

"""
Cons = vis.TextStim(win,text =Consent_txt , pos = (0,0), height = Welcome_txt_height, font = myFont, color = rgb_at_100_cd_m2)
Cons.draw() ; win.flip()
event.waitKeys(keyList = ['space'])  


#Instructions
    # Can I read the instructions in the code? If yes, can I translate the instructions?
Inst = vis.TextStim(win,text = Welcome_txt, pos = (0,0), height = Welcome_txt_height, font = myFont, color = rgb_at_100_cd_m2)
Inst.draw() ; win.flip()
event.waitKeys(keyList = ['space'])  
block = 1 # set block counter to 1

for trial in range(nTraining + nTrials):
    #Select target and masks for current trial location
    txt_target = vis.TextStim(win, pos = (pos_dot[loc[trial]][0], pos_dot[loc[trial]][1]), 
                              text = target[trial], color = rgb_at_100_cd_m2,
                              units = 'pix', height = deg2pix(stim_height), font = myFont)
    txt_mask_r = vis.TextStim(win, pos = (pos_dot[loc[trial]][0] + deg2pix(right_mask_distance_degrees), pos_dot[loc[trial]][1]), 
                              text = 'M', color = rgb_at_100_cd_m2, units='pix', 
                              height = deg2pix(stim_height), font = myFont)
    txt_mask_l = vis.TextStim(win, pos = (pos_dot[loc[trial]][0] - deg2pix(left_mask_distance_degrees), pos_dot[loc[trial]][1]), 
                              text = 'M', color = rgb_at_100_cd_m2, units = 'pix', 
                              height = deg2pix(stim_height), font = myFont)
    txt_mask_u = vis.TextStim(win, pos=(pos_dot[loc[trial]][0], pos_dot[loc[trial]][1] - deg2pix(lower_mask_distance_degrees )), 
                              text = 'E', color = rgb_at_100_cd_m2,  units='pix', 
                              height=deg2pix(stim_height), font = myFont)
    txt_mask_b = vis.TextStim(win, pos = (pos_dot[loc[trial]][0], pos_dot[loc[trial]][1] + deg2pix(upper_mask_distance_degrees)), 
                              text = 'E', color = rgb_at_100_cd_m2,
                              units='pix', height=deg2pix(stim_height), font = myFont)


    ##Now that we have all, we start presenting 
    # 1) Intertrial interval, 1s
    for i in range(ITI):
        fix.draw()
        win.flip()

    # 2) Warning: Converging dots for 0.1 s
    fix0.pos=pos_dot[0] # reset to starting position
    fix1.pos=pos_dot[1]
    fix2.pos=pos_dot[2]
    fix3.pos=pos_dot[3]
    start_fix = clock.getTime()
    for i in range(time_warning):
        fix.draw()  
        fix0.draw()
        fix1.draw()
        fix2.draw()
        fix3.draw()
        win.flip()  
        fix0.pos = fix0.pos - deg2pix(pos_dot_movement)
        fix1.pos = [fix1.pos[0] + deg2pix(pos_dot_movement),fix1.pos[1] - deg2pix(pos_dot_movement)]
        fix2.pos = [fix2.pos[0] - deg2pix(pos_dot_movement),fix2.pos[1] + deg2pix(pos_dot_movement)]
        fix3.pos = fix3.pos + deg2pix(pos_dot_movement)
    
    ## 3) Show the target for 1 frame
    start_target = clock.getTime()
    if str(target[trial]) != "":
        win.callOnFlip(sendTrigger, TRIGGER_TARGET)
    fix.draw()
    txt_target.draw()
    win.flip()    
    if str(target[trial]) != "":
        win.callOnFlip(resetTrigger)
    for i in range(1, time_target):
        fix.draw()
        txt_target.draw()
        win.flip()
    
    ## 4) ISI
    start_soa = clock.getTime()
    for i in range(soa[trial]):
        fix.draw()
        win.flip()

    ## 5) Mask for 250 ms 
    start_mask = clock.getTime()
    win.callOnFlip(sendTrigger, TRIGGER_MASK)
    fix.draw()
    txt_mask_r.draw()
    txt_mask_l.draw()
    txt_mask_u.draw()
    txt_mask_b.draw()
    win.flip()
    win.callOnFlip(resetTrigger)
    for i in range(1, time_mask):
        fix.draw()
        txt_mask_r.draw()
        txt_mask_l.draw()
        txt_mask_u.draw()
        txt_mask_b.draw()
        win.flip()

    ## 6) Deplay period 
    start_postmask = clock.getTime()
    for i in range(post_target_delay):
        fix.draw()
        win.flip()

    ## 7) Query a response
    start_rt = clock.getTime()
    event.clearEvents(); resp = []
    
    while len(resp) == 0:
        resp_labels1.draw()
        resp_labels2.draw()
        win.flip()
        resp  = event.getKeys(keyList = choice_keys)
    end_rt = clock.getTime()
    
    if len(resp) > 1:
        resp = [resp[0]]
        
    if resp != ['escape']: 
        sendTrigger(TRIGGER_RESPONSE)
        win.callOnFlip(resetTrigger)
    else: 
        print('Participant pressed escape')
        win.close()
        core.quit()
     
    # 8) Evaluate the response
    correct = float('nan')
    if cresp[trial] == 'left' and resp[0] == choice_keys[0]:
        correct = 1
    elif cresp[trial] == 'right' and resp[0] == choice_keys[0]:
        correct = 0
    elif cresp[trial] == 'left' and resp[0] == choice_keys[1]:
        correct = 0
    elif cresp[trial] == 'right' and resp[0] == choice_keys[1]:
        correct = 1
        
    ## 9) Ask for visibility
    resp_rating = [];event.clearEvents();  # always call this before recoding a new respons 
    curs_x = np.random.choice(np.arange(21)) #21 possible positions

    while -1:
        cursor.pos = (x_positions[curs_x],0)
        slider_rating = curs_x
        slider_rating_pix = x_positions[curs_x]

        rect.draw()
        cursor.draw()
        rating_text1.draw()
        rating_text2.draw()
        rating_text3.draw()
        rating_question.draw()
        win.flip()

        if keyboard[key.LEFT] and curs_x > 0:
            curs_x = curs_x - 1
            core.wait(scale_sensitivity)
        if keyboard[key.RIGHT] and curs_x < 20:
            curs_x = curs_x + 1
            core.wait(scale_sensitivity)
        if keyboard[key.ESCAPE]:
            print('Participant pressed escape')
            win.close()
            core.quit()
        if keyboard[key.SPACE]:
           sendTrigger(TRIGGER_VISIBILITY)
           win.callOnFlip(resetTrigger)
           break

        
    ## 10) Store data of current trial
    # include target present vs absent
    thisExp.addData("trialNo", trial) 
    thisExp.addData("blockNo" , block)
    thisExp.addData("cresp", cresp[trial])
    thisExp.addData("resp", resp)
    thisExp.addData("cor", correct)
    thisExp.addData("rating", slider_rating)
    thisExp.addData("rating_pix", slider_rating_pix)
    thisExp.addData("target", target[trial])
    thisExp.addData("soa", soa[trial])
    thisExp.addData("SOAms", round(soa[trial]*1/60*1000))
    thisExp.addData("location", loc[trial])
    thisExp.addData("start_fix", start_fix)
    thisExp.addData("start_target", start_target)
    thisExp.addData("start_soa", start_soa)
    thisExp.addData("start_mask", start_mask)
    thisExp.addData("start_postmask", start_postmask)
    thisExp.addData("start_rt" , start_rt)
    thisExp.addData("end_rt" , end_rt)

    # Proceed to next trial
    thisExp.nextEntry()

    if (trial + 1 == nTraining or (
        trial + 1 > nTraining and (trial + 1 - nTraining) % nTrialsPerBlock == 0
        and trial + 1 < nTraining + nTrials)):
            
        BlockEndText = 'Block No. ' + str(block) + ' of ' + str(nBlocks) + ' has been finished.\n\n'
        BlockEndText += 'Press space to start the next block.'
        pause = vis.TextStim(win,text=BlockEndText,
                                pos = (0,0), height=.05, font = myFont, color = rgb_at_100_cd_m2)
        pause.draw()
        win.flip()
        block += 1
        event.waitKeys(keyList = ['space'])

end = vis.TextStim(win,text='\nThanks a lot for participating!!!', pos=(0,0),height=.05, 
font=myFont, color=rgb_at_100_cd_m2)
end.draw();win.flip()
event.waitKeys(keyList = 'space')

##End of the experiment
win.flip()
resetTrigger()

core.quit()
