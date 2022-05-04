import pyautogui

screenWidth, screenHeight = pyautogui.size() # Get the size of the primary monitor.
print(screenWidth, screenHeight)

currentMouseX, currentMouseY = pyautogui.position() # Get the XY position of the mouse.
print(currentMouseX, currentMouseY)

pyautogui.moveTo(100, 150) # Move the mouse to XY coordinates.

#pyautogui.click()          # Click the mouse.
#pyautogui.click(100, 200)  # Move the mouse to XY coordinates and click it.
#pyautogui.click('button.png') # Find where button.png appears on the screen and click it.
dst_path = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-15/RDT D1/Session-20220121-100647_BLA-Insc-15_RDT_D1/example_ss.png"

"""#left, top, width, height
#instead of trying to guess what to take a screenshot of, give it an example input and it will choose it for you
img_to_find = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/projection_ss.png"
image_location = pyautogui.locateOnScreen(img_to_find, confidence=0.6)
print(image_location)

left = image_location.left # increase this to push image more right
top = image_location.top # increase this to push image to be more down
width = image_location.width
height = image_location.height"""

# FINAL COORDINATESs
left = 2450
top = 280
width = 1100
height = 730

im1 = pyautogui.screenshot(dst_path, region=(left,top, width, height))