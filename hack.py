from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
import os
import glob
from PIL import Image
import time
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

# %%
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get("https://gshow.globo.com/realities/bbb/bbb20/votacao/paredao-bbb20-quem-voce-quer-eliminar-felipe-manu-ou-mari-a9f49f90-84e2-4c12-a9af-b262e2dd5be4.ghtml")
# %% click na manu
while 1:
	manu = driver.find_element_by_xpath('//*[@id="roulette-root"]/div/div[1]/div[4]/div[2]/div')
	manu.click()
	try:
		element = WebDriverWait(driver, 10).until(
			EC.presence_of_element_located((By.XPATH, '//*[@id="roulette-root"]/div/div[1]/div[4]/div[2]/div[2]/div/div/div[2]/div/div[2]'))
		)
	finally:
		click = driver.find_element_by_xpath('//*[@id="roulette-root"]/div/div[1]/div[4]/div[2]/div[2]/div/div/div[2]/div/div[2]')
		click.click()
		time.sleep(5)
		driver.refresh()
		time.sleep(5)
#%%

dinovo = driver.find_element_by_xpath('//*[@id="roulette-root"]/div/div[3]/div/div/div[1]/div[2]/button')
dinovo.click()


# %% Cool AI part
itemname = driver.find_element_by_xpath('//*[@id="roulette-root"]/div/div[1]/div[4]/div[2]/div[2]/div/div/div[2]/div/div[1]/span[2]')
selecao = driver.find_element_by_xpath('//*[@id="roulette-root"]/div/div[1]/div[4]/div[2]/div[2]/div/div/div[2]/div/div[2]/img')

item = itemname.text

Image.MAX_IMAGE_PIXELS = None # to avoid image size warning

imgdir = "C:/Users/Shadow/Documents/GitHub/DeepFaceRecThesis/bbb/"
# if you want file of a specific extension (.png):
filelist = [f for f in glob.glob(imgdir + "**/*.png", recursive=True)]
savedir = "C:/Users/Shadow/Documents/GitHub/DeepFaceRecThesis/bbb/out/"

selecao.screenshot(imgdir+"recogimages.png")
start_pos = start_x, start_y = (0, 0)
img = Image.open(imgdir+'recogimages.png')
width, height = img.size
cropped_image_size = w, h = (int(width/5), int(height))

frame_num = 1
for col_i in range(0, width, w):
	for row_i in range(0, height, h):
		crop = img.crop((col_i, row_i, col_i + w, row_i + h))
		name = os.path.basename('recogimages.png')
		name = os.path.splitext(name)[0]
		save_to= os.path.join(savedir, name+"_{:03}.png")
		crop.save(save_to.format(frame_num))
		frame_num += 1
