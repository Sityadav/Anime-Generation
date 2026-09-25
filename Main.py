import os
from glob import glob
import numpy as np
import cv2
import warnings
from PySimpleGUI import popup_yes_no
from termcolor import cprint, colored
from Sub_Functions.Evaluation import evaluation
from Sub_Functions.Model import Anime_Generation
from Sub_Functions.VisualizationResults import AnalysisResults

if not os.path.exists(f'{format(os.getcwd())}\\Features'):
    os.makedirs(f'{format(os.getcwd())}\\Features')

warnings.filterwarnings('ignore', category=UserWarning)


def Get_Data(dataset, Exec=True):

    # If val is true, Execute preprocess and feature extraction steps, otherwise load features and labels
    if Exec:

        image_size = 64
        image_channel = 3  # RGB

        if dataset == 1:
            path = glob('Dataset\\dataset1\\data\\*.png')

        else:
            path = glob('Dataset\\dataset2\\images\\*.jpg')

        # initialize empty list to store images
        Images = []
        count = 1
        # Iterate each images
        for files in path:
            cprint(f"[⁉️] Count:{count}/{len(path)} ", color='grey', on_color='on_white')
            # Read image using cv2
            image = cv2.imread(files)
            im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            Images.append(cv2.resize(im, (image_size, image_size)))
            count += 1
        Images = np.reshape(Images,(-1, image_size, image_size, image_channel))

        if not os.path.exists(f'{format(os.getcwd())}\\Features\\dataset{dataset}'):
            os.makedirs(f'{format(os.getcwd())}\\Features\\dataset{dataset}')
        # save Images
        np.save(f'{format(os.getcwd())}\\Features\\dataset{dataset}\\Images.npy', Images)

    else:
        # Load images
        Images = np.load(f'{format(os.getcwd())}\\Features\\dataset{dataset}\\Images.npy')

    return Images


def Analysis(Images, dataset):

    cprint("Training Percentage Analysis for Anime generation", 'magenta', on_color='on_grey')

    # Initialize list to store comparative and performance analysis results
    COM_A = []
    COM_B = []
    COM_C = []
    COM_D = []
    COM_E = []
    COM_F = []
    COM_G = []
    COM_H = []

    PERF_A = []
    PERF_B = []
    PERF_C = []
    PERF_D = []
    PERF_E = []

    # Initialize image split percentage (20%) and epochs value
    split_percent = 0.2
    epochs = 10000

    # Iterate training percentage
    for i in range(5):

        splitvalue = int(len(Images) * split_percent)
        Split_images = Images[:splitvalue]

        cprint(f"Comparative Analysis {i}/5", 'magenta', on_color='on_grey')

        IG = Anime_Generation(Split_images, epochs)

        Image_C1 = IG.DCGAN()
        Image_C2 = IG.SRGAN()
        Image_C3 = IG.CBAM()
        Image_C4 = IG.LDS_GAN()
        Image_C5 = IG.FISTNet()
        Image_C6 = IG.HSDN()
        Image_C7 = IG.MSADGAN(opt=0, epochs=epochs)
        Image_C8 = IG.MSADGAN(opt=1, epochs=epochs)

        # Perform classification with varying epochs and get predictions
        cprint(f"Performance Analysis {i}/5", 'magenta', on_color='on_grey')

        Image_P1 = IG.MSADGAN(opt=1, epochs=2000)
        Image_P2 = IG.MSADGAN(opt=1, epochs=4000)
        Image_P3 = IG.MSADGAN(opt=1, epochs=6000)
        Image_P4 = IG.MSADGAN(opt=1, epochs=8000)
        Image_P5 = IG.MSADGAN(opt=1, epochs=epochs)

        [PSNR1C, RMSE1C, SSIM1C, FSIM1C, FID1C] = evaluation(Image_C1)
        [PSNR2C, RMSE2C, SSIM2C, FSIM2C, FID2C] = evaluation(Image_C2)
        [PSNR3C, RMSE3C, SSIM3C, FSIM3C, FID3C] = evaluation(Image_C3)
        [PSNR4C, RMSE4C, SSIM4C, FSIM4C, FID4C] = evaluation(Image_C4)
        [PSNR5C, RMSE5C, SSIM5C, FSIM5C, FID5C] = evaluation(Image_C5)
        [PSNR6C, RMSE6C, SSIM6C, FSIM6C, FID6C] = evaluation(Image_C6)
        [PSNR7C, RMSE7C, SSIM7C, FSIM7C, FID7C] = evaluation(Image_C7)
        [PSNR8C, RMSE8C, SSIM8C, FSIM8C, FID8C] = evaluation(Image_C8)

        [PSNR1P, RMSE1P, SSIM1P, FSIM1P, FID1P] = evaluation(Image_P1)
        [PSNR2P, RMSE2P, SSIM2P, FSIM2P, FID2P] = evaluation(Image_P2)
        [PSNR3P, RMSE3P, SSIM3P, FSIM3P, FID3P] = evaluation(Image_P3)
        [PSNR4P, RMSE4P, SSIM4P, FSIM4P, FID4P] = evaluation(Image_P4)
        [PSNR5P, RMSE5P, SSIM5P, FSIM5P, FID5P] = evaluation(Image_P5)

        # Append all values in empty array
        COM_A.append([PSNR1C, RMSE1C, SSIM1C, FSIM1C, FID1C])
        COM_B.append([PSNR2C, RMSE2C, SSIM2C, FSIM2C, FID2C])
        COM_C.append([PSNR3C, RMSE3C, SSIM3C, FSIM3C, FID3C])
        COM_D.append([PSNR4C, RMSE4C, SSIM4C, FSIM4C, FID4C])
        COM_E.append([PSNR5C, RMSE5C, SSIM5C, FSIM5C, FID5C])
        COM_F.append([PSNR6C, RMSE6C, SSIM6C, FSIM6C, FID6C])
        COM_G.append([PSNR7C, RMSE7C, SSIM7C, FSIM7C, FID7C])
        COM_H.append([PSNR8C, RMSE8C, SSIM8C, FSIM8C, FID8C])

        PERF_A.append([PSNR1P, RMSE1P, SSIM1P, FSIM1P, FID1P])
        PERF_B.append([PSNR2P, RMSE2P, SSIM2P, FSIM2P, FID2P])
        PERF_C.append([PSNR3P, RMSE3P, SSIM3P, FSIM3P, FID3P])
        PERF_D.append([PSNR4P, RMSE4P, SSIM4P, FSIM4P, FID4P])
        PERF_E.append([PSNR5P, RMSE5P, SSIM5P, FSIM5P, FID5P])

        # Increase training percentage for next iteration
        split_percent += 0.2

    # Save the comparative analysis results as numpy files

    np.save(f'{format(os.getcwd())}\\Analysis1\\dataset{dataset}\\COM_A.npy', COM_A)
    np.save(f'{format(os.getcwd())}\\Analysis1\\dataset{dataset}\\COM_B.npy', COM_B)
    np.save(f'{format(os.getcwd())}\\Analysis1\\dataset{dataset}\\COM_C.npy', COM_C)
    np.save(f'{format(os.getcwd())}\\Analysis1\\dataset{dataset}\\COM_D.npy', COM_D)
    np.save(f'{format(os.getcwd())}\\Analysis1\\dataset{dataset}\\COM_E.npy', COM_E)
    np.save(f'{format(os.getcwd())}\\Analysis1\\dataset{dataset}\\COM_F.npy', COM_F)
    np.save(f'{format(os.getcwd())}\\Analysis1\\dataset{dataset}\\COM_G.npy', COM_G)
    np.save(f'{format(os.getcwd())}\\Analysis1\\dataset{dataset}\\COM_H.npy', COM_H)

    print(colored(f"[✅] Execution of Training Percentage - Comparative Analysis Completed",
                  'green', on_color='on_grey'))

    # Save the performance analysis results as numpy files

    np.save(f'{format(os.getcwd())}\\Analysis1\\dataset{dataset}\\PERF_A.npy', PERF_A)
    np.save(f'{format(os.getcwd())}\\Analysis1\\dataset{dataset}\\PERF_B.npy', PERF_B)
    np.save(f'{format(os.getcwd())}\\Analysis1\\dataset{dataset}\\PERF_C.npy', PERF_C)
    np.save(f'{format(os.getcwd())}\\Analysis1\\dataset{dataset}\\PERF_D.npy', PERF_D)
    np.save(f'{format(os.getcwd())}\\Analysis1\\dataset{dataset}\\PERF_E.npy', PERF_E)

    print(colored(f"[✅] Execution of Training Percentage - Performance Analysis Completed ",
                  'green', on_color='on_grey'))


if __name__ == '__main__':

    # Prompt the user with a popup dialog and store their response in VVV.
    VVV = popup_yes_no("Do you want Complete Execution?")
    # If the user chooses "Yes" , Execute full code
    if VVV == "Yes":

        for i in range(1, 3):
            # Get Features and Labels
            Images = Get_Data(i, True)

            # Perform TP (Training Percentage) analysis on the Features and Labels.
            Analysis(Images, i)

            # Generate analysis results.
            AnalysisResults(i)

    # show graph only
    else:
        # If the user chooses "No," skip the data processing steps and only generate analysis results.
        for i in range(1, 3):
            # Generate analysis results.
            AnalysisResults(i)

