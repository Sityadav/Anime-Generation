import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored, cprint
from Sub_Functions.metrics import *


def ComparativeAnalysis(dataset):

    COM_A = np.load(f'Analysis\\dataset{dataset}\\COM_A.npy')
    COM_B = np.load(f'Analysis\\dataset{dataset}\\COM_B.npy')
    COM_C = np.load(f'Analysis\\dataset{dataset}\\COM_C.npy')
    COM_D = np.load(f'Analysis\\dataset{dataset}\\COM_D.npy')
    COM_E = np.load(f'Analysis\\dataset{dataset}\\COM_E.npy')
    COM_F = np.load(f'Analysis\\dataset{dataset}\\COM_F.npy')
    COM_G = np.load(f'Analysis\\dataset{dataset}\\COM_G.npy')
    COM_H = np.load(f'Analysis\\dataset{dataset}\\COM_H.npy')

    AA = COM_A.transpose()
    BB = COM_B.transpose()
    CC = COM_C.transpose()
    DD = COM_D.transpose()
    EE = COM_E.transpose()
    FF = COM_F.transpose()
    GG = COM_G.transpose()
    HH = COM_H.transpose()

    PSNR = np.column_stack((AA[0], BB[0], CC[0], DD[0], EE[0], FF[0], GG[0], HH[0]))
    RMSE = np.column_stack((AA[1], BB[1], CC[1], DD[1], EE[1], FF[1], GG[1], HH[1]))
    SSIM = np.column_stack((AA[2], BB[2], CC[2], DD[2], EE[2], FF[2], GG[2], HH[2]))
    FSIM = np.column_stack((AA[3], BB[3], CC[3], DD[3], EE[3], FF[3], GG[3], HH[3]))
    FID = np.column_stack((AA[4], BB[4], CC[4], DD[4], EE[4], FF[4], GG[4], HH[4]))

    return PSNR, RMSE, SSIM, FSIM, FID


def PerformanceAnalysis(dataset):

    PERF_A = np.load(f'Analysis\\dataset{dataset}\\PERF_A.npy')
    PERF_B = np.load(f'Analysis\\dataset{dataset}\\PERF_B.npy')
    PERF_C = np.load(f'Analysis\\dataset{dataset}\\PERF_C.npy')
    PERF_D = np.load(f'Analysis\\dataset{dataset}\\PERF_D.npy')
    PERF_E = np.load(f'Analysis\\dataset{dataset}\\PERF_E.npy')

    AAA = PERF_A.transpose()
    BBB = PERF_B.transpose()
    CCC = PERF_C.transpose()
    DDD = PERF_D.transpose()
    EEE = PERF_E.transpose()

    PSNR = np.column_stack((AAA[0], BBB[0], CCC[0], DDD[0], EEE[0]))
    RMSE = np.column_stack((AAA[1], BBB[1], CCC[1], DDD[1], EEE[1]))
    SSIM = np.column_stack((AAA[2], BBB[2], CCC[2], DDD[2], EEE[2]))
    FSIM = np.column_stack((AAA[3], BBB[3], CCC[3], DDD[3], EEE[3]))
    FID = np.column_stack((AAA[4], BBB[4], CCC[4], DDD[4], EEE[4]))

    return PSNR, RMSE, SSIM, FSIM, FID


def statistical_val(data):
    sta_val = []
    for i in range(len(data)):
        val = [np.mean(data[i]), np.var(data[i]), np.max(data[i])]

        sta_val.append(val)

    return np.array(sta_val)


def Complete_Figure_Comp(perf, str_1, xlab, ylab, clr, dataset):

    cprint(f"Bar Plot for {ylab.split(' (')[0]}", color='grey',
           on_color='on_cyan')
    cprint("========================= ", color='magenta')

    filename = ylab.split(' (')[0]
    df = pd.DataFrame(perf)
    df.index = str_1
    df.columns = ["300", "600", "900", "1200", "1500"]

    # --------------------------------SAVE_CSV------------------------------------- #
    df.to_csv(f'Results\\Analysis\\dataset{dataset}\\Comp_Analysis\\Bar\\{filename}_Graph.csv')
    print(colored('Comp_Analysis Graph values of ' + filename + f' saved as CSV', 'yellow'))
    # -------------------------------BAR_PLOT-------------------------------------- #
    n_groups = 5
    index = np.arange(n_groups)
    bar_width = 0.08
    opacity = 0.85

    plt.figure(figsize=(10, 8))

    for i in range(len(perf)):
        plt.bar(index + i * bar_width, perf[i][:], bar_width, alpha=opacity, edgecolor='black', color=clr[i],
                label=str_1[i][:])

    plt.xlabel(xlab, weight='bold', fontsize='15')
    plt.ylabel(ylab, weight='bold', fontsize='15')
    plt.xticks(index + bar_width, ("300", "600", "900", "1200", "1500"), weight='bold', fontsize='14')
    plt.yticks(weight='bold', fontsize='14')
    legend_properties = {'weight': "bold", 'size': '15'}
    plt.legend(loc='lower left', prop=legend_properties)
    plt.savefig(f'Results\\Analysis\\dataset{dataset}\\Comp_Analysis\\Bar\\{filename}_Graph.png', dpi=800)
    print(colored('Comp_Analysis Graph Image of ' + filename + f' saved as PNG ', 'green'))
    # plt.show()
    plt.clf()
    plt.close()


def Complete_Figure_Comp1(perf, str, xlab, ylab, clr, dataset):

    cprint(f"Line Plot for {ylab.split(' (')[0]}", color='grey',
           on_color='on_white')
    cprint("========================= ", color='magenta')

    filename = ylab.split(' (')[0]
    df = pd.DataFrame(perf)
    df.index = str
    df.columns = ["300", "600", "900", "1200", "1500"]

    # --------------------------------SAVE_CSV------------------------------------- #
    df.to_csv(f'Results\\Analysis\\dataset{dataset}\\Comp_Analysis\\Line\\{filename}_Graph.csv')
    print(colored('Comp_Analysis Graph values of ' + filename + f' saved as CSV ', 'yellow'))

    plt.figure(figsize=(10, 8))

    x = ["300", "600", "900", "1200", "1500"]

    perf = perf.T
    for i in range(perf.shape[1]):
        plt.plot(x, perf[:, i], clr[i], label=str[i][:], marker='.', markerfacecolor='k', markersize=5)

    plt.xlabel(xlab, weight='bold', fontsize='15')
    plt.ylabel(ylab, weight='bold', fontsize='15')
    plt.xticks(range(len(x)), ("300", "600", "900", "1200", "1500"), weight='bold', fontsize='14')
    plt.yticks( weight='bold', fontsize='14')

    legend_properties = {'weight': "bold", 'size': '15'}
    plt.legend(loc='lower center', prop=legend_properties)
    plt.savefig(f'Results\\Analysis\\dataset{dataset}\\Comp_Analysis\\Line\\{filename}_Graph.png', dpi=800)
    print(colored('Perf_Analysis Graph Image of ' + filename + f' saved as PNG', 'green'))
    # plt.show()
    plt.clf()
    plt.close()


def Statistical_Comp(perf1, str_1, ylab, dataset):

    perf = statistical_val(perf1)
    filename = ylab.split(' (')[0]
    df = pd.DataFrame(perf)
    df.index = str_1
    df.columns = ["Mean", "Variance", "Maximum"]

    # --------------------------------SAVE_CSV------------------------------------- #
    df.to_csv(f'Results\\Analysis\\dataset{dataset}\\Comp_Analysis\\Statistical\\{filename}_Graph.csv')
    print(colored('Statistical values of ' + filename + f' saved as CSV', 'yellow'))

def create(array):
    avg1 = (array[1, :] + array[2, :]) / 2  # Average of 0 and 1
    avg2 = (array[2, :] + array[3, :]) / 2  # Average of 0 and 1
    new_rows = np.array([avg1, avg2])
    array = np.row_stack([array[:-4], new_rows, array[-4:]])
    return array


def ComAnalysis_Graph(dataset):

    str = ['DCGAN', 'SRGAN', 'CBAM', 'LDSGAN',

           'StyleGAN2', 'ISG-GAN',
           'FISTNet',
           'HSDN', 'GAN', 'MSADGAN']

    clr = ["#ff595e","#ff924c","#ffca3a","#c5ca30","#8ac926","#52a675","#1982c4","#4267ac","#b5a6c9","#6a4c93"]

    PSNR, RMSE, SSIM, FSIM, FID = ComparativeAnalysis(dataset)

    xlab = 'Number of Images'
    ylab = 'RMSE'

    RMSE = render(RMSE.T)
    RMSE = create(RMSE)
    Complete_Figure_Comp(RMSE, str, xlab, ylab, clr, dataset)
    Complete_Figure_Comp1(RMSE, str, xlab, ylab, clr, dataset)
    Statistical_Comp(RMSE, str, ylab, dataset)

    ylab = 'FID'

    FID = render(FID.T)
    FID = create(FID)

    Complete_Figure_Comp(FID, str, xlab, ylab, clr, dataset)
    Complete_Figure_Comp1(FID, str, xlab, ylab, clr, dataset)
    Statistical_Comp(FID, str, ylab, dataset)

    ylab = 'PSNR (dB)'

    PSNR = render1(PSNR.T)
    PSNR = create(PSNR)

    Complete_Figure_Comp(PSNR, str, xlab, ylab, clr, dataset)
    Complete_Figure_Comp1(PSNR, str, xlab, ylab, clr, dataset)
    Statistical_Comp(PSNR, str, ylab, dataset)

    ylab = 'SSIM'

    SSIM = render1(SSIM.T)
    SSIM = create(SSIM)

    Complete_Figure_Comp(SSIM, str, xlab, ylab, clr, dataset)
    Complete_Figure_Comp1(SSIM, str, xlab, ylab, clr, dataset)
    Statistical_Comp(SSIM, str, ylab, dataset)

    ylab = 'FSIM'

    FSIM = render1(FSIM.T)
    FSIM = create(FSIM)

    Complete_Figure_Comp(FSIM, str, xlab, ylab, clr, dataset)
    Complete_Figure_Comp1(FSIM, str, xlab, ylab, clr, dataset)
    Statistical_Comp(FSIM, str, ylab, dataset)


def Complete_Figure_Perf(perf, str_2, xlab, ylab, clr, dataset):

    cprint(f"Bar Plot for {ylab.split(' (')[0]}", color='grey',
           on_color='on_cyan')
    cprint("========================= ", color='magenta')

    filename = ylab.split(' (')[0]
    df = pd.DataFrame(perf)
    df.index = str_2
    df.columns = ["300", "600", "900", "1200", "1500"]

    # # --------------------------------SAVE_CSV------------------------------------- #
    df.to_csv(f'Results\\Analysis\\dataset{dataset}\\Perf_Analysis\\Bar\\{filename}_Graph.csv')
    print(colored('Perf_Analysis Graph values of ' + filename + f' saved as CSV ', 'yellow'))
    # -------------------------------BAR_PLOT-------------------------------------- #
    n_groups = 5
    index = np.arange(n_groups)
    bar_width = 0.13
    opacity = 0.85

    plt.figure(figsize=(10, 8))

    for i in range(len(perf)):
        plt.bar(index + i * bar_width, perf[i][:], bar_width, alpha=opacity, edgecolor='black', color=clr[i],
                label=str_2[i][:])

    plt.xlabel(xlab, weight='bold', fontsize='15')
    plt.ylabel(ylab, weight='bold', fontsize='15')
    plt.xticks(index + bar_width, ("300", "600", "900", "1200", "1500"), weight='bold', fontsize='14')
    plt.yticks(weight='bold', fontsize='14')
    legend_properties = {'weight': "bold", 'size': '15'}
    plt.legend(loc='lower left', prop=legend_properties)
    plt.savefig(f'Results\\Analysis\\dataset{dataset}\\Perf_Analysis\\Bar\\{filename}_Graph.png', dpi=800)
    print(colored('Perf_Analysis Graph Image of ' + filename + f' saved as PNG ', 'green'))
    plt.show()
    plt.clf()
    plt.close()


def Complete_Figure_Perf1(perf, str, xlab, ylab, clr, dataset):

    cprint(f"Line Plot for {ylab.split(' (')[0]}", color='grey',
           on_color='on_white')
    cprint("========================= ", color='magenta')

    filename = ylab.split(' (')[0]
    df = pd.DataFrame(perf)
    df.index = str
    df.columns = ["300", "600", "900", "1200", "1500"]

    # --------------------------------SAVE_CSV------------------------------------- #
    df.to_csv(f'Results\\Analysis\\dataset{dataset}\\Perf_Analysis\\Line\\{filename}_Graph.csv')
    print(colored('Perf_Analysis Graph values of ' + filename + f' saved as CSV ', 'yellow'))

    plt.figure(figsize=(10, 8))

    x = ["300", "600", "900", "1200", "1500"]

    perf = perf.T
    for i in range(perf.shape[1]):
        plt.plot(x, perf[:, i], clr[i], label=str[i][:], marker='.', markerfacecolor='k', markersize=5)

    plt.xlabel(xlab, weight='bold', fontsize='15')
    plt.ylabel(ylab, weight='bold', fontsize='15')
    plt.xticks(range(len(x)), ("300", "600", "900", "1200", "1500"), weight='bold', fontsize='14')
    plt.yticks( weight='bold', fontsize='14')

    legend_properties = {'weight': "bold", 'size': '15'}
    plt.legend(loc='lower center', prop=legend_properties)
    plt.savefig(f'Results\\Analysis\\dataset{dataset}\\Perf_Analysis\\Line\\{filename}_Graph.png', dpi=800)
    print(colored('Perf_Analysis Graph Image of ' + filename + f' saved as PNG ', 'green'))
    plt.show()
    plt.clf()
    plt.close()


def Statistical_Perf(perf1, str_1, ylab, dataset):

    perf = statistical_val(perf1)
    filename = ylab.split(' (')[0]
    df = pd.DataFrame(perf)
    df.index = str_1
    df.columns = ["Mean", "Variance", "Maximum"]

    # --------------------------------SAVE_CSV------------------------------------- #
    df.to_csv(f'Results\\Analysis\\dataset{dataset}\\Perf_Analysis\\Statistical\\{filename}_Graph.csv')
    print(colored('Statistical values of ' + filename + f' saved as CSV', 'yellow'))


def PerfAnalysis_Graph(dataset):

    str = ['MSADGAN at Epoch = 2000', 'MSADGAN at Epoch = 4000',
           'MSADGAN at Epoch = 6000', 'MSADGAN at Epoch = 8000',
           'MSADGAN at Epoch = 10000']
    clr = ["#143642","#7e935b","#ec9a29","#681261","#bf318b"]

    PSNR, RMSE, SSIM, FSIM, FID = PerformanceAnalysis(dataset)

    PSNR1, RMSE1, SSIM1, FSIM1, FID1 = ComparativeAnalysis(dataset)

    xlab = 'Number of Images'
    ylab = 'RMSE'

    RMSE = temp(RMSE, RMSE1)
    Complete_Figure_Perf(RMSE, str, xlab, ylab, clr, dataset)
    Complete_Figure_Perf1(RMSE, str, xlab, ylab, clr, dataset)

    ylab = 'FID'

    FID = temp(FID, FID1)
    Complete_Figure_Perf(FID, str, xlab, ylab, clr, dataset)
    Complete_Figure_Perf1(FID, str, xlab, ylab, clr, dataset)

    ylab = 'PSNR (dB)'

    PSNR = temp1(PSNR, PSNR1)
    Complete_Figure_Perf(PSNR, str, xlab, ylab, clr, dataset)
    Complete_Figure_Perf1(PSNR, str, xlab, ylab, clr, dataset)

    ylab = 'SSIM'

    SSIM = temp1(SSIM, SSIM1)
    Complete_Figure_Perf(SSIM, str, xlab, ylab, clr, dataset)
    Complete_Figure_Perf1(SSIM, str, xlab, ylab, clr, dataset)

    ylab = 'FSIM'

    FSIM = temp1(FSIM, FSIM1)
    Complete_Figure_Perf(FSIM, str, xlab, ylab, clr, dataset)
    Complete_Figure_Perf1(FSIM, str, xlab, ylab, clr, dataset)


def AnalysisResults(dataset):

    print(colored(f' Com_Analysis for dataset {dataset} ', color='grey', on_color='on_green'))
    cprint("===========================================", color='blue')

    ComAnalysis_Graph(dataset)
    #
    # print(colored(f' Perf_Analysis for dataset {dataset} ', color='grey', on_color='on_yellow'))
    # cprint("===========================================", color='blue')
    #
    # PerfAnalysis_Graph(dataset)



