

gi_bi={
    'OPT':{
        (0.25,2): {
            '$g_i$ wateram': [49.04, 48.99, 49.16, 48.90],
            '$g_i$ (nl)': [48.69, 48.33, 48.41, 48.34],
            '$g^o_i$ (wm)': [67.80, 67.74, 67.94, 67.49],
            '$g^o_i$ (nl)': [29.45, 29.34, 29.31, 29.33],
            '$b_i$ pro (wm)': [79.52, 76.06, 73.68, 71.44],
            '$b_i$ pro (nl)': [35.72, 35.07, 34.53, 34.16],
            '$b_i$ main (wm)': [66.09, 63.91, 63.17, 60.52],
            '$b_i$ main (nl)': [29.79, 30.62, 30.81, 29.64],
        },
        (0.25,4): {
            '$g_i$ (wm)': [30.65, 30.45, 30.39, 30.31],
            '$g_i$ (nl)': [51.75, 51.24, 51.15, 51.13],
            '$g^o_i$ (wm)': [53.25, 52.92, 52.88, 52.75],
            '$g^o_i$ (nl)': [30.70, 30.35, 30.28, 30.27],
            '$b_i$ pro (wm)': [57.44, 56.36, 55.85, 55.12],
            '$b_i$ pro (nl)': [35.03, 35.01, 35.01, 34.89],
            '$b_i$ main (wm)': [44.49, 42.90, 43.26, 41.94],
            '$b_i$ main (nl)': [17.29, 17.16, 18.24, 17.52],
        },
        (0.5,2): {
            '$g_i$ (wm)': [99.67, 99.77, 99.19, 99.08],
            '$g_i$ (nl)': [75.68, 74.56, 75.03, 74.90],
            '$g^o_i$ (wm)': [124.00, 124.16, 123.06, 122.77],
            '$g^o_i$ (nl)': [67.60, 66.53, 67.00, 66.92],
            '$b_i$ pro (wm)': [129.70, 128.22, 125.87, 124.75],
            '$b_i$ pro (nl)': [65.28, 65.64, 66.54, 67.03],
            '$b_i$ main (wm)': [110.41, 110.81, 103.78, 101.76],
            '$b_i$ main (nl)': [68.45, 70.66, 66.87, 66.46],

        },
        (0.5,4): {
            '$\hat{g}_i$ watermark threshold for watermarked sentences': [91.38, 91.25, 91.68, 91.72],
            '$\\tilde{g}_i$ watermark threshold for natural sentences': [76.29, 76.95, 76.13, 76.17],
            '$\hat{g}^o_i$ ground truth of the number of green tokens for watermarked sentences': [120.37, 120.07, 120.68, 120.67],
            '$\\tilde{g}^o_i$ ground truth of the number of green tokens for natural sentences': [68.05, 68.63, 67.90, 67.96],
            '$\hat{b}_i$ substitution bound for watermarked sentences (Pro-AS1)': [122.70, 121.82, 122.19, 121.89],
            '$\\tilde{b}_i$ substitution bound for natural sentences (Pro-AS1)': [63.92, 66.18, 66.51, 67.29],
            '$\hat{b}_i$ substitution bound for watermarked sentences (Ours-AS2)': [103.73, 103.69, 104.38, 105.19],
            '$\\tilde{b}_i$ substitution bound for natural sentences (Ours-AS2)': [60.95, 62.79, 64.26, 65.81],
        }
    },
    'LLaMA':{
        (0.25,2):{
            '$g_i$ (wm)': [53.78, 53.95, 54.08, 54.14],
            '$g_i$ (nl)': [47.33, 47.36, 47.19, 47.25],
            '$g^o_i$ (wm)': [72.49, 72.93, 73.09, 73.14],
            '$g^o_i$ (nl)': [29.60, 29.65, 29.57, 29.58],
            '$b_i$ pro (wm)': [86.72, 82.80, 79.55, 77.65],
            '$b_i$ pro (nl)': [37.15, 36.06, 34.68, 34.06],
            '$b_i$ main (wm)': [73.81, 73.70, 71.34, 69.94],
            '$b_i$ main (nl)': [34.79, 36.19, 33.94, 34.14],
        },
        (0.25,4):{
            '$g_i$ (wm)': [40.04, 39.68, 39.55, 39.35, ],
            '$g_i$ (nl)': [51.04, 50.33, 50.21, 50.15, ],
            '$g^o_i$ (wm)': [70.66, 70.09, 69.89, 69.46, ],
            '$g^o_i$ (nl)': [31.24, 30.64, 30.49, 30.37, ],
            '$b_i$ pro (wm)': [77.24, 75.40, 74.62, 73.64, ],
            '$b_i$ pro (nl)': [36.23, 35.52, 35.29, 35.02, ],
            '$b_i$ main (wm)': [69.13, 68.03, 66.94, 65.77, ],
            '$b_i$ main (nl)': [25.75, 25.51, 24.99, 24.39, ],
        },
        (0.5,2):{
            '$g_i$ (wm)': [98.59, 99.06, 99.11, 98.95, ],
            '$g_i$ (nl)': [80.30, 79.83, 80.27, 80.02, ],
            '$g^o_i$ (wm)': [119.10, 119.76, 119.81, 119.61, ],
            '$g^o_i$ (nl)': [66.55, 65.99, 66.29, 66.05, ],
            '$b_i$ pro (wm)': [125.99, 124.83, 124.06, 120.48, ],
            '$b_i$ pro (nl)': [68.00, 67.92, 68.52, 67.27, ],
            '$b_i$ main (wm)': [104.96, 105.90, 104.06, 102.97, ],
            '$b_i$ main (nl)': [61.42, 63.24, 62.34, 61.75, ],
        },
        (0.5,4):{
            '$\hat{g}_i$ watermark threshold for watermarked sentences': [86.18, 87.83, 87.94, 87.61, ],
            '$\tilde{g}_i$ watermark threshold for natural sentences': [82.19, 82.55, 82.40, 82.46, ],
            '$\hat{g}^o_i$ ground truth of the number of green tokens for watermarked sentences': [114.30, 116.55, 116.82, 116.38, ],
            '$\tilde{g}^o_i$ ground truth of the number of green tokens for natural sentences': [67.20, 67.60, 67.57, 67.62, ],
            '$\hat{b}_i$ substitution bound for watermarked sentences (Pro-AS1)': [117.48, 119.19, 119.11, 118.14, ],
            '$\tilde{b}_i$ substitution bound for natural sentences (Pro-AS1)': [67.11, 68.10, 68.33, 68.51, ],
            '$\hat{b}_i$ substitution bound for watermarked sentences (Ours-AS2)': [104.02, 107.36, 105.23, 103.95, ],
            '$\tilde{b}_i$ substitution bound for natural sentences (Ours-AS2)': [55.18, 58.00, 56.83, 56.26, ],
        }
    }
}

wr={
    'opt':{
        (0.25,2):{
            '$G^b_{avg}$':67.31,
            '$G^a_{avg}$':[11.53259533, 10.41943419, 7.41697417, 7.745387454, ],
            '$Acc_r$':[0.998769988, 0.998769988, 1, 0.998769988, ],
        },
        (0.25,4):{},
        (0.5,2):{},
        (0.5,4):{},
    },
    'llama':{

    },
}