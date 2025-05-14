import numpy as np
from experiment_log import PytorchExperimentLogger
from arguparse import args_parser

args = args_parser()
exp_logger = PytorchExperimentLogger('./log', "elog_hdc_test2", ShowTerminal=True)
SKIP = args.skip

def lookup_generate(dim, datatype, n_keys):
    p_0, p_1 = 0.5, 0.5
    if datatype == 1:
        # binary
        p = np.array([p_0, p_1])
        base_hv = np.random.choice((False, True), (n_keys, dim), p=p)
        print("base_hv", base_hv)
    elif datatype == 2:
        # bipolar
        p = np.array([p_0, p_1])
        base_hv = np.random.choice((-1, 1), (n_keys, dim), p=p)
        print("base_hv", base_hv)
        base_hv = base_hv.astype(np.int8)

    return base_hv[0]


def encoding3ChannelPos(dim, datatype, img, changed_ratio):
    IMG_HEIGHT = img.shape[0]
    IMG_WIDTH = img.shape[1]
    column = []
    row = []
    column1 = lookup_generate(dim,datatype,1).astype(np.uint8)
    row1 = lookup_generate(dim,datatype,1).astype(np.uint8)
    print("column:", column1)
    print("row:", row1)
    column.append(column1)
    row.append(row1)

    halfDim = int(dim/2)
    changed_num = int(halfDim * changed_ratio)
    row_unchanged_half = row1[changed_num:].copy()  # second half
    unit = int(changed_num / IMG_HEIGHT)

    for i in range(1,IMG_HEIGHT):
        if i % SKIP != 0:
            row_i = row[i - 1].copy()
            row.append(row_i)

        else:
            row_i = row[i - 1].copy()
            row_i = row_i[i * unit: (i + 1) * unit]
            row_i = 1 - row_i
            row_iM1 = row[i - 1].copy()
            row_iM1 = row_iM1[:changed_num]
            row_i = np.concatenate((row_iM1[:i * unit], row_i, row_iM1[(i + 1) * unit:]))
            row_i = np.concatenate((row_i, row_unchanged_half))
            row.append(row_i)

    unit = int(changed_num / IMG_WIDTH)
    column_unchanged_half = column1[:halfDim].copy()
    column_unchanged_Second_half = column1[halfDim + changed_num: dim].copy()
    for i in range(1, IMG_WIDTH):
        if i % SKIP != 0:
            column_i = column[i - 1].copy()
            column.append(column_i)
        else:
            column_i = column[i - 1].copy()
            column_i = column_i[halfDim: halfDim + changed_num]  # change second half
            column_i = column_i[i * unit: (i + 1) * unit]
            column_i = 1 - column_i
            column_iM1 = column[i - 1].copy()
            column_iM1 = column_iM1[halfDim: halfDim + changed_num]  # concatenate the rest part
            column_i = np.concatenate((column_iM1[:i * unit], column_i, column_iM1[(i + 1) * unit:]))
            column_i = np.concatenate((column_unchanged_half, column_i, column_unchanged_Second_half))
            column.append(column_i)

    pos = []
    for i in range(IMG_HEIGHT):
        row_all = []
        for j in range(IMG_WIDTH):
            tmp = np.logical_xor(row[i], column[j])
            row_all.append(tmp)

        pos.append(row_all)
    pos = np.asarray(pos).astype(np.uint8)
    print("pos",pos.shape)
    return pos

def encoding3ChannelColor(dimension, datatype, sample):
    IMG_HEIGHT = sample.shape[0]
    IMG_WIDTH = sample.shape[1]
    sample1 = sample[:, :, 0:1].reshape(IMG_HEIGHT, IMG_WIDTH)
    sample2 = sample[:, :, 1:2].reshape(IMG_HEIGHT, IMG_WIDTH)
    sample3 = sample[:, :, 2:3].reshape(IMG_HEIGHT, IMG_WIDTH)
    dim = int(dimension/3)
    grayscale_table_3Channel = []
    for j in range(3):
        if j % 3 == 2:
            dim = dimension - 2 * dim
        gray0 = lookup_generate(dim, datatype, 1)
        grayscale_table = []
        grayscale_table.append(gray0)
        unit2 = int(dim / 256)
        for i in range(1, 256):
            gray_i = grayscale_table[i - 1].copy()
            gray_i = gray_i[i * unit2: (i + 1) * unit2]
            gray_i = 1 - gray_i
            gray_i = np.concatenate((grayscale_table[i - 1][:i * unit2], gray_i, grayscale_table[i - 1][(i + 1) * unit2:]))
            grayscale_table.append(gray_i)

        grayscale_table = np.array(grayscale_table).astype(np.uint8)
        grayscale_table_3Channel.append(grayscale_table)

    grayscale_table_result = []

    for idx_row in range(IMG_HEIGHT):
        grayscale_row_result = []
        for idx_col in range(IMG_WIDTH):
            gray_val1 = sample1[idx_row][idx_col]
            gray_val2 = sample2[idx_row][idx_col]
            gray_val3 = sample3[idx_row][idx_col]
            tmp = np.concatenate((grayscale_table_3Channel[0][gray_val1], grayscale_table_3Channel[1][gray_val2], \
                                  grayscale_table_3Channel[2][gray_val3])).astype(np.uint8)
            grayscale_row_result.append(tmp)
        grayscale_table_result.append(grayscale_row_result)
    grayscale_table_result = np.asarray(grayscale_table_result).astype(np.uint8)
    return grayscale_table_result

def encoding_3Chanenel(dim, datatype, img, changed_ratio):
    pos = encoding3ChannelPos(dim, datatype, img, changed_ratio)
    grayscale_table = encoding3ChannelColor(dim, datatype, img)
    IMG_HEIGHT = img.shape[0]
    IMG_WIDTH = img.shape[1]
    encoded_img = []
    for i in range(IMG_HEIGHT):
        row_encoded = []
        for j in range(IMG_WIDTH):
            pix_HV = np.logical_xor(grayscale_table[i][j], pos[i][j])
            row_encoded.append(pix_HV)
        encoded_img.append(row_encoded)

    encoded_img = np.array(encoded_img).astype(np.uint8)
    return encoded_img

