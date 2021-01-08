# uncompyle6 version 3.4.0
# Python bytecode 2.7 (62211)
# Decompiled from: Python 3.7.3 (default, Mar 27 2019, 09:23:15)
# [Clang 10.0.1 (clang-1001.0.46.3)]
# Embedded file name: Parastitcher3.py
# Compiled at: 2018-09-06 04:29:28
"""
This program uses a main subordinate approach to consume a queue 
of elaborations using teraconverter
Copyright (c) 2016:
Massimiliano Guarrasi (1), Giulio Iannello (2), Alessandro Bria (2)
(1): CINECA
(2): University Campus Bio-Medico of Rome
The program was made in the framework of the HUMAN BRAIN PROJECT.
All rights reserved.

EXAMPLE of usage (X is the major version, Y is the minor version, Z is the patch):

For align step:
mpirun -np XX python ParastitcherX.Y.Z.py -2 --projin=xml_import_file --projout=xml_displcomp_file [--sV=VV] [--sH=HH] [--sD=DD] [--imin_channel=C] [ ... ] 

where:
- XX is the desided level of parallelism plus 1 (for the main process)
- VV, HH, DD are the half size of the NCC map along V, H, and D directions, respectively
- C is the input channel to be used for align computation

For fusion step:
mpirun -np XX python ParastitcherX.Y.Z.py -6 --projin=xml_import_file --volout=destination_folder --volout_plugin=format_string [--slicewidth=WWW] [--sliceheight=HHH] [--slicedepth=DDD] [--resolutions=RRR] [ ... ] 

where:
- format_string is one the formats: "TIFF (series, 2D)", "TIFF, tiled, 2D", "TIFF, tiled, 3D", "TIFF, tiled, 4D", 
- DDD, HHH, WWW are the values used to partition the image for parallel execution
- RRR are the requested resolutions (according to the convention used by teraconverter)
See teraconverter documentation for more details

*******************************
*        Change Log           *
*******************************

2018-09-05. Giulio. @CHSNGED on non-Windows platforms 'prefix' is automatically switched to './' if executables are not in the system path
2018-08-16. Giulio. @CHANGED command line interface: parameters for step 6 are the same than the sequential implementation
2018-08-16. Giulio. @ADDED debug control
2018-08-07. Giulio. @CREATED from parastitcher2.0.3.py and paraconverter2.3.2.py

*******************************

terastitcher -2 --projin=/Users/iannello/Home/Windows/myTeraStitcher/TestData/Ailey/blending/test_00_01_02_03/xml_import_org.xml --projout=/Users/iannello/Home/Windows/myTeraStitcher/TestData/Ailey/blending/test_00_01_02_03/xml_displcomp_seq.xml

mpirun -np 3 python /Users/iannello/Home/Windows/paratools/parastitcher2.0.3.py -2 --projin=/Users/iannello/Home/Windows/myTeraStitcher/TestData/Ailey/blending/test_00_01_02_03/xml_import_org.xml --projout=/Users/iannello/Home/Windows/myTeraStitcher/TestData/Ailey/blending/test_00_01_02_03/xml_displcomp_par2.xml
mpirun -np 3 python /Users/iannello/Home/Windows/paratools/Parastitcher3.0.0.py -2 --projin=/Users/iannello/Home/Windows/myTeraStitcher/TestData/Ailey/blending/test_00_01_02_03/xml_import_org.xml --projout=/Users/iannello/Home/Windows/myTeraStitcher/TestData/Ailey/blending/test_00_01_02_03/xml_displcomp_par2.xml

teraconverter --sfmt="TIFF (unstitched, 3D)" -s=/Users/iannello/Home/Windows/myTeraStitcher/TestData/Ailey/blending/test_00_01_02_03/xml_merging.xml --dfmt="TIFF (series, 2D)" -d=/Users/iannello/Home/Windows/myTeraStitcher/TestData/temp/result_p1 --resolutions=012 --depth=256 --width=256 --height=256

mpirun -np 3 python /Users/iannello/Home/Windows/paratools/paraconverter2.3.2.py --sfmt="TIFF (unstitched, 3D)" -s=/Users/iannello/Home/Windows/myTeraStitcher/TestData/Ailey/blending/test_00_01_02_03/xml_merging.xml --dfmt="TIFF (tiled, 3D)" -d=/Users/iannello/Home/Windows/myTeraStitcher/TestData/temp/result_p1 --resolutions=012 --depth=256 --width=256 --height=256
mpirun -np 3 python /Users/iannello/Home/Windows/paratools/Parastitcher3.0.0.py -6 --sfmt="TIFF (unstitched, 3D)" -s=/Users/iannello/Home/Windows/myTeraStitcher/TestData/Ailey/blending/test_00_01_02_03/xml_merging.xml --dfmt="TIFF (tiled, 3D)" -d=/Users/iannello/Home/Windows/myTeraStitcher/TestData/temp/result_p1 --resolutions=012 --depth=256 --width=256 --height=256

"""
import os, sys, shutil, time, datetime, operator, platform
from math import *
from glob import glob
from mpi4py import MPI
from collections import deque
from subprocess import *
import os.path, pickle

prefix = ""
debug_level = 0
default_tile_size = 256
resume_status_fname = "para_resume_status.bin"
suspend_resume_enabled = False
save_status_prefix = ""


def partition(m, n, N):
    """
    return the number of partitions along V and H, respectively that are optimal to
    partition a block of size m_V x n_H in at least P sub-blocks

    m: block size along V
    n: block size along H
    N:   number of required partitions

    return:
    p_m, p_n: the  number of partitions along V and H, respectively

    PRE:
    """
    m = float(m)
    n = float(n)
    N = float(N)
    p_m_min = sqrt(N * m / n)
    p_n_min = sqrt(N * n / m)
    c_min = m * p_n_min + n * p_m_min
    p_m = ceil(p_m_min)
    p_n = floor(p_n_min)
    if p_n * p_m < N:
        p_m = floor(p_m_min)
        p_n = ceil(p_n_min)
        if p_n * p_m < N:
            p_m = ceil(p_m_min)
    if 1 <= p_m and p_m <= m / 2.0 and 1 <= p_n and p_n <= n / 2.0:
        p_m0 = p_m
        c = m * p_n + n * p_m
    else:
        return (-1, -1)
    p_m_cur = p_m + 1
    p_n_cur = (c - n * p_m_cur) / m
    while p_m_cur <= m / 2.0 and p_m_cur * p_n_cur >= N:
        if 1 <= floor(p_n_cur) and p_m_cur * floor(p_n_cur) >= N:
            p_m = p_m_cur
            p_n = floor(p_n_cur)
            c = m * p_n + n * p_m
        p_m_cur = p_m_cur + 1
        p_n_cur = (c - n * p_m_cur) / m

    p_m_cur = p_m0 - 1
    p_n_cur = (c - n * p_m_cur) / m
    while floor(p_n_cur) <= p_m_cur and p_m_cur * p_n_cur >= N:
        if floor(p_n_cur) <= n / 2.0 and p_m_cur * floor(p_n_cur) >= N:
            p_m = p_m_cur
            p_n = floor(p_n_cur)
            c = m * p_n + n * p_m
        p_m_cur = p_m_cur - 1
        p_n_cur = (c - n * p_m_cur) / m

    return (int(p_m), int(p_n))


def extract_params():
    """
    Extract parameter from line of commands.
    Output:
       params = list of parameters from original command line
    """
    params = sys.argv
    return params


def check_flag(params, string, delete):
    """
    Check if a parameter (string) was beeen declared in the line of commands (params) and return the associated value.
    If delete is true the related string will be deleted
    If string is not present, return None
    Input:
       params = list of parameters from original command line
       string = string to be searched
       delete = Boolean variable to check if the selected string must be deleted after copied in value variable
    Output:
       value = parameter associated to the selected string
    """
    i = 0
    value = None
    size = len(string)
    for line in params:
        tmp = line.find(string)
        if tmp != -1:
            start = tmp + size
            sel_string = line[start:]
            if delete:
                params.pop(i)
            value = sel_string
        i += 1

    return value


def pop_left(dictionary):
    """
    Cuts the first element of dictionary and returns its first element (key:value)
    Input/Output:
      dictionary = Dictionary of string containing the command lines to use. After reading the dictionary the first element is deleted from the dictionary.
    Output:
      first_el = first element (values) of the dictionary
    """
    if len(dictionary) > 0:
        first_el = {list(dictionary.keys())[0]: list(dictionary.values())[0]}
        dictionary.pop(list(dictionary.keys())[0])
    else:
        first_el = None
    return first_el


def worker(input_file):
    """
    Perform elaboration for each element of the queue.
    Input/Output
       input_file = command to be executed
    """
    myrank = comm.Get_rank()
    t1 = time.time()
    print(
        (
            "Scheduled job n. ",
            list(input_file.keys())[0],
            " is executed by rank: ",
            myrank,
        )
    )
    if debug_level > 0:
        execution_string = (
            prefix
            + list(input_file.values())[0]
            + " > "
            + "output_"
            + str(list(input_file.keys())[0])
            + ".out"
        )
    else:
        execution_string = prefix + list(input_file.values())[0]
    print(execution_string)
    os.system(execution_string)
    t2 = time.time()
    print((" ---> Processor ", myrank, " has calculated for ", t2 - t1))
    return input_file


def subordinate():
    """
    Subordinate process.
    """
    myrank = comm.Get_rank()
    WORKTAG = 1
    DIETAG = 2
    while True:
        status = MPI.Status()
        input_name = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.tag == DIETAG:
            end_signal = ["Exit cpu n. ", myrank]
            print((end_signal[0], end_signal[1]))
            comm.send(end_signal, dest=0, tag=1)
            return
        result = worker(input_name)
        comm.send(result, dest=0, tag=0)


def read_input(inputf, nline=0):
    """
    Reads the file included in inputf at least up to line number nline (if declared).

    """
    i = 0
    data = []
    f = open(inputf, "r")
    for line in f:
        line = line.strip()
        l = line.split(" ", 1)
        data.append(l)
        if nline != 0 and i > nline:
            break
        i += 1

    f.close()
    return data


def extract_np(inputf):
    """
    extract the number of slices along z from the input xml file.

    """
    data = read_input(inputf, 8)
    for tmp_string in data:
        if tmp_string[1].find('stack_slices="') != -1:
            start = tmp_string[1].find('stack_slices="') + 14
            break

    end = tmp_string[1].find('" />')
    sel_string = tmp_string[1][start:end]
    nslices = int(sel_string)
    start = tmp_string[1].find('stack_rows="') + 12
    end = start + tmp_string[1][start:].find('"')
    sel_string = tmp_string[1][start:end]
    nrows = int(sel_string)
    start = tmp_string[1].find('stack_columns="') + 15
    end = start + tmp_string[1][start:].find('"')
    sel_string = tmp_string[1][start:end]
    ncols = int(sel_string)
    return (nrows, ncols, nslices)


def find_last_slash(string):
    """
    Search for / in a string. If one or more / was found, divide the string in a list of two string:
    the first containf all the character at left of the last / (included),
    and the second contains the remanent part of the text.
    If no / was found, the first element of the list will be set to ''
    """
    len_string = len(string)
    check = 0
    index = []
    i = 0
    for chara in string:
        if chara == "/" or chara == "\\":
            index.append(i)
            check = 1
        i += 1

    if check == 1:
        last_slash = max(index)
        output_string = [string[0 : last_slash + 1], string[last_slash + 1 :]]
    else:
        output_string = ["", string]
    return output_string


def add_chars(params):
    string = ["volin_plugin=", "imin_plugin"]
    i = 0
    for line in params:
        for local_string in string:
            tmp = line.find(local_string)
            if tmp != -1:
                size = len(local_string)
                sel_string = line.split("=")
                input_string = sel_string[1]
                mod_string = '"' + input_string + '"'
                tot_mod_string = sel_string[0] + "=" + mod_string
                params[i] = tot_mod_string

        i += 1

    return params


def main_step2(queue):
    """
    dispatch the work among processors

    queue is a list of job input

    """
    WORKTAG = 1
    DIETAG = 2
    nprocs = comm.Get_size()
    for rank in range(1, min(len(queue) + 1, nprocs)):
        input_file = pop_left(queue)
        comm.send(input_file, dest=rank, tag=WORKTAG)

    while queue:
        input_file = pop_left(queue)
        status = MPI.Status()
        flag = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        comm.send(input_file, dest=status.source, tag=WORKTAG)

    status = MPI.Status()
    flag = comm.recv(source=MPI.ANY_SOURCE, tag=0, status=status)
    for rank in range(1, nprocs):
        comm.send(0, dest=rank, tag=DIETAG)

    for rank in range(1, nprocs):
        exit_m = comm.recv(source=rank, tag=1, status=status)


def do_additional_partition(nprocs, nrows, ncols, n_ss):
    """
    All parameters should be float
    """
    print(
        (
            "do_additional_partition ( nprocs =",
            nprocs,
            ", nrows =",
            nrows,
            ", ncols =",
            ncols,
            ", nlayers =",
            n_ss,
        )
    )
    if n_ss >= 2 * nprocs:
        return (1, 1)
    else:
        if floor(nrows / 2) * floor(ncols / 2) * n_ss < 2 * nprocs:
            print(("WARNING: not enough paritions for", nprocs, "processors"))
            print(
                (
                    "Dataset partitioned in",
                    int(floor(nrows / 2)),
                    "x",
                    int(floor(ncols / 2)),
                    "x",
                    n_ss,
                    "=",
                    int(floor(nrows / 2)) * int(floor(ncols / 2)) * n_ss,
                    "partitions",
                )
            )
            return (max(1, floor(nrows / 2)), max(1, floor(ncols / 2)))
        m = max(nrows, ncols)
        n = min(nrows, ncols)
        p_m, p_n = partition(m, n, ceil(2 * nprocs / n_ss))
        if nrows < ncols:
            temp = p_m
            p_m = p_n
            p_n = temp
        return (p_m, p_n)


def score_function(params):
    """
    Assigns a score value with the formula:
          score = 100*N_of_voxel/max(N_of_voxel)
    Input:
       params =  dictionary containing {input_name : [Nx,Ny,Nz]}
    Output:
       scores = dictionary containing {input_name : score}
    """
    tmp_scores = {}
    scores = {}
    imp_key = list(params.keys())
    for i in imp_key:
        tmp = params[i]
        npoints = tmp[0] * tmp[1] * tmp[2]
        tmp_scores[i] = npoints

    den = max(tmp_scores.values())
    for i in tmp_scores:
        scores[i] = 100.0 * tmp_scores[i] / den

    return scores


def sort_elaborations(scores):
    """
    Create a list of input_name sorted by score
    Input:
      scores = dictionary of the form  {input_name : score}
    Output:
      scored = a list of input_name sorted by score
    """
    scored = sorted(scores, key=scores.__getitem__, reverse=True)
    return scored


def sort_work(params, priority):
    """
    Returns a dictionary as params but ordered by score
    Input:
       params = dictionary of the form  {input_name : value}
       priority = the list of input_name ordered by score calculated by score_function
    Output:
       sorted_dict = the same dictionary as params but ordered by score
    """
    sorted_dict = {}
    i = 0
    for index in priority:
        sorted_dict.update({i: params[index]})
        i = i + 1

    return sorted_dict


def main_step6(queue, rs_fname):
    """
    Dispatch the work among processors.
    Input:
       queue = list of job inputs
    """
    n_tasks = len(queue)
    if suspend_resume_enabled:
        rs_file = open(rs_fname, "rb")
        done = pickle.load(rs_file)
        rs_file.close()
    else:
        done = []
    WORKTAG = 1
    DIETAG = 2
    nprocs = comm.Get_size()
    for rank in range(1, min(len(queue) + 1, nprocs)):
        input_file = pop_left(queue)
        while input_file != None and list(input_file.keys())[0] in done:
            input_file = pop_left(queue)

        if input_file == None:
            break
        comm.send(input_file, dest=rank, tag=WORKTAG)

    print("MASTER: first loop terminated")
    while queue:
        input_file = pop_left(queue)
        while input_file != None and list(input_file.keys())[0] in done:
            input_file = pop_left(queue)

        if input_file == None:
            break
        status = MPI.Status()
        flag = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        assert (
            status.tag == 0
        ), "Wrong tag: a message signalling a finished task expected"
        done.append(list(flag.keys())[0])
        if suspend_resume_enabled:
            rs_file = open(rs_fname, "wb")
            pickle.dump(done, rs_file)
            rs_file.close()
        comm.send(input_file, dest=status.source, tag=WORKTAG)

    print("MASTER: second loop terminated")
    while len(done) < n_tasks:
        status = MPI.Status()
        flag = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        assert (
            status.tag == 0
        ), "Wrong tag: a message signalling a finished task expected"
        done.append(list(flag.keys())[0])
        if suspend_resume_enabled:
            rs_file = open(rs_fname, "wb")
            pickle.dump(done, rs_file)
            rs_file.close()

    print("MASTER: third loop terminated")
    status = MPI.Status()
    for rank in range(1, nprocs):
        comm.send(0, dest=rank, tag=DIETAG)

    for rank in range(1, nprocs):
        exit_m = comm.recv(source=rank, tag=1, status=status)

    if suspend_resume_enabled:
        os.remove(rs_fname)
    return


def read_params():
    """
    Read parameters from input string and from a file
    Input:
    Output:
       input_name = Input file
       output_name = Standard output directory
       wb1 = Axprossimative depth for the tiles
       wb2 = Axprossimative height for the tiles
       wb3 = Axprossimative width for the tiles
       sfmt = Source format
       dfmt = Destination format
       iresolutions = List of integer values containing all the desidered values for level of resolution
       max_res = Maximum level of resolution available (integer)
       params = Array containing instruction derived from the remanent part of the imput string
       last_string = Remanent part of the input string
       height = Height of the input immage
       width = Width of the input immage
       depth = Depth of the input immage
    """
    params = extract_params()
    params.pop(0)
    params = check_double_quote(params)
    input_name = read_item(params, "-projin=", "./input.xml")
    info_string = (
        "teraconverter "
        + ' --sfmt="TIFF (unstitched, 3D)"'
        + ' --dfmt="TIFF (tiled, 3D)"'
        + ' -s="'
        + input_name
        + '" -d=/'
    )
    execution_string = (
        prefix + info_string + ' --info="' + os.getcwd() + '/__dims__.txt"'
    )
    os.system(execution_string)
    print(execution_string)
    file_in = read_item(params, "--origin=", os.getcwd() + "/__dims__.txt")
    print(("Origin file is: ", file_in))
    input_params_in_file = [
        "HEIGHT=",
        "WIDTH=",
        "DEPTH=",
        "BYTESxCHAN=",
        "DIM_C=",
        "VXL_V=",
        "VXL_H=",
        "VXL_D=",
    ]
    params_from_file = search_for_entry(input_params_in_file, file_in)
    os.remove(os.getcwd() + "/__dims__.txt")
    height = int(params_from_file[0])
    width = int(params_from_file[1])
    depth = int(params_from_file[2])
    bytes_x_chan = int(params_from_file[3])
    n_chans = int(params_from_file[4])
    vxl_V = abs(float(params_from_file[5]))
    vxl_H = abs(float(params_from_file[6]))
    vxl_D = abs(float(params_from_file[7]))
    output_name = read_item(params, "-volout=", "./OUT")
    wb1 = read_item(params, "--slicedepth=", 0)
    wb2 = read_item(params, "--sliceheight=", 0)
    wb3 = read_item(params, "--slicewidth=", 0)
    sfmt = '"TIFF (unstitched, 3D)"'
    volout_plugin = read_item(params, "--volout_plugin=", '"TiledXY|2Dseries"')
    if volout_plugin == "TiledXY|3Dseries":
        dfmt = '"TIFF (tiled, 3D)"'
    else:
        dfmt = '"TIFF (tiled, 2D)"'
    resolutions = read_item(params, "--resolutions=", "0")
    iresolutions = [int(resolutions[0])]
    len_res = len(resolutions)
    if len_res > 0:
        for i in range(1, len_res):
            iresolutions.append(int(resolutions[i]))

    max_res = max(iresolutions)
    isotropic = read_item(params, "--isotropic", "False")
    if isotropic == "":
        isotropic = True
    else:
        isotropic = False
    if isotropic:
        vxlsz_Vx2 = 2 * vxl_V
        vxlsz_Hx2 = 2 * vxl_H
        vxlsz_D = vxl_D
        h = 0
        while h < max_res and max(vxlsz_Vx2, vxlsz_Hx2) < vxlsz_D:
            h += 1
            vxlsz_Vx2 *= 2
            vxlsz_Hx2 *= 2

    else:
        h = 0
    max_res_D = max_res - h
    print(
        (
            "vxl_V, vxl_H, vxl_D, isotropic, h, max_res, max_res_D :",
            vxl_V,
            vxl_H,
            vxl_D,
            isotropic,
            h,
            max_res,
            max_res_D,
        )
    )
    last_string = collect_instructions(params)
    return (
        input_name,
        output_name,
        wb1,
        wb2,
        wb3,
        sfmt,
        dfmt,
        iresolutions,
        max_res,
        isotropic,
        max_res_D,
        params,
        last_string,
        height,
        width,
        depth,
        bytes_x_chan,
        n_chans,
    )


def read_item(input_arr, item, default, message=True):
    """
    Read the value related to "item" from the list "input_arr" and if no item are present set it to "default".
    Please note: The function convert the output to the same type of "default" variable
    Input:
       input_arr = List of strings from imput command line
       item = The item to search
       default = The default value if no item are present
    Output:
       value = Output value for the selected item
    """
    tmp = check_flag(input_arr, item, True)
    if tmp == None:
        value = default
        if message:
            print(
                (
                    "The value for ",
                    item,
                    " was not declared. It will be set to",
                    value,
                    "by default.",
                )
            )
    elif isinstance(default, int):
        value = int(tmp)
    elif isinstance(default, float):
        value = float(tmp)
    else:
        value = tmp
    return value


def collect_instructions(inst):
    """
    Collect the remanent part of a list of strings in a unique string
    Input:
      inst = Input list of strings
    Output:
      results = String containing all the elements of inst
    """
    len_inst = len(inst)
    if len_inst > 0:
        for i in range(0, len_inst):
            if i == 0:
                results = str(inst[i])
            else:
                results = results + " " + str(inst[i])

    else:
        results = ""
    return results


def search_for_entry(string_2_serch, file_in, nline=0):
    """
    Extract from the input file (file_in) up to the line number nline (if declared) the value assigned to string_2_serch.
    Input:
      string_2_serch = string (or list of string) containing the variable to search (e.g. 'HEIGHT=')
      file_in = name of the file containing the information we neeed (e.g: prova.txt or /pico/home/prova.txt)
      nline = optional, number of the final row of the file we need to analyze
    Output:
      Output = value or (list of values) assigned to the variable conteined in string_2_serch
    """
    i = 0
    data = []
    f = open(file_in, "r")
    for line in f:
        line = line.strip()
        l = line.split(" ", 1)
        data = data + l
        if nline != 0 and i > nline:
            break
        i += 1

    f.close()
    len_string = len(string_2_serch)
    if len_string <= 0:
        print("No possible options! No values will be created!")
    elif len_string == 1:
        tmp = check_flag(data, string_2_serch[0], True)
        if tmp == None:
            output = "0"
            print(
                (
                    "The name of ",
                    string_2_serch,
                    " was not declared. It will be set to",
                    output,
                    "by default.",
                )
            )
        else:
            output = tmp
    elif len_string > 1:
        ii = 0
        output = []
        for i in string_2_serch:
            tmp = check_flag(data, i, True)
            if tmp == None:
                output.append("0")
                print(
                    (
                        "The name of ",
                        i,
                        " was not declared. It will be set to",
                        output[ii],
                        "by default.",
                    )
                )
            else:
                output.append(tmp)
            ii = ii + 1

    else:
        print("No possible options! No values will be created!")
    return output


def sort_list(len_1, len_2, len_3):
    """
    Create a list sorting the indexes along three directions:
    Input:
       len_1 = Number of elements of the array for the first index
       len_2 = Number of elements of the array for the second index
       len_3 = Number of elements of the array for the third index
    Output:
       order = An ordered list containig an a sequence of lists of 3 alements (one for each direction) that identify the position on the local index
    """
    order = []
    for i in range(0, len_1):
        for j in range(0, len_2):
            for k in range(0, len_3):
                order.append([i, j, k])

    return order


def sort_start_end(
    start_1, start_2, start_3, end_1, end_2, end_3, size_1, size_2, size_3
):
    """
    Sort start points and edn point in two lists of elements
    Input:
       start_1 = Array containing all the starting indexes for the tiles on the Depth direction
       start_2 = Array containing all the starting indexes for the tiles on the Height direction
       start_3 = Array containing all the starting indexes for the tiles on the Width direction
       end_1 = Array containing all the ending indexes for the tiles on the Depth direction
       end_2 = Array containing all the ending indexes for the tiles on the Height direction
       end_3 = Array containing all the ending indexes for the tiles on the Width direction
       size_1 = Array containing the size of the tile in the Depth direction
       size_2 = Array containing the size of the tile in the Height direction
       size_3 = Array containing the size of the tile in the Width direction
    Output:
       order = An ordered list containig an a sequence of lists of 3 alements (one for each direction) that identify the position on the local index
       start_list = Ordered list of lists of starting points. E.g.: [[width_in[0], height_in[0], depth_in[0]], [width_in[1], height_in[1], depth_in[1]], ... ,[width_in[N], height_in[N], depth_in[N]]]
       end_list = Ordered list of lists of starting points. E.g.: [[width_fin[0], height_fin[0], depth_in[0]], [width_fin[1], height_fin[1], depth_fin[1]], ... ,[width_fin[N], height_fin[N], depth_fin[N]]]
       len_arr = Dictionary containing elements like {index:[size_width(i),size_height(i),size_depth(i)],.....}
    """
    len_1 = len(start_1)
    len_2 = len(start_2)
    len_3 = len(start_3)
    order = sort_list(len_1, len_2, len_3)
    len_list = len(order)
    start_list = []
    end_list = []
    len_arr = {}
    for i in range(0, len_list):
        tmp = [start_3[order[i][2]], start_2[order[i][1]], start_1[order[i][0]]]
        start_list.append(tmp)
        tmp = [end_3[order[i][2]], end_2[order[i][1]], end_1[order[i][0]]]
        end_list.append(tmp)
        tmp = [size_3[order[i][2]], size_2[order[i][1]], size_1[order[i][0]]]
        len_arr.update({i: tmp})

    return (order, start_list, end_list, len_arr)


def check_double_quote(inpstring):
    """
    Check if some strings needs of a double quote (if some space are inside the string, it will need to be inside two double quote). E.g.: --sfmt="TIFF (unstitched, 3D)"
    Input:
       inpstring: input string or array of strings
    Output:
       newstring = new string (or array of strings) corrected by quoting if necessary
    """
    if type(inpstring) == list:
        newstring = []
        for index in inpstring:
            tmp1 = index.find(" ")
            if tmp1 != -1:
                tmp2 = index.find('"')
                if tmp2 == -1:
                    dummy = index.find("=")
                    if dummy != -1:
                        newstring.append(
                            index[0 : dummy + 1] + '"' + index[dummy + 1 :] + '"'
                        )
                    else:
                        newstring.append('"' + index + '"')
                else:
                    newstring.append(index)
            else:
                newstring.append(index)

    else:
        tmp1 = inpstring.find(" ")
        if tmp1 != -1:
            tmp2 = inpstring.find('"')
            if tmp2 == -1:
                dummy = inpstring.find("=")
                if dummy != -1:
                    newstring = (
                        inpstring[0 : dummy + 1] + '"' + inpstring[dummy + 1 :] + '"'
                    )
                else:
                    newstring = '"' + inpstring + '"'
            else:
                newstring = inpstring
        else:
            newstring = inpstring
    return newstring


def eliminate_double_quote(inpstring):
    """
    Check if the string is already enclosed by quotes
    Input:
       inpstring: input string or array of strings
    Output:
       newstring = new string (or array of strings) corrected by eliminating enclosing quotes if any
    """
    len_str = len(inpstring)
    if (
        inpstring[0] == '"'
        and inpstring[(len_str - 1)] == '"'
        or inpstring[0] == "'"
        and inpstring[(len_str - 1)] == "'"
    ):
        newstring = inpstring[1 : len_str - 1]
    return newstring


def generate_first_command(
    input_name,
    output_name,
    wb1,
    wb2,
    wb3,
    sfmt,
    dfmt,
    iresolutions,
    max_res,
    params,
    last_string,
):
    """
    Generate first command line
    Input:
       input_name = Input file
       output_name = Standard output directory
       wb1 = Axprossimative depth for the tiles
       wb2 = Axprossimative height for the tiles
       wb3 = Axprossimative width for the tiles
       sfmt = Source format
       dfmt = Destination format
       iresolutions = List of integer values containing all the desidered values for level of resolution
       max_res = Maximum level of resolution available (integer)
       params = Array containing instruction derived from the remanent part of the imput string
       last_string = Remanent part of the input string
    Output:
       first_string = Command line to preprocess the data
    """
    first_string = "teraconverter " + "--height=" + str(wb2) + " --width=" + str(wb3)
    first_string = (
        first_string + " --depth=" + str(wb1) + " --sfmt=" + sfmt + " --dfmt=" + dfmt
    )
    tmp_res = ""
    for i in iresolutions:
        tmp_res = tmp_res + str(i)

    first_string = (
        first_string
        + " --resolutions="
        + tmp_res
        + ' -s="'
        + input_name
        + '" -d="'
        + output_name
        + '" '
    )
    if last_string != []:
        first_string = first_string + last_string
    first_string = first_string + " --makedirs"
    return first_string


def generate_final_command(
    input_name,
    output_name,
    wb1,
    wb2,
    wb3,
    sfmt,
    dfmt,
    iresolutions,
    max_res,
    params,
    last_string,
):
    """
    Generate last command line to merge metadata
    Input:
       input_name = Input file
       output_name = Standard output directory
       wb1 = Axprossimative depth for the tiles
       wb2 = Axprossimative height for the tiles
       wb3 = Axprossimative width for the tiles
       sfmt = Source format
       dfmt = Destination format
       iresolutions = List of integer values containing all the desidered values for level of resolution
       max_res = Maximum level of resolution available (integer)
       params = Array containing instruction derived from the remanent part of the imput string
       last_string = Remanent part of the input string
    Output:
       final_string = Command line to merge metadata
    """
    final_string = "teraconverter " + "--height=" + str(wb2) + " --width=" + str(wb3)
    final_string = (
        final_string + " --depth=" + str(wb1) + " --sfmt=" + sfmt + " --dfmt=" + dfmt
    )
    tmp_res = ""
    for i in iresolutions:
        tmp_res = tmp_res + str(i)

    final_string = (
        final_string
        + " --resolutions="
        + tmp_res
        + ' -s="'
        + input_name
        + '" -d="'
        + output_name
        + '" '
    )
    if last_string != []:
        final_string = final_string + last_string
    final_string = final_string + " --metadata"
    return final_string


def generate_parallel_command(
    start_list,
    end_list,
    input_name,
    output_name,
    wb1,
    wb2,
    wb3,
    sfmt,
    dfmt,
    iresolutions,
    max_res,
    params,
    last_string,
):
    """
    Generate the list of parallel command lines
    Input:
       start_list = Ordered list of lists of starting points. E.g.: [[width_in[0], height_in[0], depth_in[0]], [width_in[1], height_in[1], depth_in[1]], ... ,[width_in[N], height_in[N], depth_in[N]]]
       end_list = Ordered list of lists of starting points. E.g.: [[width_fin[0], height_fin[0], depth_in[0]], [width_fin[1], height_fin[1], depth_fin[1]], ... ,[width_fin[N], height_fin[N], depth_fin[N]]]
       input_name = Input file
       output_name = Standard output directory
       wb1 = Axprossimative depth for the tiles
       wb2 = Axprossimative height for the tiles
       wb3 = Axprossimative width for the tiles
       sfmt = Source format
       dfmt = Destination format
       iresolutions = List of integer values containing all the desidered values for level of resolution
       max_res = Maximum level of resolution available (integer)
       params = Array containing instruction derived from the remanent part of the imput string
       last_string = Remanent part of the input string
    Output:
       list_string = Dictionary of strings containing the command lines to process the data. E.G.: {i:command[i]}
    """
    index = len(start_list)
    list_string = {}
    for i in range(0, index):
        dummy = ""
        dummy = "teraconverter " + "--height=" + str(wb2) + " --width=" + str(wb3)
        dummy = dummy + " --depth=" + str(wb1) + " --sfmt=" + sfmt + " --dfmt=" + dfmt
        tmp_res = ""
        for j in iresolutions:
            tmp_res = tmp_res + str(j)

        dummy = (
            dummy
            + " --resolutions="
            + tmp_res
            + ' -s="'
            + input_name
            + '" -d="'
            + output_name
            + '" '
        )
        if last_string != []:
            dummy = dummy + last_string
        dummy = dummy + " --parallel"
        dummy = (
            dummy + " --H0=" + str(start_list[i][0]) + " --H1=" + str(end_list[i][0])
        )
        dummy = (
            dummy + " --V0=" + str(start_list[i][1]) + " --V1=" + str(end_list[i][1])
        )
        dummy = (
            dummy + " --D0=" + str(start_list[i][2]) + " --D1=" + str(end_list[i][2])
        )
        list_string.update({i: dummy})

    return list_string


def opt_algo(D, w, n):
    """
    Solves the tiling problem
    patitioning the interval [0, D-1] into k subintervals of size
    2^n b and one final subinterval of size r = D - k 2^n b
    Input:
      D = dimension of the original array
      w = approximate estimation of value for b
      n = desideres level of refinement (e.g. : n = 0 => maximum level of refinement; n =1 => number of point divided by 2^1=2; n = 2 => number of point divided by 2^2=4;)
    Output:
      arr_sizes = [b, r, k, itera]
         b = normalized size of standard blocks (size of standard blocks = b * 2^n)
         r = rest (if not equal to 0, is the size of the last block)
         k = number of standard blocks
         itera = number of itarations to converge
    """
    h = 0
    b_h = w
    k_h = floor(D / (pow(2.0, n) * b_h))
    b = b_h
    r = D % (pow(2.0, n) * b_h)
    k = k_h
    itera = 0
    verif = bool(1)
    while verif:
        if D % (pow(2.0, n) * b_h) == 0:
            b = b_h
            r = 0
            k = k_h
            verif = bool(0)
        elif D % (pow(2.0, n) * b_h) > r:
            b = b_h
            r = D % (pow(2.0, n) * b_h)
            k = k_h
        if h == floor(w / 2):
            verif = bool(0)
        h = min(floor(w / 2), h + max(1, floor(b_h - D / (pow(2.0, n) * (k_h + 1)))))
        b_h = w - h
        k_h = floor(D / (pow(2.0, n) * b_h))
        itera = itera + 1

    b = int(b)
    r = int(r)
    k = int(k)
    arr_sizes = [b, r, k, itera]
    return arr_sizes


def prep_array(wb, r, k):
    """
    Create a 1D array containing the number of elements per tile.
    Input:
         wb = size of standard blocks
         r = rest (if not equal to 0, is the size of the last block)
         k = number of standard blocks
    Output:
       array = A list containing the number of element for every tiles.
    """
    for i in range(0, k):
        if i == 0:
            array = [int(wb)]
        elif i > 0:
            array.append(int(wb))
        else:
            print(("Option not permitted!!!!!! i =", i))
            sys.exit(1)

    if r != 0:
        if k != 0:
            array.append(r)
        else:
            array = [r]
    return array


def create_sizes(size, wb, max_res, norest=False):
    """
    Create a 3D array containing the size for each tile on the desidered direction
    Input:
       start_wb = Start parameter for b
       size = size (in pixel) of the input immage
       wb = Rough depth for the tiles in the desidered direction
       max_res = Maximum level of resolution available (integer)
       norest = Boolean variable to chech if we need of the last array element (if it is different from the preavious one)
    Output:
       arr = Array containing the size for each tile on the desidered direction
    """
    values = opt_algo(size, wb, max_res)
    b = values[0]
    r = values[1]
    k = values[2]
    itera = values[3]
    wb = int(pow(2, max_res) * b)
    arr = prep_array(wb, r, k)
    if norest:
        tmp_len = len(arr)
        if arr[(tmp_len - 1)] != arr[(tmp_len - 2)]:
            print(("Attention! : ", arr[(tmp_len - 1)], " points was deleted!"))
            arr.pop()
    return arr


def create_starts_end(array, start_point=0, open_dx=True):
    """
    Create arrays containing all the starting and ending indexes for the tiles on the desidered direction
    Input:
       array = Array containing the size for each tile on the desidered direction
       start_point = Starting index for the input immage (optional)
       open_dx = If true (the default value) ==> ending indexes = subsequent starting indexes ==> Open end
    Output:
       star_arr = Array containing all the starting indexes for the tiles on the desidered direction
       end_arr = Array containing all the ending indexes for the tiles on the desidered direction
    """
    len_arr = len(array)
    ind_arr = list(range(0, len_arr))
    start_arr = []
    end_arr = []
    if open_dx:
        dx_pad = 0
    else:
        dx_pad = -1
    for i in ind_arr:
        if i != 0:
            start_point = start_point + array[(i - 1)]
        start_arr.append(start_point)
        end_point = start_point + array[i] + dx_pad
        end_arr.append(end_point)

    return (start_arr, end_arr)


def ctrl_parallelism(sfmt, dfmt):
    partition_depth = True
    partition_width = True
    partition_height = True
    if sfmt == "TIFF (3D)" or dfmt == "TIFF (series, 2D)":
        partition_width = False
        partition_height = False
    if sfmt == "TIFF (series, 2D)":
        partition_width = False
    return (partition_depth, partition_width, partition_height)


def create_commands(gi_np, info=False):
    """
    Create commands to run in parallel
    Input:
    Output:
       first_string = String to initialize parallel computation
       list_string = Dictionary of strings containing the command lines to process the data. E.G.: {i:command[i]}
       len_arr = Dictionary containing elements like {index:[size_width(i),size_height(i),size_depth(i)],.....}
       final_string = String to merge all metadadata
    """
    (
        input_name,
        output_name,
        wb1,
        wb2,
        wb3,
        sfmt,
        dfmt,
        iresolutions,
        max_res,
        isotropic,
        max_res_D,
        params,
        last_string,
        height,
        width,
        depth,
        bytes_x_chan,
        n_chans,
    ) = read_params()
    print("#" * 80)
    print(("Input file = ", input_name))
    print(("Output directory", output_name))
    if wb1 == 0:
        wb1 = default_tile_size
    if wb2 == 0:
        wb2 = default_tile_size
    if wb3 == 0:
        wb3 = default_tile_size
    print(("Rough depth for the tiles in width direction = ", wb3))
    print(("Rough depth for the tiles in height direction = ", wb2))
    print(("Rough depth for the tiles in depth direction = ", wb1))
    print(("Source Format = ", sfmt))
    print(("Destination Format = ", dfmt))
    print(("Resolutions = ", iresolutions))
    print(("Max Resolutions", max_res))
    print(("Width (in voxel) of the immage = ", width))
    print(("Height (in voxel) of the immage = ", height))
    print(("Depth (in voxel) of the immage = ", depth))
    print(params)
    if isotropic:
        last_string = last_string + " --isotropic"
    print(("Last input elements of the original string = ", last_string))
    print("#" * 80)
    size_1 = create_sizes(depth, wb1, max_res_D)
    size_2 = create_sizes(height, wb2, max_res)
    size_3 = create_sizes(width, wb3, max_res)
    assert (
        dfmt != "HDF5 (BigDataViewer)" or dfmt != "HDF5 (Imaris IMS)"
    ), "Paraconverter cannot be used with HDF5 output formats"
    partition_depth, partition_width, partition_height = ctrl_parallelism(
        eliminate_double_quote(sfmt), eliminate_double_quote(dfmt)
    )
    print(
        (
            "--------> ",
            eliminate_double_quote(sfmt),
            eliminate_double_quote(dfmt),
            partition_depth,
            partition_width,
            partition_height,
        )
    )
    if len(size_1) >= 2 * gi_np or not partition_width and not partition_height:
        size_2 = [height]
        size_3 = [width]
    elif len(size_1) * len(size_2) >= 2 * gi_np or not partition_width:
        size_3 = [width]
    print(
        (
            "number of work units (Depth, Height, Width): ",
            len(size_1),
            len(size_2),
            len(size_3),
        )
    )
    print(("size of work units (Depth, Height, Width): ", size_1, size_2, size_3))
    if info:
        first_string = ""
        list_string = ""
        final_string = ""
        len_arr = 0
        voxel_num = round(
            1.1
            * gi_np
            * (size_2[0] * size_3[0] * max(64, pow(2, max_res)))
            * n_chans
            * bytes_x_chan
            / 1073741824,
            3,
        )
        print("#" * 80)
        print(
            "Memory needed for "
            + str(gi_np)
            + " concurrent processes: "
            + str(voxel_num)
            + " GBytes"
        )
        print("#" * 80)
    else:
        start_3, end_3 = create_starts_end(size_3, 0)
        start_2, end_2 = create_starts_end(size_2, 0)
        start_1, end_1 = create_starts_end(size_1, 0)
        order, start_list, end_list, len_arr = sort_start_end(
            start_1, start_2, start_3, end_1, end_2, end_3, size_1, size_2, size_3
        )
        first_string = generate_first_command(
            input_name,
            output_name,
            wb1,
            wb2,
            wb3,
            sfmt,
            dfmt,
            iresolutions,
            max_res,
            params,
            last_string,
        )
        list_string = generate_parallel_command(
            start_list,
            end_list,
            input_name,
            output_name,
            wb1,
            wb2,
            wb3,
            sfmt,
            dfmt,
            iresolutions,
            max_res,
            params,
            last_string,
        )
        final_string = generate_final_command(
            input_name,
            output_name,
            wb1,
            wb2,
            wb3,
            sfmt,
            dfmt,
            iresolutions,
            max_res,
            params,
            last_string,
        )
    return (first_string, list_string, output_name, len_arr, final_string)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    myrank = comm.Get_rank()
    comm.Barrier()
    tmp = read_item(sys.argv, "--info", "no_info")
    if tmp == "no_info":
        info = False
    else:
        info = True
    step2 = False
    tmp = read_item(sys.argv, "--displcompute", "no_step", False)
    if tmp != "no_step":
        step2 = True
    tmp = read_item(sys.argv, "-2", "no_step", False)
    if tmp != "no_step":
        step2 = True
    step6 = False
    tmp = read_item(sys.argv, "--merge", "no_step", False)
    if tmp != "no_step":
        step6 = True
    tmp = read_item(sys.argv, "-6", "no_step", False)
    if tmp != "no_step":
        step6 = True
    if platform.system() != "Windows":
        if step2:
            if prefix == "":
                if os.system("which ./terastitcher") == 0:
                    prefix = "./"
                elif os.system("which terastitcher") != 0:
                    raise ValueError("The executable of terasticher is not reachable")
        elif step6:
            if prefix == "":
                if os.system("which ./teraconverter") == 0:
                    prefix = "./"
                elif os.system("which teraconverter") != 0:
                    raise ValueError("The executable of teraconverter is not reachable")
    if myrank == 0:
        t1 = time.time()
        print("*" * 80)
        print(
            (
                str(datetime.datetime.utcnow()),
                " -- Calculation started on ",
                nprocs,
                "- 1 cores.",
            )
        )
        print("*" * 80)
    comm.Barrier()
    if myrank == 0:
        if step2:
            execution_flag = "-2"
            params = extract_params()
            print(params)
            params.pop(0)
            print("Alignment will be performed")
            tmp = check_flag(params, "subvoldim=", True)
            if tmp == None:
                nsubstring_min = 200
                print(
                    (
                        "Number of substring was not declared. It will be set to",
                        nsubstring_min,
                        "by default.",
                    )
                )
            else:
                tmp = int(tmp)
                nsubstring_min = tmp
            tmp = check_flag(params, "projin=", False)
            if tmp == None:
                input_name = "xml_import.xml"
                print(
                    (
                        "Name of the input file was not declared. It will be set to",
                        input_name,
                        "by default.",
                    )
                )
            else:
                input_name = tmp
            tmp = check_flag(params, "projout=", True)
            if tmp == None:
                output_name = "xml_compdispl.xml"
                print(
                    (
                        "name of the output file was not declared. It will be set to",
                        output_name,
                        "by default.",
                    )
                )
            else:
                output_name = tmp
            len_out = len(output_name)
            output_name = output_name[0 : len_out - 4]
            params = add_chars(params)
            nrows, ncols, nslices = extract_np(input_name)
            n_ss = int(ceil(float(nslices) / float(nsubstring_min)))
            last_size = int(floor(float(nslices) / float(n_ss)))
            first_size = last_size + 1
            n_of_first_stacks = nslices % n_ss
            n_of_last_stacks = n_ss - n_of_first_stacks
            p_r, p_c = do_additional_partition(
                float(nprocs - 1), float(nrows), float(ncols), float(n_ss)
            )
            s_r = int(floor(nrows / p_r))
            r_r = nrows % p_r
            r_start = [0]
            r_end = []
            for i in range(1, int(p_r - r_r)):
                r_end.append(i * s_r)
                r_start.append(i * s_r)

            offset = int((p_r - r_r) * s_r)
            for i in range(int(r_r)):
                r_end.append(offset + i * s_r)
                r_start.append(offset + i * s_r)

            r_end.append(nrows - 1)
            print(r_start, r_end)
            s_c = int(floor(ncols / p_c))
            r_c = ncols % p_c
            c_start = [0]
            c_end = []
            for i in range(1, int(p_c - r_c)):
                c_end.append(i * s_c)
                c_start.append(i * s_c)

            offset = int((p_c - r_c) * s_c)
            for i in range(int(r_c)):
                c_end.append(offset + i * s_c)
                c_start.append(offset + i * s_c)

            c_end.append(ncols - 1)
            print(c_start, c_end)
            if len(r_start) > 1 or len(c_start) > 1:
                gr_dir_names = []
                for i in range(len(r_start)):
                    for j in range(len(c_start)):
                        gr_dir_names.append(
                            "gr_R["
                            + str(r_start[i])
                            + ","
                            + str(r_end[i])
                            + "]_C["
                            + str(c_start[j])
                            + ","
                            + str(c_end[j])
                            + "]/"
                        )

            cmd_string = {}
            params_str = (" ").join(params)
            tmp_out_name = find_last_slash(output_name)
            tmp_xml_dir = tmp_out_name[0] + "tmp/"
            try:
                shutil.rmtree(tmp_xml_dir)
            except OSError:
                pass

            print("removed directory tree " + tmp_xml_dir + " if any")
            os.mkdir(tmp_xml_dir)
            print("created directory " + tmp_xml_dir)
            if len(r_start) == 1 and len(c_start) == 1:
                start_dict = {}
                end_dict = {}
                start_tmp = 0
                end_tmp = 0
                new_params = {}
                new_output_name = tmp_xml_dir + tmp_out_name[1]
                for i in range(n_ss):
                    start_dict.update({i: end_tmp})
                    if i < n_of_first_stacks:
                        new_params.update({i: first_size})
                        end_tmp += first_size
                    else:
                        new_params.update({i: last_size})
                        end_tmp += last_size
                    end_dict.update({i: end_tmp - 1})
                    tmp_string = (
                        execution_flag
                        + " "
                        + params_str
                        + " --projout="
                        + new_output_name
                        + "-"
                        + str(start_dict[i]).zfill(6)
                        + "-"
                        + str(end_dict[i]).zfill(6)
                        + ".xml"
                        + " --subvoldim="
                        + str(new_params[i])
                        + " --D0="
                        + str(start_dict[i])
                        + " --D1="
                        + str(end_dict[i])
                        + " --noprogressbar"
                    )
                    cmd_string.update({i: tmp_string})

            else:
                for r in range(len(r_start)):
                    for c in range(len(c_start)):
                        gr_xml_dir = (
                            tmp_out_name[0] + gr_dir_names[(r * len(c_start) + c)]
                        )
                        try:
                            shutil.rmtree(gr_xml_dir)
                        except OSError:
                            pass

                        print("removed directory tree " + gr_xml_dir + " if any")
                        os.mkdir(gr_xml_dir)
                        print("created directory " + gr_xml_dir)
                        start_dict = {}
                        end_dict = {}
                        start_tmp = 0
                        end_tmp = 0
                        new_params = {}
                        new_output_name = gr_xml_dir + tmp_out_name[1]
                        for i in range(n_ss):
                            start_dict.update({i: end_tmp})
                            if i < n_of_first_stacks:
                                new_params.update({i: first_size})
                                end_tmp += first_size
                            else:
                                new_params.update({i: last_size})
                                end_tmp += last_size
                            end_dict.update({i: end_tmp - 1})
                            tmp_string = (
                                execution_flag
                                + " "
                                + params_str
                                + " --projout="
                                + new_output_name
                                + "-"
                                + str(start_dict[i]).zfill(6)
                                + "-"
                                + str(end_dict[i]).zfill(6)
                                + ".xml"
                                + " --subvoldim="
                                + str(new_params[i]).zfill(6)
                                + " --D0="
                                + str(start_dict[i]).zfill(6)
                                + " --D1="
                                + str(end_dict[i]).zfill(6)
                            )
                            tmp_string = (
                                tmp_string
                                + " --R0="
                                + str(r_start[r])
                                + " --R1="
                                + str(r_end[r])
                                + " --C0="
                                + str(c_start[c])
                                + " --C1="
                                + str(c_end[c])
                                + " --noprogressbar --parallel"
                            )
                            if c < len(c_start) - 1:
                                tmp_string = tmp_string + " --disable_last_col"
                            if r < len(r_start) - 1:
                                tmp_string = tmp_string + " --disable_last_row"
                            cmd_string.update(
                                {(r * len(c_start) + c) * n_ss + i: tmp_string}
                            )

            work_list = cmd_string
            main_step2(work_list)
            if len(r_start) == 1 and len(c_start) == 1:
                slash_pos = len(tmp_xml_dir) - 1
                if debug_level > 0:
                    execution_string = (
                        prefix
                        + "mergedisplacements -d="
                        + tmp_xml_dir[0:slash_pos]
                        + " -o="
                        + tmp_out_name[0]
                        + tmp_out_name[1]
                        + ".xml"
                        + " > "
                        + "ouput_mdispls.out"
                    )
                else:
                    execution_string = (
                        prefix
                        + "mergedisplacements -d="
                        + tmp_xml_dir[0:slash_pos]
                        + " -o="
                        + tmp_out_name[0]
                        + tmp_out_name[1]
                        + ".xml"
                    )
                print(execution_string, end=" ")
                print()
                os.system(execution_string)
            else:
                for dname in gr_dir_names:
                    slash_pos = dname.find("/")
                    suffix = dname[dname.find("_") : slash_pos]
                    if debug_level > 0:
                        execution_string = (
                            prefix
                            + "mergedisplacements -d="
                            + tmp_out_name[0]
                            + dname[0:slash_pos]
                            + " -o="
                            + tmp_out_name[0]
                            + "tmp/"
                            + tmp_out_name[1]
                            + suffix
                            + ".xml"
                            + " > "
                            + "ouput_"
                            + suffix
                            + ".out"
                        )
                    else:
                        execution_string = (
                            prefix
                            + "mergedisplacements -d="
                            + tmp_out_name[0]
                            + dname[0:slash_pos]
                            + " -o="
                            + tmp_out_name[0]
                            + "tmp/"
                            + tmp_out_name[1]
                            + suffix
                            + ".xml"
                        )
                    print(execution_string)
                    os.system(execution_string)

                slash_pos = len(tmp_xml_dir) - 1
                if debug_level > 0:
                    execution_string = (
                        prefix
                        + "mergedisplacements --mgroups -d="
                        + tmp_xml_dir[0:slash_pos]
                        + " -o="
                        + tmp_out_name[0]
                        + tmp_out_name[1]
                        + ".xml"
                        + " > "
                        + "ouput_mgroups.out"
                    )
                else:
                    execution_string = (
                        prefix
                        + "mergedisplacements --mgroups -d="
                        + tmp_xml_dir[0:slash_pos]
                        + " -o="
                        + tmp_out_name[0]
                        + tmp_out_name[1]
                        + ".xml"
                    )
                print(execution_string)
                os.system(execution_string)
            if len(r_start) > 1 or len(c_start) > 1:
                for r in range(len(r_start)):
                    for c in range(len(c_start)):
                        gr_xml_dir = (
                            tmp_out_name[0] + gr_dir_names[(r * len(c_start) + c)]
                        )
                        shutil.rmtree(gr_xml_dir)
                        print(
                            "deleted directory " + gr_xml_dir + " and all files in it"
                        )

            shutil.rmtree(tmp_xml_dir)
            print("deleted directory " + tmp_xml_dir + " and all files in it")
        elif step6:
            if info:
                (
                    first_string,
                    list_string,
                    output_name,
                    len_arr,
                    final_string,
                ) = create_commands(nprocs - 1, True)
            else:
                (
                    first_string,
                    list_string,
                    output_name,
                    len_arr,
                    final_string,
                ) = create_commands(nprocs - 1)
                if save_status_prefix == "":
                    save_status_prefix = output_name + "/"
                rs_fname = save_status_prefix + resume_status_fname
                if not os.path.exists(rs_fname):
                    if debug_level > 0:
                        execution_string = (
                            prefix + first_string + " > " + "output_first.out"
                        )
                    else:
                        execution_string = prefix + first_string
                    os.system(execution_string)
                    print(execution_string)
                    if suspend_resume_enabled:
                        rs_file = open(rs_fname, "wb")
                        pickle.dump([], rs_file)
                        rs_file.close()
                cmd_string = list_string
                npoints = len_arr
                scores = score_function(npoints)
                elaborations = sort_elaborations(scores)
                work_list = sort_work(cmd_string, elaborations)
                main_step6(work_list, rs_fname)
                if debug_level > 0:
                    execution_string = (
                        prefix + final_string + " > " + "output_final.out"
                    )
                else:
                    execution_string = prefix + final_string
                os.system(execution_string)
                print(execution_string)
    elif step2:
        prefix = prefix + "terastitcher "
        subordinate()
    elif step6:
        if info:
            dummy = 0
        else:
            subordinate()
    comm.Barrier()
    if myrank == 0:
        t2 = time.time()
        print("*" * 80)
        print(
            (
                str(datetime.datetime.utcnow()),
                "-- Calculation ended after ",
                t2 - t1,
                " seconds",
            )
        )
        print("*" * 80)
