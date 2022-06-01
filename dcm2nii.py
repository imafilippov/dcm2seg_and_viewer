import dicom2nifti
import re
import os
from datetime import datetime
import glob


DIR = os.getcwd()

# DIR = '/.../DIRECTORY WHERE ALL SUB_IDs AND DICOM FOLDERS ARE STORED'
# the script expects the below folder structure: (based on output from MR Solutions preclinical MRI)

#  DIR
#   |--001
#   |   |--DICOM
#   |   |    |--0001
#   |   |        |--1
#   |   |           |--0001_00001.dcm
#   |   |--Image
#   |        |--0001
#   |            |--1
#   |               |--0001_00001.SUR
#   |
#   |--002 ... etc.

while True:
    defaults = input('use defaults for this one? \ny / n?')
    if not re.match('^[y|n]', defaults):
        print('only "y" or "n" allowed!')
    elif defaults in ['y', 'yes']:
        default = True
        print(f'DIR is {DIR}')
        dicom_directory = DIR
        print(f'dicom_directory is {dicom_directory}')
        output_directory = os.path.join(dicom_directory, str(datetime.date(datetime.now())) + '_nifti_output')
        print(f'output is {output_directory}')
        if os.path.exists(output_directory):
            print(f'{output_directory} path is valid')
        else:
            make_dir = input('dir does not exist. create it? \ny / n?')
            if make_dir in ['y', 'yes']:
                os.mkdir(output_directory)
                print(f'new dir made: {output_directory}!')
        compression = True
        break

    elif defaults in ['n', 'no']:
        default = False
        print(f'use {DIR} as root dir?')
        while True:
            confirmation = input('y / n : ')
            if not re.match('^[y|n]', confirmation):
                print('only "y" or "n" allowed!')
            elif confirmation in ['n', 'no']:
                DIR = ''
                print('DIR is cleared.')
                break
            elif confirmation in ['y', 'yes']:
                print(f'DIR is still {DIR}')
                break

        dicom_directory = os.path.join(DIR, input(f'enter DICOM DIRECTORY to convert: {DIR}'))
        print(f'you chose {dicom_directory} Correct?')
        while True:
            confirmation = input('y / n : ')
            if not re.match('^[y|n]', confirmation):
                print('only "y" or "n" allowed!')
            elif confirmation in ['n', 'no']:
                dicom_directory = input('enter DICOM DIRECTORY to convert: ')
                print(f'you chose {dicom_directory}\n Correct?')
            elif confirmation in ['y', 'yes']:
                break

        output_directory = os.path.join(dicom_directory, str(datetime.date(datetime.now())) + '_nifti_output')
        print(f'Save to default save location: \n"{output_directory}"')
        while True:
            confirmation = input('y / n : ')
            if not re.match('^[y|n]', confirmation):
                print('only "y" or "n" allowed!')
            elif confirmation in ['n', 'no']:
                output_directory = input('enter OUTPUT DIRECTORY as save location: ')
                print(f'you chose {output_directory}\\\n Correct?')
            elif confirmation in ['y', 'yes']:
                if os.path.exists(output_directory):
                    print(f'{output_directory} path is valid')
                    break
                else:
                    make_dir = input('dir does not exist. create it? \ny / n?')
                    if make_dir in ['y', 'yes']:
                        os.mkdir(output_directory)
                        print(f'new dir made: {output_directory}!')
                        break
                    elif make_dir in ['n', 'no']:
                        continue

        print('Compression: Save as .nii.gz?: ')
        while True:
            confirmation = input('y / n : ')
            if not re.match('^[y|n]', confirmation):
                print('only "y" or "n" allowed!')
            elif confirmation in ['n', 'no']:
                compression = False
                break
            elif confirmation in ['y', 'yes']:
                compression = True
                break
        break

scan_list = [6656, 6753]
scan_list = [item for item in input("Split only with '/'. It literally wont work if you dont."
                                    "\nEnter the list items : ").split('/')]

# scan_list = [item for item in ser_list.split('/')]
print(scan_list)

# TODO later
# if '+' in scan for scan in scan_list:
#   convert3d, add the scans together after they are output as nifti files for a 2:avg scan

dicom_list = []
# for dicom in glob.glob(f'{dicom_directory}\\*\\DICOM\\*', recursive=True):
for dicom in glob.glob(os.path.join(dicom_directory, '*', 'DICOM', '*'), recursive=True):
    for seq in scan_list:
        if str(seq) in dicom.split(os.sep)[-1]:
            print(dicom)
            dicom_list.append(dicom)


for dicom in dicom_list:
    output = os.path.join(output_directory, dicom.split(os.sep)[-3])
    if os.path.exists(output):
        print(f'{output} path is valid')
    else:
        if default is True:
            os.mkdir(output)
            print(f'new dir made: {output}!')
        else:
            make_dir = input(f'{output} does not exist. create it? \ny / n?')
            if make_dir in ['y', 'yes']:
                os.mkdir(output)
                print(f'new dir made: {output}!')
            elif make_dir in ['n', 'no']:
                print('have fun not having files')
    try:
        dicom2nifti.convert_directory(dicom, output, compression=compression, reorient=True)
    except:
        print('could not convert')


print('thanks for playing!')
