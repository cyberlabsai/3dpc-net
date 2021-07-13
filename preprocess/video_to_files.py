'''
Pega as informacoes de video do dataset OULU e transforma ela em arquivos de imagem 
'''
import argparse
import os 
import logging
import shutil
import cv2
from frameExtractor import Extract_images_multiprocess

logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(funcName)s: %(message)s'))
handler.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

def read_protocol_txt_file(filename):
    logger.info(f'Reading File: {filename}')

    protocol_info = {'Live': [], 'Print_Attack': [], 'Video_Replay_Attack': []}

    with open(filename, 'r') as txt_file:
        for line in txt_file.readlines():
            line = line.replace('\n', '')
            label, video_file = line.split(',')
            if label == '+1':
                protocol_info['Live'].append(video_file)
            elif label == '-1':
                protocol_info['Print_Attack'].append(video_file)
            elif label == '-2':
                protocol_info['Video_Replay_Attack'].append(video_file)
            else:
                logger.error(f'Unexpected line ({line}) in file {filename}')

    return protocol_info

def load_protocol_info(protocols_folder, protocol, subset, subProtocolDivision=None):
    protocol_folder = os.path.join(protocols_folder, f"Protocol_{protocol}")
    logger.info(f'Protocol Folder Path: {protocol_folder}')

    if protocol == '1' or protocol == '2':
        protocol_test_file = os.path.join(protocol_folder, f'{subset}.txt')
        logger.info(f'Protocol Test File Path: {protocol_test_file}')
        protocol_info = read_protocol_txt_file(protocol_test_file)
    else:
        protocol_test_file = os.path.join(protocol_folder, f'{subset}_{subProtocolDivision}.txt')
        logger.info(f'Protocol Test File Path: {protocol_test_file}')
        protocol_info = read_protocol_txt_file(protocol_test_file)

    # if subset == 'train':
    #     raise NotImplementedError('We do not implemented the video to file extraction for training files yet')
    # elif subset == 'dev':
    #     raise NotImplementedError('We do not implemented the video to file extraction for dev files yet')
    # if subset == 'test':
    #     if protocol == '1' or protocol == '2':
    #         protocol_test_file = os.path.join(protocol_folder, 'Test.txt')
    #         logger.info(f'Protocol Test File Path: {protocol_test_file}')
    #         protocol_info = read_protocol_txt_file(protocol_test_file)
    #     else:
    #         protocol_test_file = os.path.join(protocol_folder, f'Test_{subProtocolDivision}.txt')
    #         logger.info(f'Protocol Test File Path: {protocol_test_file}')
    #         protocol_info = read_protocol_txt_file(protocol_test_file)

    return protocol_info

def create_protocol_folder(processed_folder, protocol, subset, subProtocolDivision=None):
    '''
        Cria a pasta para aquele protocolo especifico dentro da pasta processed
    '''
    if not os.path.isdir(processed_folder):
        raise Exception('The processed folder informed does not exist')

    processed_folder = os.path.join(processed_folder, f'Protocol_{protocol}')

    if not os.path.isdir(processed_folder):
        os.mkdir(processed_folder)

    if subProtocolDivision is not None:
        processed_folder = os.path.join(processed_folder, f'Division_{subProtocolDivision}')
        if not os.path.isdir(processed_folder):
            os.mkdir(processed_folder)

    subset_folder = os.path.join(processed_folder, subset)
    if not os.path.isdir(subset_folder):
        os.mkdir(subset_folder)
    else:
        ans = input(f'The folder {subset_folder} already exists. To process the files again we will delete'
                    'its tree and creat all again. Are you sure of it? (y/n)')
        if ans.lower() != 'y':
            logger.info('Exiting then..')
            exit()
        else:
            shutil.rmtree(subset_folder)
            os.mkdir(subset_folder)

    return subset_folder

def check_arguments(args):
    '''
        Checa se os argumentos estao ok. Se o path para os raw files do subset selecionado foram passados
    '''
    if ((args['subset'] == 'Train' and args['train_folder'] is None) or
        (args['subset'] == 'Test' and args['test_folder'] is None) or
        (args['subset'] == 'Dev' and args['dev_folder'] is None)):
        raise Exception(f'If you want to process the subset {args["subset"]} you need to pass a path for the {args["subset"]}_files folder')



def thread_work(args, subset_folder, protocol_info):
    eval_threads = []

    for label, data in protocol_info.items():
        eval_threads.append(Extract_images_multiprocess(args, data, subset_folder, label))

    for evalu in eval_threads:
        evalu.start()

    processes_finished = [False] * len(eval_threads)
    while True:
        all_finished = True
        for i, evalu in enumerate(eval_threads):
            if not evalu.Q.empty():
                process_dist = evalu.Q.get()
                processes_finished[i] = True

        if sum(processes_finished) == len(processes_finished):
            for evalu in eval_threads:
                del(evalu)
            break

def main(args):
    if args['protocol'] == '1' or args['protocol'] == '2':
        check_arguments(args)
        protocol_info = load_protocol_info(args['protocols_folder'],  args['protocol'], args['subset'])
        subset_folder = create_protocol_folder(args['processed_folder'], args['protocol'], args['subset'])
        
        thread_work(args, subset_folder, protocol_info)
        
    else:
        for subProtocolDivision in range(2, 3):
        # for subProtocolDivision in range(1, 7):
            check_arguments(args)
            protocol_info = load_protocol_info(args['protocols_folder'],  args['protocol'], args['subset'], subProtocolDivision=subProtocolDivision)
            subset_folder = create_protocol_folder(args['processed_folder'], args['protocol'], args['subset'], subProtocolDivision=subProtocolDivision)
            
            thread_work(args, subset_folder, protocol_info)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument('--protocol', choices=['1', '2', '3', '4'], default='1',
                    help='Which Protocol do you want to use')
    ap.add_argument('--protocols-folder', required=True, metavar='PATH',
                    help='Path to Protocols folder')
    ap.add_argument('--subset', choices=['Train', 'Dev', 'Test'], default='test',
                    help='Which subset you want to extract the images from')
    ap.add_argument('--train-folder', metavar='PATH', default=None,
                    help='Path to Train Files folder')
    ap.add_argument('--test-folder', metavar='PATH', default=None,
                    help='Path to Test Files folder')
    ap.add_argument('--dev-folder', metavar='PATH', default=None,
                    help='Path to Dev Files folder')
    ap.add_argument('--processed-folder', required=True, metavar='PATH',
                    help='Path to the folder where the processed files will be stored')

    args = vars(ap.parse_args())
    main(args)