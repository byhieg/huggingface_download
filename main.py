import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing

import requests
import json

from hf_model import ModelFileInfo, LFS

model_name = 'chatglm2-6b'
branch_name = 'main'
retry = 5
timeout = 10
token = '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'
buffer_size = 1024 * 4
author = 'THUDM'


def add_auth_header(token):
    return {"Authorization": f'Bearer {token}'}


def redict(url, token=None):
    for i in range(retry):
        header = {}
        if token:
            header = add_auth_header(token)
        try:
            response = requests.head(url, timeout=timeout, headers=header, allow_redirects=True)
            if response.url:
                return response.url
            raise RuntimeError(f'reason is {response.reason},text is {response.text}')
        except Exception as e:
            print(f'{repr(e)}')

    return None


def get(url, token=None):
    for i in range(retry):
        header = {}
        if token:
            header = add_auth_header(token)
        try:
            response = requests.get(url, timeout=timeout, headers=header)
            if response.ok:
                return response.text
            else:
                if response.status_code == 401:
                    print(
                        f'This Repo requires access token, generate an access token form huggingface, '
                        f'and pass it using token')

                if response.status_code == 403:
                    print(
                        f'You need to manually Accept the agreement for this model on HuggingFace site, '
                        f'No bypass will be implemeted')
                raise RuntimeError(f'reason is {response.reason},text is {response.text}')
        except Exception as e:
            print(f'{repr(e)}')

    return None


def head(url, token=None):
    for i in range(retry):
        header = {}
        if token:
            header = add_auth_header(token)
        try:
            response = requests.head(url, timeout=timeout, headers=header)
            if response.ok:
                return response.headers
            else:
                if response.status_code == 401:
                    print(
                        f'This Repo requires access token, generate an access token form huggingface, '
                        f'and pass it using token')

                if response.status_code == 403:
                    print(
                        f'You need to manually Accept the agreement for this model on HuggingFace site, '
                        f'No bypass will be implemeted')
                raise RuntimeError(f'reason is {response.reason},text is {response.text}')
        except Exception as e:
            print(f'{repr(e)}')

    return None


def http_download_offset_range(url, offset=0, range=0, token=None):
    headers = {
        "Range": f'bytes={offset}-{range - 1}'
    }
    if token:
        headers.update(add_auth_header(token))
    with closing(requests.get(url, stream=True, headers=headers)) as reader:
        for chunk in reader.iter_content(chunk_size=buffer_size):
            yield chunk


def check_sum(file_path, expected_check_sum):
    import hashlib
    with open(file_path, 'rb') as r:
        return hashlib.sha256(r.read()).hexdigest() == expected_check_sum


def process_hf_model(local_model_path, model_name, branch_name, sub_dir_name, hf_model_list):
    json_models_file_tree_url = f'https://huggingface.co/api/models/{model_name}/tree/{branch_name}/{sub_dir_name}'

    print(f"Getting File Download Files List Tree from: {json_models_file_tree_url}")

    model_files_response = get(json_models_file_tree_url)
    if not model_files_response:
        raise RuntimeError(f'{json_models_file_tree_url} request failed')
    file_list = json.loads(model_files_response)

    for file in file_list:
        if 'lfs' in file.keys():
            hf_model_lfs = LFS(file['lfs']['oid'], file['lfs']['size'], file['lfs']['pointerSize'])
        else:
            hf_model_lfs = None
        hf_model = ModelFileInfo(file['type'], file['oid'], file['size'], file['path'], hf_model_lfs)
        hf_model_list.append(hf_model)
        hf_model.append_path = os.path.join(local_model_path, hf_model.path)
        if hf_model.type == 'directory':
            hf_model.is_dir = True
            hf_model.skip_download = True
            process_hf_model(local_model_path, model_name, branch_name, hf_model.path, hf_model_list)
            continue

        hf_model.download_link = f'https://huggingface.co/{model_name}/raw/{branch_name}/{hf_model.path}'

        if hf_model.lfs:
            hf_model.is_lfs = True
            resolver_path = f'https://huggingface.co/{model_name}/resolve/{branch_name}/{hf_model.path}'
            hf_model.download_link = redict(resolver_path, token=token)


def get_hf_model(author, model_name):
    if author:
        hf_model_url = f'https://huggingface.co/api/models/{author}/{model_name}'
    else:
        hf_model_url = f'https://huggingface.co/api/models/{model_name}'

    response = get(hf_model_url)
    if not response:
        raise RuntimeError(f'{hf_model_url} request failed')
    return json.loads(response)


def download_file(download_link, local_path, thread_num=1, token=None, check_sum_value=None, file_length=None):
    headers = head(download_link, token)
    if not headers:
        raise RuntimeError(f'headers can not get for {download_link}')
    file_content_length = int(headers.get('Content-Length'))

    if thread_num <= 1:
        dir_path, base_name = os.path.split(local_path)
        os.makedirs(dir_path, exist_ok=True)
        offset = 0
        for i in range(retry):
            try:
                if os.path.isfile(local_path):
                    offset = os.stat(local_path).st_size
                with open(local_path, 'ab') as writer:
                    for data in http_download_offset_range(download_link, offset, file_content_length, token):
                        writer.write(data)
                break
            except Exception as e:
                print(f'{e}')
                time.sleep(2)

        if check_sum_value:
            if not check_sum(local_path, check_sum_value):
                raise RuntimeError(f'download {local_path} failed,sum is error')

        if not check_sum_value and file_length:
            file_size = os.stat(local_path).st_size
            if not file_size == file_length:
                raise RuntimeError(f'download {local_path} failed,local length {file_size} is not '
                                   f'equal remote file length {file_length}')
    else:

        def __multi_thread_download(_offset, _range, _fd):
            try:
                file_offset = _offset
                for _data in http_download_offset_range(download_link, _offset, _range, token):
                    os.pwrite(_fd, _data, file_offset)
                    file_offset += len(_data)
            except Exception as e:
                print(f'{e}')

        slice_size = file_content_length / thread_num
        for i in range(retry):
            try:
                dir_path, base_name = os.path.split(local_path)
                os.makedirs(dir_path, exist_ok=True)
                with open(local_path, 'wb') as f:
                    fd = f.fileno()
                    with ThreadPoolExecutor(max_workers=thread_num) as _executor:
                        _futures = []
                        for j in range(thread_num):
                            _offset = j * slice_size
                            _range = _offset + slice_size
                            if j == thread_num - 1:
                                _range = file_content_length
                            _future = _executor.submit(__multi_thread_download, _offset, _range, fd)
                            _futures.append(_future)

                        for _future in as_completed(_futures):
                            _future.result()
                if check_sum_value:
                    if not check_sum(local_path, check_sum_value):
                        raise RuntimeError(f'download {local_path} failed,sum is error')

                if not check_sum_value and file_length:
                    file_size = os.stat(local_path).st_size
                    if not file_size == file_length:
                        raise RuntimeError(f'download {local_path} failed,local length {file_size} is not '
                                           f'equal remote file length {file_length}')
                break
            except Exception as e:
                import traceback
                traceback.print_exc()
                time.sleep(2)


if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))

    tmp_dir = os.path.join(dir_path,'tmp')
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)

    model_info = get_hf_model(author=author, model_name=model_name)
    model_last_modified_time = model_info['lastModified']
    model_sha = model_info['sha']
    model_id = model_info['_id']
    print(f'{author}:{model_name} {model_id} last_time {model_last_modified_time} sha {model_sha}')
    hf_model_list = []
    if author:
        model_name = f'{author}/{model_name}'
    process_hf_model(tmp_dir, model_name, branch_name, '',hf_model_list)
    print(f'download info ready......')
    for hf_model in hf_model_list:
        if not hf_model.skip_download and not hf_model.is_lfs:
            time_start = time.time()
            print(f"Requesting {hf_model.download_link}")
            download_file(hf_model.download_link, hf_model.append_path, thread_num=1, file_length=hf_model.size)
            time_end = time.time()
            print(f"Downloaded {hf_model.append_path} time is {time_end - time_start}")
        elif not hf_model.skip_download and hf_model.is_lfs:
            time_start = time.time()
            print(f"Requesting {hf_model.download_link}")
            download_file(hf_model.download_link, hf_model.append_path, thread_num=10, file_length=hf_model.size,check_sum_value=hf_model.lfs.oid)
            time_end = time.time()
            print(
                f"Downloaded {hf_model.append_path} time is {time_end - time_start},speed is {(hf_model.size / (time_end - time_start)) / 1024 / 1024} MB/s")
