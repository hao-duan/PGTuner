import os
import urllib.request
import ftplib
from tqdm import tqdm

# download URL "http://..."
def report_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    progress = downloaded / total_size * 100
    print(f"download progress：{progress:.2f}%")


# download URL "ftp://..."
def download_file(ftp_address, ftp_path, filename, save_path):
    ftp = ftplib.FTP(ftp_address)
    ftp.login()
    ftp.cwd(ftp_path)

    total_size = ftp.size(filename)

    with open(save_path, 'wb') as local_file:
        downloaded = 0

        def write_to_file(data):
            nonlocal downloaded
            local_file.write(data)
            downloaded += len(data)
            progress = float(downloaded) / total_size * 100
            print(f"download progress：{progress:.2f}%", end='\r')

        ftp.retrbinary('RETR ' + filename, write_to_file)

    ftp.quit()
    print("\nFile has been downloaded to：", save_path)

if __name__ == '__main__':
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)

    # example1
    urldict = {'crawl': 'http://downloads.zjulearning.org.cn/data/crawl.tar.gz',
               'glove': 'http://downloads.zjulearning.org.cn/data/glove-100.tar.gz',
               'msong': 'https://drive.google.com/file/d/1UZ0T-nio8i2V8HetAx4-kt_FMK-GphHj/view',
               'uqv': 'https://drive.google.com/file/d/1HIdQSKGh7cfC7TnRvrA2dnkHBNkVHGsF/view?usp=sharing',
               'paper': 'https://drive.google.com/file/d/1t4b93_1Viuudzd5D3I6_9_9Guwm1vmTn/view'}

    filanemes = ['paper', 'msong']
    for fn in tqdm(filanemes, total = len(filanemes)):
        print(f'downloading: {fn}')
        data_path = os.path.join(parent_directory, '{}.tar.gz'.format(fn))
        url = urldict[fn]
        urllib.request.urlretrieve(url, data_path, reporthook=report_progress)

    # example2
    ftp_address = 'ftp.irisa.fr'
    ftp_path = '/local/texmex/corpus/'

    filename = 'gist.tar.gz'
    save_path = '../Data/gist.tar.gz'
    download_file(ftp_address, ftp_path, filename, save_path)

