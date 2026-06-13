import tarfile
# 请把 '你下载的文件名.tar.gz' 替换成实际的文件名
with tarfile.open('oa_comm_xml.PMC000xxxxxx.baseline.2026-01-23.tar.gz', 'r:gz') as tar:
    tar.extractall(path='./data/medical_papers/')