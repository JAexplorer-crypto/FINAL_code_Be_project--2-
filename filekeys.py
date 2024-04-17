import hashlib

def get_keys(filePath):
    md5 = hashlib.md5()
    sha1 = hashlib.sha1()
    sha256 = hashlib.sha256()

    with open(filePath, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            md5.update(data)
            sha1.update(data)
            sha256.update(data)

    return [["MD5", md5.hexdigest()], ["SHA1", sha1.hexdigest()], ["SHA256", sha256.hexdigest()]]